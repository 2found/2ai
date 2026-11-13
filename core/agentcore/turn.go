package agentcore

import (
	"fmt"
	"slices"
)

// unknownProviderImageLimit is the portable floor for an image-capable path
// whose vendor metadata does not publish a request-wide image cap. It matches
// the strictest mainstream compatible endpoints seen in OMP's provider budget;
// known adapters advertise their larger limits through ModelCapabilities.
const unknownProviderImageLimit = 5

// ModelCapabilityError reports a request that a provider/model pair is known
// not to accept. It is intentionally non-retryable: repeating the same rung
// cannot add a missing capability, while reason may still advance to a capable
// fallback rung.
type ModelCapabilityError struct {
	Provider string
	Model    string
	Feature  string
}

func (e *ModelCapabilityError) Error() string {
	return fmt.Sprintf("provider %s model %s does not support %s", e.Provider, e.Model, e.Feature)
}

// requestForCapabilities applies only explicit negative capability knowledge.
// Unknown means preserve the request, which keeps arbitrary compatible/local
// endpoints backward compatible. Provider hints may be removed here without
// weakening loop-owned enforcement (for example OutputSchema is still checked
// by Agent.outputValidator after the response).
func requestForCapabilities(provider LLMProvider, model string, discovered ModelCapabilities, req ChatRequest) (ChatRequest, error) {
	caps := CapabilitiesOf(provider, model).Overlay(discovered)
	if err := req.ToolChoice.Validate(); err != nil {
		return ChatRequest{}, fmt.Errorf("agentcore: invalid tool choice: %w", err)
	}
	for _, tool := range req.Tools {
		if err := tool.Strict.Validate(); err != nil {
			return ChatRequest{}, fmt.Errorf("agentcore: tool %q: %w", tool.Name, err)
		}
	}
	if len(req.Tools) > 0 && caps.Tools == CapabilityUnsupported {
		if req.ToolChoice.Mode == ToolChoiceNone {
			// "none" can be honored even on a text-only model by removing the
			// catalogue before network I/O. Other modes offered real capability
			// the caller may rely on, so those escalate as before.
			req.Tools = nil
			req.ToolChoice = ToolChoice{}
			req.ParallelToolCalls = nil
		} else {
			return ChatRequest{}, &ModelCapabilityError{Provider: provider.Name(), Model: model, Feature: "native tool calls"}
		}
	}
	if req.ToolChoice.Mode == ToolChoiceNamed {
		found := false
		for _, tool := range req.Tools {
			if tool.Name == req.ToolChoice.Name {
				found = true
				break
			}
		}
		if !found {
			return ChatRequest{}, fmt.Errorf("agentcore: named tool choice %q is not present in this request", req.ToolChoice.Name)
		}
	}
	if (req.ToolChoice.Mode == ToolChoiceRequired || req.ToolChoice.Mode == ToolChoiceNamed) && len(req.Tools) == 0 {
		return ChatRequest{}, fmt.Errorf("agentcore: tool choice %q requires at least one tool", req.ToolChoice.Mode)
	}
	if caps.ToolChoice == CapabilityUnsupported {
		switch req.ToolChoice.Mode {
		case ToolChoiceNone:
			// Enforce the caller's safety intent without relying on a wire feature.
			req.Tools = nil
			req.ToolChoice = ToolChoice{}
			req.ParallelToolCalls = nil
		case ToolChoiceDefault, ToolChoiceAuto:
			req.ToolChoice = ToolChoice{}
			req.ParallelToolCalls = nil
		default:
			return ChatRequest{}, &ModelCapabilityError{Provider: provider.Name(), Model: model, Feature: "forced tool choice"}
		}
	}
	if caps.ReasoningEffort == CapabilityUnsupported {
		req.ReasoningEffort = ""
	}
	if caps.StructuredOutput == CapabilityUnsupported {
		req.OutputSchema = nil
	}
	if caps.PromptCaching == CapabilityUnsupported {
		req.CacheKey = ""
		req.CacheRetention = ""
	}
	if caps.MaxOutputTokens > 0 && req.MaxTokens > caps.MaxOutputTokens {
		req.MaxTokens = caps.MaxOutputTokens
	}
	if caps.ImageInput == CapabilityUnsupported {
		req.Messages = withoutImageContent(req.Messages)
	} else {
		limit := caps.MaxInputImages
		if limit <= 0 {
			limit = unknownProviderImageLimit
		}
		req.Messages = clampImageContent(req.Messages, limit)
	}
	return req, nil
}

// clampImageContent drops the oldest image parts until the complete outgoing
// request fits the active provider/model limit. It is copy-on-write: escalation
// to a later rung starts from the unmodified canonical transcript and may apply
// a different limit. Text parts and tool-call linkage stay intact, and every
// affected message gets a visible breadcrumb instead of losing images silently.
func clampImageContent(messages []Message, limit int) []Message {
	if limit < 0 {
		limit = 0
	}
	total := 0
	for _, message := range messages {
		for _, part := range message.ContentParts {
			if part.Type == ContentPartImage {
				total++
			}
		}
	}
	drop := total - limit
	if drop <= 0 {
		return messages
	}

	var out []Message
	for i, message := range messages {
		if drop == 0 {
			break
		}
		omitted := 0
		kept := make([]ContentPart, 0, len(message.ContentParts))
		for _, part := range message.ContentParts {
			if part.Type == ContentPartImage && drop > 0 {
				drop--
				omitted++
				continue
			}
			kept = append(kept, part)
		}
		if omitted == 0 {
			continue
		}
		if out == nil {
			out = append([]Message(nil), messages...)
		}
		replacement := message
		replacement.ContentParts = kept
		note := fmt.Sprintf("[%d older image attachment(s) omitted: provider request limit is %d]", omitted, limit)
		if replacement.Content == "" {
			replacement.Content = note
		} else {
			replacement.Content += "\n" + note
		}
		out[i] = replacement
	}
	if out == nil {
		return messages
	}
	return out
}

// withoutImageContent makes an explicit negative image capability visible to
// the model instead of silently dropping tool output. It is copy-on-write so a
// failed rung cannot mutate the request later escalated to a vision-capable
// fallback model.
func withoutImageContent(messages []Message) []Message {
	var out []Message
	for i, message := range messages {
		images := 0
		kept := make([]ContentPart, 0, len(message.ContentParts))
		for _, part := range message.ContentParts {
			if part.Type == ContentPartImage {
				images++
				continue
			}
			kept = append(kept, part)
		}
		if images == 0 {
			continue
		}
		if out == nil {
			out = append([]Message(nil), messages...)
		}
		replacement := message
		replacement.ContentParts = kept
		note := fmt.Sprintf("[%d image output(s) omitted: provider/model path does not support image input]", images)
		if replacement.Content == "" {
			replacement.Content = note
		} else {
			replacement.Content += "\n" + note
		}
		out[i] = replacement
	}
	if out == nil {
		return messages
	}
	return out
}

// retainedTranscript is what a completed compaction stores on its durable
// entry: the transcript it left behind, minus the run's own system prompt.
//
// The prompt is excluded because it is not part of the conversation — every run
// rebuilds it from the definition, recalled memory, and the extensions' prompt
// sections, then prepends it (a resumed run re-states the same contract). The
// durable log has never carried it, so storing it here would give a resumed run
// two system prompts: the stored one and the freshly derived one.
//
// Matching on the exact prompt content rather than "drop the first message" is
// what keeps this honest when the leading system message is something else —
// a goal pin promoted into the head by an earlier compaction, or a caller whose
// seed history opens with its own system message. Those belong to the
// conversation and must survive.
func retainedTranscript(messages []Message, system string) []Message {
	if len(messages) > 0 && system != "" &&
		messages[0].Role == RoleSystem && messages[0].Content == system {
		messages = messages[1:]
	}
	// Clone: the caller keeps mutating res.Messages, and a durable entry must be
	// a snapshot of the moment it was written.
	return slices.Clone(messages)
}

// isTruncatedStop reports whether the provider ended the response at its output
// limit rather than by choice. Stop reasons pass through vendor-verbatim, so
// this is a set: OpenAI-wire "length", Anthropic's "max_tokens", Codex's
// "incomplete". A truncated message's tool calls may carry silently incomplete
// arguments and are never executed (see the dispatch guard in loop.go).
func isTruncatedStop(reason string) bool {
	switch reason {
	case "length", "max_tokens", "incomplete":
		return true
	}
	return false
}
