package agentcore

import (
	"context"
	"fmt"
	"slices"
)

// Legacy stream-fixture frame bound, pending migration of the old transport tests.
const frameBytes = 4 << 10

// streamAbort is the sentinel a StreamInterceptor raises to cut a turn's
// provider call mid-flight. It is NOT a provider failure: callRung must not
// spend a retry on it and reason must not escalate it — the loop catches it,
// appends the injection, and re-issues the same request itself.
type streamAbort struct {
	// inject is the correction the retried turn must see.
	inject []Message
	// usage is whatever the aborted attempt still spent: the tokens were
	// generated and billed even though the message is discarded, so the run's
	// accounting folds it in rather than letting a rule fire for free.
	usage Usage
}

func (e *streamAbort) Error() string { return "stream aborted by interceptor" }

// committedStreamError marks a provider stream that failed after content was
// already delivered to the caller's sink. Retrying that request — on the same
// model or an escalation rung — is no longer replay-safe: the replacement
// answer would be appended to text the caller has already rendered. The cause
// stays wrapped for errors.Is/errors.As, and usage preserves whatever the
// failed attempt reported so a partial response is not free.
type committedStreamError struct {
	cause error
	usage Usage
}

func (e *committedStreamError) Error() string {
	return fmt.Sprintf("provider stream failed after emitting output: %v", e.cause)
}

func (e *committedStreamError) Unwrap() error { return e.cause }

// maxStreamAbortsPerTurn backstops a StreamInterceptor that never stops
// matching: past it the turn's provider call is retried once with interception
// disabled, so the stream completes unmodified. A guard that can spin the
// provider forever is an availability bug wearing a policy costume — the
// plugin's own per-turn cap is the real bound; this only covers a broken one.
const maxStreamAbortsPerTurn = 8

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

// streamTurn consumes the provider's delta channel for one turn, forwarding
// content fragments to the sink as they arrive when no interceptor is installed
// and holding an intercepted attempt until it is accepted. It accumulates the
// full assistant message (text + tool calls + usage) so the Act path is identical
// to the non-streaming turn. frame, when non-nil, receives a throttled snapshot of
// the text accumulated SO FAR — first delta, then per frameBytes — for the
// durable partial-frame record; it is per-attempt, so a retried stream starts
// its snapshots fresh rather than mixing two attempts' text.
//
// When stream interceptors are installed the provider call runs on a cancelable
// child context: the first interceptor to abort cancels it — cutting the HTTP
// stream mid-token — and the partial message is discarded into the streamAbort
// sentinel. The channel is drained after cancel so a provider that does not
// select on ctx.Done (the scripted test providers) still finishes its goroutine
// instead of leaking it against a full buffer.
func (a *Agent) streamTurn(ctx context.Context, provider LLMProvider, req ChatRequest, sink StreamSink, streams []StreamInterceptor, frame func(snapshot string)) (ChatResponse, error) {
	streamCtx := ctx
	cancel := context.CancelFunc(nil)
	if len(streams) > 0 {
		streamCtx, cancel = context.WithCancel(ctx)
		defer cancel()
	}
	ch, err := provider.Stream(streamCtx, req)
	if err != nil {
		return ChatResponse{}, err
	}
	msg := Message{Role: RoleAssistant}
	var resp ChatResponse
	var heldTokens []string
	committed := false
	terminal := false
	sinceFrame := 0
	for d := range ch {
		// Some providers attach their last known usage to the error delta. Merge
		// it before inspecting Err so a failed partial stream is still accounted.
		resp.Usage = mergeUsage(resp.Usage, d.Usage)
		if d.Err != nil {
			if committed {
				return ChatResponse{}, &committedStreamError{cause: d.Err, usage: resp.Usage}
			}
			return ChatResponse{}, d.Err
		}
		if d.ContentDelta != "" {
			msg.Content += d.ContentDelta
			if len(streams) > 0 {
				if dec := interceptStreamDelta(ctx, streams, msg.Content); dec.Abort {
					cancel()
					for range ch {
					}
					return ChatResponse{}, &streamAbort{inject: dec.Inject, usage: resp.Usage}
				}
			}
			if len(streams) > 0 {
				// A rule can match a later chunk, after earlier chunks already looked
				// harmless. Hold the attempt until it completes so an aborted draft
				// never leaks a prefix into the visible stream. Streams without rules
				// retain true token-by-token delivery.
				heldTokens = append(heldTokens, d.ContentDelta)
			} else {
				sink(StreamEvent{Type: StreamToken, Token: d.ContentDelta})
				committed = true
			}
			if frame != nil {
				sinceFrame += len(d.ContentDelta)
				if msg.Content == d.ContentDelta || sinceFrame >= frameBytes {
					frame(msg.Content)
					sinceFrame = 0
				}
			}
		}
		if d.ToolCall != nil {
			msg.ToolCalls = append(msg.ToolCalls, *d.ToolCall)
		}
		if d.ReasoningBlock != nil {
			msg.ReasoningBlocks = append(msg.ReasoningBlocks, *d.ReasoningBlock)
		}
		// Usage is not the Done delta's private property: providers report it in
		// pieces (input up front, output at the end) and as running totals, so
		// take the newest non-zero value of each field wherever it arrives. A
		// last-write-wins assignment on Done alone zeroes the whole turn's spend
		// for any provider that reports early — and the run's budget gate meters
		// on that number, so the loss is invisible until a run overshoots.
		if d.Done {
			resp.StopReason = d.StopReason
			terminal = true
		}
	}
	// A provider wrapper may close its channel while propagating cancellation.
	// Never turn that closure into a successful empty assistant response. Check
	// before releasing held interceptor output so cancelled drafts stay hidden.
	// Once a terminal event was received, however, completion won the race: a
	// sibling cancellation that lands a moment later must not erase a fully
	// delivered response and make durable child work run twice on recovery.
	if err := streamCtx.Err(); err != nil && !terminal {
		if committed {
			return ChatResponse{}, &committedStreamError{cause: err, usage: resp.Usage}
		}
		return ChatResponse{}, err
	}
	for _, token := range heldTokens {
		sink(StreamEvent{Type: StreamToken, Token: token})
	}
	resp.Message = msg
	return resp, nil
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
