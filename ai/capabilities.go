package ai

import (
	"encoding/json"
	"strings"

	"github.com/2found/2ai/ai/protocol"
)

// CapabilitiesFor returns adapter-level knowledge and conservative wire-policy
// limits for a provider/model pair. It intentionally stays sparse: this is not
// a model catalog, and unknown support preserves the optimistic request path.
// Live discovery may overlay these defaults with explicit model-level facts.
func CapabilitiesFor(vendor, model string) protocol.ModelCapabilities {
	supported := protocol.CapabilitySupported
	unsupported := protocol.CapabilityUnsupported
	switch NormalizeVendor(vendor) {
	case "openai":
		return protocol.ModelCapabilities{
			Tools: supported, ToolChoice: supported, ReasoningEffort: supported,
			ImageInput: supported, MaxInputImages: 200,
			StructuredOutput: supported, PromptCaching: supported,
			// AgentRay currently speaks Chat Completions here, not the stateful
			// Responses API.
			StatefulResponses: unsupported,
		}
	case VendorOpenAIResponses:
		return protocol.ModelCapabilities{
			Tools: supported, ToolChoice: supported, ReasoningEffort: supported,
			ImageInput: supported, MaxInputImages: 200,
			StructuredOutput: supported, PromptCaching: supported,
			StatefulResponses: supported,
		}
	case "anthropic":
		return protocol.ModelCapabilities{
			Tools: supported, ToolChoice: supported, ImageInput: supported, MaxInputImages: 90,
			StructuredOutput: supported, PromptCaching: supported,
			// The neutral request has no Anthropic thinking/tool-choice mapper yet.
			ReasoningEffort:   unsupported,
			StatefulResponses: unsupported,
		}
	case VendorOpenAICodex:
		return protocol.ModelCapabilities{
			Tools: supported, ToolChoice: supported, ReasoningEffort: supported,
			ImageInput: supported, MaxInputImages: 200,
		}
	case VendorGoogleAntigravity:
		maxOutput := 65536
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "claude-") {
			maxOutput = 64000
		}
		return protocol.ModelCapabilities{
			Tools: supported, ToolChoice: supported, ReasoningEffort: supported,
			MaxOutputTokens: maxOutput,
			// The current Cloud Code function-response adapter has no native
			// inline-image carrier. Say so explicitly instead of accepting parts
			// and degrading only after request-level policy has run.
			ImageInput: unsupported,
		}
	case VendorClaudeCode:
		return protocol.ModelCapabilities{
			Tools: supported, ToolChoice: supported, ImageInput: supported, MaxInputImages: 90,
			StructuredOutput: supported, PromptCaching: supported,
			ReasoningEffort:   unsupported,
			StatefulResponses: unsupported,
		}
	case VendorDevin:
		// The Cascade envelope carries tool definitions, images and thinking
		// effort; there is no structured-output or prompt-caching knob.
		return protocol.ModelCapabilities{
			Tools: supported, ToolChoice: supported, ReasoningEffort: supported,
			ImageInput: supported, MaxInputImages: 20,
			MaxOutputTokens:   128000,
			StatefulResponses: unsupported,
		}
	case "google":
		return protocol.ModelCapabilities{
			Tools: supported, ToolChoice: supported, ReasoningEffort: supported,
			ImageInput: supported, MaxInputImages: 200,
			StructuredOutput: supported, PromptCaching: supported,
			StatefulResponses: unsupported,
		}
	case "openrouter":
		return protocol.ModelCapabilities{
			ImageInput: supported, MaxInputImages: 90,
		}
	default:
		return protocol.ModelCapabilities{}
	}
}

func support(value bool) protocol.CapabilitySupport {
	if value {
		return protocol.CapabilitySupported
	}
	return protocol.CapabilityUnsupported
}

// discoveredCapabilities is the intentionally permissive subset of model-list
// metadata used by OpenAI-compatible routers and local engines. Every field is
// optional; only an explicit fact changes request behavior.
type discoveredCapabilities struct {
	SupportsTools             *bool                      `json:"supports_tools"`
	SupportsToolChoice        *bool                      `json:"supports_tool_choice"`
	SupportsReasoning         *bool                      `json:"supports_reasoning"`
	SupportsVision            *bool                      `json:"supports_vision"`
	SupportsStructuredOutputs *bool                      `json:"supports_structured_outputs"`
	SupportsPromptCaching     *bool                      `json:"supports_prompt_caching"`
	SupportsStatefulResponses *bool                      `json:"supports_stateful_responses"`
	MaxInputImages            *int                       `json:"max_input_images"`
	MaxImages                 *int                       `json:"max_images"`
	MaxOutputTokens           *int                       `json:"max_output_tokens"`
	MaxCompletionTokens       *int                       `json:"max_completion_tokens"`
	OutputTokenLimit          *int                       `json:"output_token_limit"`
	Capabilities              map[string]json.RawMessage `json:"capabilities"`
	SupportedParameters       []string                   `json:"supported_parameters"`
	InputModalities           []string                   `json:"input_modalities"`
	Architecture              struct {
		InputModalities []string `json:"input_modalities"`
	} `json:"architecture"`
}

func (d discoveredCapabilities) modelCapabilities() protocol.ModelCapabilities {
	var out protocol.ModelCapabilities
	// Some routers place booleans in a free-form capabilities object. Accept
	// common spellings, but never infer false from a missing key.
	for key, raw := range d.Capabilities {
		name := normalizeCapabilityName(key)
		if name == "maxinputimages" || name == "maximages" {
			if value, ok := explicitPositiveInt(raw); ok {
				out.MaxInputImages = value
			}
			continue
		}
		if name == "maxoutputtokens" || name == "maxcompletiontokens" || name == "outputtokenlimit" {
			if value, ok := explicitPositiveInt(raw); ok {
				out.MaxOutputTokens = value
			}
			continue
		}
		value, ok := explicitBool(raw)
		if !ok {
			continue
		}
		switch name {
		case "tools", "toolcalling", "functioncalling":
			out.Tools = support(value)
		case "toolchoice":
			out.ToolChoice = support(value)
		case "reasoning", "reasoningeffort", "thinking":
			out.ReasoningEffort = support(value)
		case "vision", "image", "imageinput":
			out.ImageInput = support(value)
		case "structuredoutput", "structuredoutputs", "responseformat", "jsonschema":
			out.StructuredOutput = support(value)
		case "promptcache", "promptcaching", "cachecontrol":
			out.PromptCaching = support(value)
		case "statefulresponses", "previousresponseid":
			out.StatefulResponses = support(value)
		}
	}

	// OpenRouter-style supported_parameters is positive evidence only: several
	// gateways return partial lists, so absence must not become unsupported.
	for _, parameter := range d.SupportedParameters {
		switch normalizeCapabilityName(parameter) {
		case "tools":
			out.Tools = protocol.CapabilitySupported
		case "toolchoice":
			out.ToolChoice = protocol.CapabilitySupported
		case "reasoning", "reasoningeffort":
			out.ReasoningEffort = protocol.CapabilitySupported
		case "responseformat", "structuredoutputs":
			out.StructuredOutput = protocol.CapabilitySupported
		case "promptcachekey", "cachecontrol":
			out.PromptCaching = protocol.CapabilitySupported
		case "previousresponseid":
			out.StatefulResponses = protocol.CapabilitySupported
		}
	}

	modalities := d.InputModalities
	if len(modalities) == 0 {
		modalities = d.Architecture.InputModalities
	}
	if len(modalities) > 0 {
		out.ImageInput = protocol.CapabilityUnsupported
		for _, modality := range modalities {
			if strings.EqualFold(strings.TrimSpace(modality), "image") {
				out.ImageInput = protocol.CapabilitySupported
				break
			}
		}
	}
	// Dedicated support fields are the least ambiguous signal and win over a
	// contradictory generic map/list from the same response.
	if d.SupportsTools != nil {
		out.Tools = support(*d.SupportsTools)
	}
	if d.SupportsToolChoice != nil {
		out.ToolChoice = support(*d.SupportsToolChoice)
	}
	if d.SupportsReasoning != nil {
		out.ReasoningEffort = support(*d.SupportsReasoning)
	}
	if d.SupportsVision != nil {
		out.ImageInput = support(*d.SupportsVision)
	}
	if d.SupportsStructuredOutputs != nil {
		out.StructuredOutput = support(*d.SupportsStructuredOutputs)
	}
	if d.SupportsPromptCaching != nil {
		out.PromptCaching = support(*d.SupportsPromptCaching)
	}
	if d.SupportsStatefulResponses != nil {
		out.StatefulResponses = support(*d.SupportsStatefulResponses)
	}
	if d.MaxImages != nil && *d.MaxImages > 0 {
		out.MaxInputImages = *d.MaxImages
	}
	if d.MaxInputImages != nil && *d.MaxInputImages > 0 {
		out.MaxInputImages = *d.MaxInputImages
	}
	out.MaxOutputTokens = firstPositivePointer(out.MaxOutputTokens, d.MaxOutputTokens, d.MaxCompletionTokens, d.OutputTokenLimit)
	return out
}

func firstPositivePointer(fallback int, values ...*int) int {
	for _, value := range values {
		if value != nil && *value > 0 {
			return *value
		}
	}
	return fallback
}

func explicitPositiveInt(raw json.RawMessage) (int, bool) {
	var value int
	if err := json.Unmarshal(raw, &value); err == nil && value > 0 {
		return value, true
	}
	return 0, false
}

func explicitBool(raw json.RawMessage) (bool, bool) {
	var value bool
	if err := json.Unmarshal(raw, &value); err == nil {
		return value, true
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		switch strings.ToLower(strings.TrimSpace(text)) {
		case "true", "supported", "yes":
			return true, true
		case "false", "unsupported", "no":
			return false, true
		}
	}
	return false, false
}

func normalizeCapabilityName(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	return strings.NewReplacer("_", "", "-", "", ".", "", " ", "").Replace(value)
}
