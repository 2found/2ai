package ai

import (
	"context"
	"strings"

	"github.com/2found/2ai/ai/protocol"
)

// Model is one entry from a provider's live list-models response. IDs are
// whatever the vendor returned — never a hardcoded catalog.
type Model struct {
	// API and Provider identify Pi's wire implementation and provider separately.
	// Input declares supported modalities for transcript replay. The workspace
	// catalog fields below remain available to existing Go callers.
	API            string   `json:"api,omitempty"`
	Provider       string   `json:"provider,omitempty"`
	Input          []string `json:"input,omitempty"`
	ProviderID     string   `json:"provider_id"`
	ProviderVendor string   `json:"provider_vendor"`
	ProviderName   string   `json:"provider_name"`
	ID             string   `json:"id"`
	// ContextWindow is the model's input window in tokens: the vendor's own
	// figure when its list-models response carried one, otherwise this package's
	// fallback, otherwise 0 for "unknown". It is what the compaction budget is
	// capped against, so 0 must stay distinguishable from a real number.
	ContextWindow int `json:"context_window,omitempty"`
	// Capabilities is sparse tri-state metadata. Missing fields mean unknown,
	// not unsupported; explicit live discovery values refine adapter defaults.
	Capabilities protocol.ModelCapabilities `json:"capabilities,omitempty"`
}

// Provider is the runtime unit: identity, auth, live model list, Chat/Stream.
type Provider interface {
	protocol.LLMProvider
	ID() string
	Vendor() string
	DisplayName() string
	BaseURL() string
	APIKey() string
	ListModels(ctx context.Context) ([]Model, error)
}

// Spec constructs a provider. Vendor is openai | openai-responses | anthropic |
// google | gemini, an OAuth subscription vendor (claude-code | openai-codex |
// google-antigravity), or any OpenAI-compatible name (which requires BaseURL).
// ID is the caller's stable handle (a workspace provider row id); empty ID
// falls back to Vendor.
type Spec struct {
	ID      string
	Vendor  string
	Name    string
	APIKey  string
	BaseURL string
	HTTP    HTTPDoer
	// Compat selects OpenAI-compatible wire details such as max_tokens versus
	// max_completion_tokens. Zero uses the vendor default.
	Compat Compat
	// TokenSource is the OAuth account pool a subscription vendor draws
	// per-request credentials from. Required for OAuth vendors, ignored by
	// the rest.
	TokenSource TokenSource
}

// NormalizeVendor maps aliases onto the built-in vendor ids.
func NormalizeVendor(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", "openai":
		return "openai"
	case "openai-responses", "openai-response", "responses":
		return VendorOpenAIResponses
	case "anthropic":
		return "anthropic"
	case "google", "gemini":
		return "google"
	case "openai-compat", "openai-compatible", "compat":
		return "openai-compat"
	default:
		return strings.ToLower(strings.TrimSpace(v))
	}
}

// APIKeyOptional reports OpenAI-compatible engines whose normal local/self-
// hosted deployment accepts unauthenticated requests. These are explicit
// vendor identities rather than a hostname heuristic so the same configuration
// works on a laptop (localhost) and a server (a private service/DNS name). A
// supplied key is still sent, which supports secured deployments of the same
// engines.
func APIKeyOptional(vendor string) bool {
	switch strings.ToLower(strings.TrimSpace(vendor)) {
	case "ollama", "lm-studio", "lmstudio", "llama.cpp", "llama-cpp", "vllm", "localai", "local-ai", "litellm":
		return true
	default:
		return false
	}
}

// chatViaStream drains a streaming-only provider's delta channel into a
// ChatResponse. Codex and Antigravity have no non-streaming wire, so their
// Chat is this loop — kept once here so drain semantics (mid-stream error,
// usage capture, stop reason) can't drift between two copies.
func chatViaStream(ctx context.Context, p protocol.LLMProvider, req protocol.ChatRequest) (protocol.ChatResponse, error) {
	ch, err := p.Stream(ctx, req)
	if err != nil {
		return protocol.ChatResponse{}, err
	}
	var resp protocol.ChatResponse
	resp.Message.Role = protocol.RoleAssistant
	for d := range ch {
		if d.Err != nil {
			return protocol.ChatResponse{}, d.Err
		}
		resp.Message.Content += d.ContentDelta
		if d.ToolCall != nil {
			resp.Message.ToolCalls = append(resp.Message.ToolCalls, *d.ToolCall)
		}
		if d.Usage.InputTokens != 0 || d.Usage.OutputTokens != 0 || d.Usage.CacheReadTokens != 0 {
			resp.Usage = d.Usage
		}
		if d.StopReason != "" {
			resp.StopReason = d.StopReason
		}
	}
	return resp, nil
}
