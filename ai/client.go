package ai

import (
	"fmt"
	"strings"

	"github.com/2found/2ai/ai/protocol"
)

// ClientSpec is the resolved configuration for constructing a wire client:
// vendor name, decrypted key, and an optional base-URL override.
//
// It is deliberately smaller than Spec: a ClientSpec yields a bare
// protocol.LLMProvider (Chat/Stream and nothing else), which is what a run
// needs. Spec adds the identity and live model list that a workspace-managed
// provider needs, and builds on this.
type ClientSpec struct {
	Name    string // "openai" | "anthropic" | an OAuth vendor | any OpenAI-compatible vendor
	APIKey  string
	BaseURL string
	Compat  Compat // optional; zero value falls back to the vendor default
	// OpenAIWire chooses the transport for an OpenAI provider identity. Empty
	// and OpenAIWireChat use Chat Completions; OpenAIWireResponses uses the
	// public Responses API while Name() remains "openai". Keeping provider
	// identity separate from wire format lets one workspace row serve models
	// with different API contracts, matching the model.api split in OMP.
	OpenAIWire OpenAIWire
	// SessionScope distinguishes independently configured provider rows that use
	// the same vendor/base URL. OAuth account-rotation state must never confuse
	// two sibling pools just because both speak (for example) openai-codex.
	SessionScope string
	// TokenSource is the OAuth account pool a subscription vendor
	// (claude-code | openai-codex | google-antigravity) draws per-request
	// credentials from. Required for those vendors, ignored by the rest.
	TokenSource TokenSource
}

// OpenAIWire is the API dialect used behind an OpenAI provider identity.
// Provider identity owns credentials and routing; the selected model owns the
// wire. Explicit openai-responses vendors remain supported for operators who
// want the wire fixed for the whole provider row.
type OpenAIWire string

const (
	OpenAIWireChat      OpenAIWire = "chat-completions"
	OpenAIWireResponses OpenAIWire = "responses"
)

// NewClient resolves a ClientSpec into an protocol.LLMProvider. Adding a
// vendor is additive here — a new case (or, for OpenAI-compatible vendors, just
// a compat entry + base_url) — and never requires touching the agent loop.
func NewClient(spec ClientSpec) (protocol.LLMProvider, error) {
	name := strings.ToLower(strings.TrimSpace(spec.Name))
	if v := NormalizeOAuthVendor(name); v != "" {
		name = v
	}
	switch name {
	case "", "openai":
		switch spec.OpenAIWire {
		case "", OpenAIWireChat:
		case OpenAIWireResponses:
			p := NewOpenAIResponsesProvider(spec.APIKey, spec.BaseURL)
			p.Vendor = "openai"
			return p, nil
		default:
			return nil, fmt.Errorf("ai: unknown OpenAI wire %q", spec.OpenAIWire)
		}
		compat := spec.Compat
		if compat.MaxTokensField == "" {
			compat = DefaultCompat()
		}
		return NewOpenAIProvider(spec.APIKey, spec.BaseURL, compat), nil
	case VendorOpenAIResponses:
		return NewOpenAIResponsesProvider(spec.APIKey, spec.BaseURL), nil
	case "anthropic":
		return NewAnthropicProvider(spec.APIKey, spec.BaseURL), nil
	case VendorClaudeCode:
		// Claude Code speaks the Messages API but authenticates with a pooled
		// OAuth grant: Bearer auth + the CLI fingerprint, token drawn per request.
		inner := NewAnthropicProvider("", spec.BaseURL)
		inner.OAuth = true
		return newPooledProvider(VendorClaudeCode, inner, spec.TokenSource, spec.SessionScope)
	case VendorOpenAICodex:
		inner := NewCodexProvider()
		if b := strings.TrimSpace(spec.BaseURL); b != "" {
			inner.BaseURL = strings.TrimRight(b, "/")
		}
		return newPooledProvider(VendorOpenAICodex, inner, spec.TokenSource, spec.SessionScope)
	case VendorGoogleAntigravity:
		inner := NewAntigravityProvider()
		if b := strings.TrimSpace(spec.BaseURL); b != "" {
			inner.BaseURL = strings.TrimRight(b, "/")
		}
		return newPooledProvider(VendorGoogleAntigravity, inner, spec.TokenSource, spec.SessionScope)
	case VendorDevin:
		inner := NewDevinProvider()
		if b := strings.TrimSpace(spec.BaseURL); b != "" {
			inner.BaseURL = strings.TrimRight(b, "/")
		}
		return newPooledProvider(VendorDevin, inner, spec.TokenSource, spec.SessionScope)
	case VendorXaiOAuth:
		inner := NewOpenAIProvider("", "https://api.x.ai/v1", DefaultCompat())
		if spec.BaseURL != "" {
			inner.BaseURL = strings.TrimRight(spec.BaseURL, "/")
		}
		inner.Vendor = VendorXaiOAuth
		return newPooledProvider(VendorXaiOAuth, &xaiOAuthProvider{inner}, spec.TokenSource, spec.SessionScope)
	case "deepseek", "opencode", "opencode-zen", "opencode-go":
		if spec.BaseURL == "" {
			spec.BaseURL = ProviderDefaultURL(name)
		}
		compat := spec.Compat
		if compat.MaxTokensField == "" {
			compat = DefaultCompat()
		}
		p := NewOpenAIProvider(spec.APIKey, spec.BaseURL, compat)
		p.Vendor = name
		return p, nil
	case "google", "gemini":
		// Gemini on Google's OpenAI-compatible surface. An explicit BaseURL
		// overrides the default endpoint (e.g. a regional proxy).
		p := NewGeminiProvider(spec.APIKey)
		if b := strings.TrimSpace(spec.BaseURL); b != "" {
			p.BaseURL = strings.TrimRight(b, "/")
		}
		return p, nil
	default:
		// OpenAI-compatible vendors are config, not code: route them through the
		// OpenAI provider with a caller-supplied base_url + compat. The provider
		// keeps the vendor's identity so traces and per-turn key refresh attribute
		// to the vendor's tier, not "openai".
		if spec.Compat.MaxTokensField != "" && strings.TrimSpace(spec.BaseURL) != "" {
			p := NewOpenAIProvider(spec.APIKey, spec.BaseURL, spec.Compat)
			p.Vendor = strings.ToLower(strings.TrimSpace(spec.Name))
			return p, nil
		}
		return nil, fmt.Errorf("ai: unknown provider %q", spec.Name)
	}
}
