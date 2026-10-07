package ai

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"net/url"
	"slices"
	"strings"
)

// NativeClient binds host credentials and endpoint defaults to the lossless
// stream API. Credentials stay in the client, never in checkpointed models.
type NativeClient struct {
	spec                    ClientSpec
	api, provider, endpoint string
	compat                  map[string]any
}

func NewNativeClient(spec ClientSpec) (*NativeClient, error) {
	c := &NativeClient{spec: spec}
	vendor := NormalizeVendor(spec.Name)
	// OAuth alias ids (e.g. "devin-agent", "claudecode") name the same vendors.
	if alias := NormalizeOAuthVendor(vendor); alias != "" {
		vendor = alias
	}
	switch vendor {
	case VendorXaiOAuth:
		c.api, c.provider, c.endpoint = "openai-completions", vendor, strings.TrimRight(spec.BaseURL, "/")
		if c.endpoint == "" {
			c.endpoint = "https://api.x.ai/v1"
		}
	case VendorGoogleAntigravity:
		c.api, c.provider, c.endpoint = vendor, vendor, strings.TrimRight(spec.BaseURL, "/")
		defaults := c.endpoint == ""
		if defaults {
			c.endpoint = "https://daily-cloudcode-pa.googleapis.com"
		}
		c.compat = map[string]any{"antigravityDefaultEndpoints": defaults}
	case VendorClaudeCode:
		c.api, c.provider, c.endpoint = "anthropic-messages", vendor, strings.TrimRight(spec.BaseURL, "/")
		if c.endpoint == "" {
			c.endpoint = "https://api.anthropic.com"
		}
	case VendorOpenAICodex:
		c.api, c.provider, c.endpoint = "openai-codex-responses", vendor, strings.TrimRight(spec.BaseURL, "/")
		if c.endpoint == "" {
			c.endpoint = "https://chatgpt.com/backend-api"
		}
	case VendorDevin:
		c.api, c.provider, c.endpoint = "devin-agent", vendor, strings.TrimRight(spec.BaseURL, "/")
		if c.endpoint == "" {
			c.endpoint = DevinDefaultBaseURL
		}
	default:
		// Share established vendor/compat defaults with the existing client factory;
		// execution below uses only native transports.
		client, err := NewClient(spec)
		if err != nil {
			return nil, err
		}
		c.provider = client.Name()
		switch p := client.(type) {
		case *OpenAIProvider:
			c.api, c.endpoint = "openai-completions", p.BaseURL
			c.compat = map[string]any{"maxTokensField": p.Compat.MaxTokensField}
		case *OpenAIResponsesProvider:
			c.api, c.endpoint = "openai-responses", strings.TrimSuffix(p.BaseURL, "/responses")
		case *AnthropicProvider:
			c.api, c.endpoint = "anthropic-messages", p.BaseURL
		default:
			return nil, errors.New("ai: client has no native transport")
		}
	}
	if IsOAuthVendor(vendor) && spec.TokenSource == nil {
		return nil, errors.New("ai: native OAuth client requires an account pool")
	}
	endpoint, err := url.Parse(c.endpoint)
	if err != nil || endpoint.Host == "" || endpoint.User != nil || (endpoint.Scheme != "https" && endpoint.Scheme != "http") {
		return nil, errors.New("ai: native endpoint must be an HTTP(S) URL without credentials")
	}
	return c, nil
}

// Candidate resolves a model for the AI fallback provider. ContextWindow is a
// host limit; the model metadata and opaque native messages survive checkpoints.
func (c *NativeClient) Candidate(id string, contextWindow int) (FallbackCandidate, error) {
	if strings.TrimSpace(id) == "" {
		return FallbackCandidate{}, errors.New("ai: native model ID is required")
	}
	capabilities := CapabilitiesFor(c.provider, id)
	maxTokens := capabilities.MaxOutputTokens
	if maxTokens <= 0 {
		maxTokens = 16384
	}
	model, err := json.Marshal(map[string]any{"id": id, "name": id, "agentrayProviderId": c.spec.SessionScope, "provider": c.provider, "api": c.api, "baseUrl": c.endpoint, "contextWindow": contextWindow, "maxTokens": maxTokens, "reasoning": true, "input": []string{"text", "image"}, "cost": map[string]float64{}, "compat": c.compat})
	if err != nil {
		return FallbackCandidate{}, err
	}
	return FallbackCandidate{Model: slices.Clone(model), Stream: func(ctx context.Context, requested json.RawMessage, transcript TranscriptContext, options map[string]any) (*AssistantMessageEventStream, error) {
		// A bound credential must never be redirected by model mutation.
		if string(requested) != string(model) {
			return nil, &PreparationError{Cause: errors.New("ai: native client model binding changed")}
		}
		controls := maps.Clone(options)
		if controls == nil {
			controls = map[string]any{}
		}
		controls["apiKey"] = c.spec.APIKey
		return (NativeProvider{Tokens: c.spec.TokenSource}).Stream(ctx, requested, transcript, controls)
	}}, nil
}
