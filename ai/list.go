package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

const (
	defaultGoogleBaseURL = "https://generativelanguage.googleapis.com"
)

// HTTPDoer is the injected HTTP seam so list-models and Chat share a client
// and tests can point at httptest servers.
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

func defaultHTTP() HTTPDoer {
	return &http.Client{Timeout: 30 * time.Second}
}

func doJSON(ctx context.Context, client HTTPDoer, req *http.Request) ([]byte, int, error) {
	if client == nil {
		client = defaultHTTP()
	}
	resp, err := client.Do(req.WithContext(ctx))
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	var data []byte
	if resp.StatusCode >= 400 {
		data = readProviderErrorBody(resp.Body)
	} else {
		data, _ = io.ReadAll(resp.Body)
	}
	return data, resp.StatusCode, nil
}

// listedModel is one entry from a vendor's list-models response: the id, plus
// the context window when the vendor reported one. A zero window means "this
// vendor did not say", which is a different fact from "this model has no
// window" — callers resolve it through ContextWindowFor rather than treating it
// as a number.
type listedModel struct {
	ID            string
	ContextWindow int
	Capabilities  protocol.ModelCapabilities
}

// listOpenAIModels calls GET {base}/models (OpenAI and OpenAI-compatible).
//
// OpenAI proper reports no context window. OpenAI-*compatible* servers usually
// do, under a name of their own choosing: OpenRouter uses context_length (and
// repeats it under top_provider), vLLM uses max_model_len, and several routers
// use context_window or max_context_length. Reading all of them costs nothing
// and is the difference between a real window and a guess for exactly the
// vendors a workspace is most likely to point at.
func listOpenAIModels(ctx context.Context, client HTTPDoer, baseURL, apiKey string) ([]listedModel, error) {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if base == "" {
		base = defaultOpenAIBaseURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/models", nil)
	if err != nil {
		return nil, err
	}
	setOptionalBearerAuth(req, apiKey)
	req.Header.Set("Accept", "application/json")
	data, status, err := doJSON(ctx, client, req)
	if err != nil {
		return nil, err
	}
	if status >= 400 {
		return nil, fmt.Errorf("list models: status %d: %s", status, strings.TrimSpace(string(data)))
	}
	var decoded struct {
		Data []struct {
			ID               string `json:"id"`
			ContextLength    int    `json:"context_length"`
			ContextWindow    int    `json:"context_window"`
			MaxModelLen      int    `json:"max_model_len"`
			MaxContextLength int    `json:"max_context_length"`
			TopProvider      struct {
				ContextLength int `json:"context_length"`
			} `json:"top_provider"`
			discoveredCapabilities
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, fmt.Errorf("list models: decode: %w", err)
	}
	out := make([]listedModel, 0, len(decoded.Data))
	for _, m := range decoded.Data {
		id := strings.TrimSpace(m.ID)
		if id == "" {
			continue
		}
		out = append(out, listedModel{
			ID: id, ContextWindow: firstPositive(
				m.ContextLength, m.ContextWindow, m.MaxModelLen, m.MaxContextLength, m.TopProvider.ContextLength,
			),
			Capabilities: m.discoveredCapabilities.modelCapabilities(),
		})
	}
	return out, nil
}

// firstPositive returns the first value above zero, so a vendor that spells the
// context window differently still lands on one number and an absent field stays
// absent rather than becoming a zero-length window.
func firstPositive(vals ...int) int {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return 0
}

// listAnthropicModels calls GET {base}/v1/models.
//
// The Messages API's model list reports id, display name, and type — no context
// window — so these entries resolve through ContextWindowFor. context_window is
// read anyway so that if Anthropic ever adds it, the live value wins over the
// table without another change here.
func listAnthropicModels(ctx context.Context, client HTTPDoer, baseURL, apiKey string) ([]listedModel, error) {
	return listModelsV1(ctx, client, baseURL, func(req *http.Request) {
		req.Header.Set("x-api-key", apiKey)
	})
}

// listModelsV1 is the shared GET {base}/v1/models fetch: the Anthropic wire
// list endpoint, used by both the x-api-key provider and the claude-code
// OAuth surface (which differs only in the headers it sets).
func listModelsV1(ctx context.Context, client HTTPDoer, baseURL string, auth func(*http.Request)) ([]listedModel, error) {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if base == "" {
		base = defaultAnthropicBaseURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	auth(req)
	req.Header.Set("anthropic-version", anthropicVersion)
	req.Header.Set("Accept", "application/json")
	data, status, err := doJSON(ctx, client, req)
	if err != nil {
		return nil, err
	}
	if status >= 400 {
		return nil, &protocol.ProviderError{Provider: "anthropic", Status: status, Message: strings.TrimSpace(string(data))}
	}
	var decoded struct {
		Data []struct {
			ID              string `json:"id"`
			ContextWindow   int    `json:"context_window"`
			MaxOutputTokens int    `json:"max_output_tokens"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, fmt.Errorf("list models: decode: %w", err)
	}
	out := make([]listedModel, 0, len(decoded.Data))
	for _, m := range decoded.Data {
		if id := strings.TrimSpace(m.ID); id != "" {
			out = append(out, listedModel{ID: id, ContextWindow: m.ContextWindow,
				Capabilities: protocol.ModelCapabilities{MaxOutputTokens: m.MaxOutputTokens}})
		}
	}
	return out, nil
}

// listGoogleModels calls the Gemini models.list API. A base URL that already
// speaks the OpenAI-compatible surface (path contains /openai) uses GET /models
// instead.
//
// Gemini reports inputTokenLimit per model, which is exactly the number the
// compaction budget wants: the input side of the window, with the output
// allowance (outputTokenLimit) already excluded.
func listGoogleModels(ctx context.Context, client HTTPDoer, baseURL, apiKey string) ([]listedModel, error) {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if strings.Contains(base, "/openai") {
		return listOpenAIModels(ctx, client, base, apiKey)
	}
	if base == "" {
		base = defaultGoogleBaseURL
	}
	endpoint := base
	switch {
	case strings.HasSuffix(endpoint, "/models"):
		// already a list-models URL
	case strings.Contains(endpoint, "/v1beta"):
		endpoint += "/models"
	default:
		endpoint += "/v1beta/models"
	}
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, err
	}
	q := u.Query()
	if apiKey != "" {
		q.Set("key", apiKey)
	}
	u.RawQuery = q.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	if apiKey != "" {
		req.Header.Set("x-goog-api-key", apiKey)
	}
	data, status, err := doJSON(ctx, client, req)
	if err != nil {
		return nil, err
	}
	if status >= 400 {
		return nil, fmt.Errorf("list models: status %d: %s", status, strings.TrimSpace(string(data)))
	}
	var decoded struct {
		Models []struct {
			Name             string `json:"name"`
			InputTokenLimit  int    `json:"inputTokenLimit"`
			OutputTokenLimit int    `json:"outputTokenLimit"`
		} `json:"models"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, fmt.Errorf("list models: decode: %w", err)
	}
	out := make([]listedModel, 0, len(decoded.Models))
	for _, m := range decoded.Models {
		id := strings.TrimSpace(m.Name)
		id = strings.TrimPrefix(id, "models/")
		if id != "" {
			out = append(out, listedModel{ID: id, ContextWindow: m.InputTokenLimit,
				Capabilities: protocol.ModelCapabilities{MaxOutputTokens: m.OutputTokenLimit}})
		}
	}
	return out, nil
}

func listModelsForVendor(ctx context.Context, client HTTPDoer, vendor, baseURL, apiKey string) ([]listedModel, error) {
	switch NormalizeVendor(vendor) {
	case "anthropic":
		return listAnthropicModels(ctx, client, baseURL, apiKey)
	case "google":
		return listGoogleModels(ctx, client, baseURL, apiKey)
	default:
		// openai, openai-compat, and any other OpenAI-compatible vendor
		return listOpenAIModels(ctx, client, baseURL, apiKey)
	}
}
