package ai

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/2found/2ai/ai/protocol"
)

// New builds a provider from a Spec. Wire routing goes through NewClient, so a
// run and a list-models call share credentials and base URL by construction
// rather than by two switches that have to agree.
func New(spec Spec) (Provider, error) {
	vendor := NormalizeVendor(spec.Vendor)
	if v := NormalizeOAuthVendor(vendor); v != "" {
		vendor = v
	}
	if spec.BaseURL=="" && (vendor=="deepseek" || vendor=="opencode" || vendor=="opencode-zen" || vendor=="opencode-go") { spec.BaseURL=ProviderDefaultURL(vendor) }
	id := strings.TrimSpace(spec.ID)
	if id == "" {
		id = vendor
	}
	name := strings.TrimSpace(spec.Name)
	if name == "" {
		name = vendor
	}

	var (
		inner protocol.LLMProvider
		err   error
	)
	switch vendor {
	case "openai":
		inner, err = NewClient(ClientSpec{
			Name: "openai", APIKey: spec.APIKey, BaseURL: spec.BaseURL, Compat: spec.Compat,
		})
	case VendorOpenAIResponses:
		inner, err = NewClient(ClientSpec{
			Name: VendorOpenAIResponses, APIKey: spec.APIKey, BaseURL: spec.BaseURL,
		})
	case "anthropic":
		inner, err = NewClient(ClientSpec{
			Name: "anthropic", APIKey: spec.APIKey, BaseURL: spec.BaseURL,
		})
	case "google":
		inner, err = NewClient(ClientSpec{
			Name: "google", APIKey: spec.APIKey, BaseURL: spec.BaseURL,
		})
	case VendorClaudeCode, VendorOpenAICodex, VendorGoogleAntigravity, VendorDevin, VendorXaiOAuth:
		// OAuth subscription vendors: no API key — the wire client draws a live
		// token from the account pool per request.
		inner, err = NewClient(ClientSpec{
			Name: vendor, BaseURL: spec.BaseURL, TokenSource: spec.TokenSource,
			SessionScope: id,
		})
	default:
		if strings.TrimSpace(spec.BaseURL) == "" {
			return nil, fmt.Errorf("ai: provider %q requires a base URL", spec.Vendor)
		}
		compat := spec.Compat
		if compat.MaxTokensField == "" {
			compat = DefaultCompat()
		}
		inner, err = NewClient(ClientSpec{
			Name: spec.Vendor, APIKey: spec.APIKey, BaseURL: spec.BaseURL, Compat: compat,
		})
	}
	if err != nil {
		return nil, err
	}
	if spec.HTTP != nil {
		injectHTTP(inner, spec.HTTP)
	}
	var modelResponses protocol.LLMProvider
	if vendor == "openai" {
		// Keep an identity-preserving Responses peer ready for models whose live
		// metadata explicitly selects that API. Listing and chatting stay on the
		// same provider row/key/base URL; only the model's wire changes.
		modelResponses, err = NewClient(ClientSpec{
			Name: "openai", APIKey: spec.APIKey, BaseURL: spec.BaseURL,
			OpenAIWire: OpenAIWireResponses,
		})
		if err != nil {
			return nil, err
		}
		if spec.HTTP != nil {
			injectHTTP(modelResponses, spec.HTTP)
		}
	}

	w := &wired{
		id:             id,
		vendor:         vendor,
		name:           name,
		baseURL:        strings.TrimRight(strings.TrimSpace(spec.BaseURL), "/"),
		apiKey:         spec.APIKey,
		inner:          inner,
		modelResponses: modelResponses,
		http:           spec.HTTP,
	}
	if pooled, ok := inner.(*pooledProvider); ok {
		// OAuth vendors list models through the same account pool they chat
		// with: acquire a token, then call the vendor's list endpoint.
		w.tokenSource = spec.TokenSource
		w.listModels = oauthModelLister(vendor, pooled.inner, spec.HTTP, w.baseURL)
	}
	if _, ok := inner.(protocol.KeyUpdater); ok {
		return &wiredKeyProvider{wired: w}, nil
	}
	if _, ok := modelResponses.(protocol.KeyUpdater); ok {
		return &wiredKeyProvider{wired: w}, nil
	}
	return w, nil
}

// oauthModelLister resolves the vendor's list-models call for a pooled
// provider: it takes an acquired account token and returns the vendor's live
// catalog. A nil result means the vendor has no lister (shouldn't happen —
// every OAuth vendor ships one).
func oauthModelLister(vendor string, inner protocol.LLMProvider, http HTTPDoer, baseURL string) func(context.Context, OAuthToken) ([]Model, error) {
	switch vendor {
	case VendorClaudeCode:
		return func(ctx context.Context, tok OAuthToken) ([]Model, error) {
			return listClaudeCodeModels(ctx, http, baseURL, tok)
		}
	case VendorOpenAICodex:
		p, _ := inner.(*CodexProvider)
		return func(ctx context.Context, tok OAuthToken) ([]Model, error) {
			return p.listCodexModels(ctx, http, tok)
		}
	case VendorGoogleAntigravity:
		p, _ := inner.(*AntigravityProvider)
		return func(ctx context.Context, tok OAuthToken) ([]Model, error) {
			return p.listAntigravityModels(ctx, http, tok)
		}
	case VendorXaiOAuth:
		return func(ctx context.Context, tok OAuthToken) ([]Model, error) {
			base := baseURL
			if base == "" {
				base = ProviderDefaultURL(VendorXaiOAuth)
			}
			raw, err := listModelsForVendor(ctx, http, "xai", base, tok.AccessToken)
			out := make([]Model, 0, len(raw))
			for _, m := range raw {
				out = append(out, Model{ID: m.ID, ContextWindow: m.ContextWindow, Capabilities: m.Capabilities})
			}
			return out, err
		}
	case VendorDevin:
		p, _ := inner.(*DevinProvider)
		return func(ctx context.Context, tok OAuthToken) ([]Model, error) {
			return p.listDevinModels(ctx, http, tok)
		}
	}
	return nil
}

// injectHTTP hands the caller's HTTP client to the wire provider, so a test
// server, a proxy, or a custom timeout applies to the run as well as to
// list-models. Only *http.Client can be installed: the providers hold a
// concrete client, not an interface.
func injectHTTP(inner protocol.LLMProvider, client HTTPDoer) {
	std, ok := client.(*http.Client)
	if !ok {
		return
	}
	switch p := inner.(type) {
	case *xaiOAuthProvider:
		p.HTTP = std
		p.StreamHTTP = std
	case *OpenAIProvider:
		p.HTTP = std
	case *OpenAIResponsesProvider:
		p.HTTP = std
		p.StreamHTTP = std
	case *AnthropicProvider:
		p.HTTP = std
	case *CodexProvider:
		p.HTTP = std
		p.StreamHTTP = std
	case *AntigravityProvider:
		p.HTTP = std
		p.StreamHTTP = std
	case *DevinProvider:
		p.HTTP = std
		p.StreamHTTP = std
	case *pooledProvider:
		injectHTTP(p.inner, client)
	}
}

type wired struct {
	id, vendor, name, baseURL, apiKey string
	keyMu                             sync.RWMutex
	inner                             protocol.LLMProvider
	// modelResponses is the optional Responses peer for an ordinary OpenAI
	// provider row. It keeps the same identity and is selected only by explicit
	// per-model metadata learned from ListModels.
	modelResponses protocol.LLMProvider
	http           HTTPDoer
	// tokenSource and listModels are set only for OAuth vendors: the pool the
	// list-models call draws its credential from, and the vendor's lister.
	tokenSource TokenSource
	listModels  func(ctx context.Context, tok OAuthToken) ([]Model, error)
	modelsMu    sync.RWMutex
	// discovered keeps explicit live model facts by id. It is an optimization
	// and refinement only: a fresh process still has adapter defaults, and a
	// failed list call never erases the last successful knowledge.
	discovered map[string]protocol.ModelCapabilities
}

func (w *wired) ID() string          { return w.id }
func (w *wired) Vendor() string      { return w.vendor }
func (w *wired) DisplayName() string { return w.name }
func (w *wired) BaseURL() string     { return w.baseURL }
func (w *wired) APIKey() string {
	w.keyMu.RLock()
	defer w.keyMu.RUnlock()
	return w.apiKey
}
func (w *wired) Name() string        { return w.inner.Name() }
func (w *wired) SupportsTools() bool { return w.inner.SupportsTools() }

func (w *wired) ModelCapabilities(model string) protocol.ModelCapabilities {
	caps := protocol.CapabilitiesOf(w.inner, model)
	w.modelsMu.RLock()
	live := w.discovered[model]
	w.modelsMu.RUnlock()
	return caps.Overlay(live)
}

func (w *wired) Chat(ctx context.Context, req protocol.ChatRequest) (protocol.ChatResponse, error) {
	return w.providerForModel(req.Model).Chat(ctx, req)
}

func (w *wired) Stream(ctx context.Context, req protocol.ChatRequest) (<-chan protocol.ChatDelta, error) {
	return w.providerForModel(req.Model).Stream(ctx, req)
}

func (w *wired) providerForModel(model string) protocol.LLMProvider {
	if w.modelResponses == nil {
		return w.inner
	}
	w.modelsMu.RLock()
	caps := w.discovered[model]
	w.modelsMu.RUnlock()
	if caps.StatefulResponses == protocol.CapabilitySupported {
		return w.modelResponses
	}
	return w.inner
}

// wiredKeyProvider exists only for provider rows whose wire client accepts a
// mutable static key. Keeping UpdateAPIKey off wired itself prevents an OAuth
// account pool from regaining KeyUpdater through this identity decorator.
type wiredKeyProvider struct{ *wired }

func (w *wiredKeyProvider) UpdateAPIKey(key string) {
	if key == "" {
		return
	}
	w.keyMu.Lock()
	w.apiKey = key
	w.keyMu.Unlock()
	if u, ok := w.inner.(protocol.KeyUpdater); ok {
		u.UpdateAPIKey(key)
	}
	if u, ok := w.modelResponses.(protocol.KeyUpdater); ok {
		u.UpdateAPIKey(key)
	}
}

func (w *wired) ListModels(ctx context.Context) ([]Model, error) {
	var listed []Model
	if w.listModels != nil {
		// OAuth vendors authenticate the list call with a pooled account token,
		// same as a chat request; the outcome reports back so a dead account
		// rotates out.
		tok, err := w.tokenSource.Acquire(ctx)
		if err != nil {
			return nil, protocol.NewProviderError(w.vendor, nil, err.Error())
		}
		listed, err = w.listModels(ctx, tok)
		w.tokenSource.Report(ctx, tok, err)
		if err != nil {
			return nil, err
		}
	} else {
		raw, err := listModelsForVendor(ctx, w.http, w.vendor, w.effectiveBaseURL(), w.APIKey())
		if err != nil {
			return nil, err
		}
		listed = make([]Model, 0, len(raw))
		for _, m := range raw {
			listed = append(listed, Model{ID: m.ID, ContextWindow: m.ContextWindow, Capabilities: m.Capabilities})
		}
	}
	out := make([]Model, 0, len(listed))
	discovered := make(map[string]protocol.ModelCapabilities, len(listed))
	for _, m := range listed {
		// The vendor's own figure wins; the table only fills a gap. A vendor that
		// starts reporting the window therefore takes over automatically, and a
		// stale table entry can never override a live one.
		window := m.ContextWindow
		if window <= 0 {
			window = ContextWindowFor(w.vendor, m.ID)
		}
		caps := protocol.CapabilitiesOf(w.inner, m.ID).Overlay(m.Capabilities)
		discovered[m.ID] = m.Capabilities
		out = append(out, Model{
			ProviderID:     w.id,
			ProviderVendor: w.vendor,
			ProviderName:   w.name,
			ID:             m.ID,
			ContextWindow:  window,
			Capabilities:   caps,
		})
	}
	w.modelsMu.Lock()
	w.discovered = discovered
	w.modelsMu.Unlock()
	return out, nil
}

func (w *wired) effectiveBaseURL() string {
	if w.baseURL != "" {
		return w.baseURL
	}
	switch w.vendor {
	case "anthropic":
		return defaultAnthropicBaseURL
	case "google":
		return defaultGoogleBaseURL
	default:
		return defaultOpenAIBaseURL
	}
}
