package ai

import (
	"context"
	"fmt"
	"sync"

	"github.com/2found/2ai/ai/protocol"
)

// Collection holds registered providers and routes Chat/Stream to the owner
// of the requested model. ListModels unions the live catalogs of every
// registered provider (failures are returned per-provider, not swallowed).
type Collection struct {
	mu        sync.RWMutex
	order     []string
	providers map[string]Provider

	// last is the most recent successful union, keyed by model id → provider id.
	// Refreshed by ListModels. Chat/Stream consult it to pick an owner.
	last []Model
}

// NewCollection returns an empty collection.
func NewCollection() *Collection {
	return &Collection{providers: map[string]Provider{}}
}

// CollectionFromSpecs registers one provider per Spec. Used by the workspace
// API and persist tests so listing and Chat share the same construction path.
func CollectionFromSpecs(specs []Spec) (*Collection, error) {
	col := NewCollection()
	for _, spec := range specs {
		p, err := New(spec)
		if err != nil {
			return nil, err
		}
		col.Register(p)
	}
	return col, nil
}

// Register adds or replaces a provider. The provider's ID is the map key.
func (c *Collection) Register(p Provider) {
	if p == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	id := p.ID()
	if _, exists := c.providers[id]; !exists {
		c.order = append(c.order, id)
	}
	c.providers[id] = p
}

// Get returns a registered provider by id.
func (c *Collection) Get(id string) (Provider, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	p, ok := c.providers[id]
	return p, ok
}

// Providers returns registered providers in registration order.
func (c *Collection) Providers() []Provider {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]Provider, 0, len(c.order))
	for _, id := range c.order {
		if p, ok := c.providers[id]; ok {
			out = append(out, p)
		}
	}
	return out
}

// ListError is a per-provider list-models failure. The rest of the collection
// still contributes its models.
type ListError struct {
	ProviderID string `json:"provider_id"`
	Error      string `json:"error"`
}

// ListResult is the union of listed models plus any per-provider errors.
type ListResult struct {
	Models []Model     `json:"models"`
	Errors []ListError `json:"errors,omitempty"`
}

// ListModels refreshes every registered provider's catalog via its vendor
// list-models HTTP API and returns the union. The last-known map is updated
// so a subsequent Chat/Stream can find the owner.
func (c *Collection) ListModels(ctx context.Context) (ListResult, error) {
	providers := c.Providers()
	result := ListResult{Models: []Model{}}
	for _, p := range providers {
		models, err := p.ListModels(ctx)
		if err != nil {
			result.Errors = append(result.Errors, ListError{ProviderID: p.ID(), Error: err.Error()})
			continue
		}
		result.Models = append(result.Models, models...)
	}
	c.mu.Lock()
	c.last = append([]Model(nil), result.Models...)
	c.mu.Unlock()
	return result, nil
}

// Owner returns the provider that listed modelID. If the cache is empty it
// refreshes first. First registered owner wins on an ID collision.
func (c *Collection) Owner(ctx context.Context, modelID string) (Provider, error) {
	c.mu.RLock()
	cached := append([]Model(nil), c.last...)
	c.mu.RUnlock()
	if len(cached) == 0 {
		listed, err := c.ListModels(ctx)
		if err != nil {
			return nil, err
		}
		cached = listed.Models
	}
	for _, m := range cached {
		if m.ID == modelID {
			if p, ok := c.Get(m.ProviderID); ok {
				return p, nil
			}
		}
	}
	return nil, fmt.Errorf("ai: no registered provider owns model %q", modelID)
}

// Chat implements protocol.LLMProvider: the owning provider serves the request.
func (c *Collection) Chat(ctx context.Context, req protocol.ChatRequest) (protocol.ChatResponse, error) {
	p, err := c.Owner(ctx, req.Model)
	if err != nil {
		return protocol.ChatResponse{}, err
	}
	return p.Chat(ctx, req)
}

// Stream implements protocol.LLMProvider.
func (c *Collection) Stream(ctx context.Context, req protocol.ChatRequest) (<-chan protocol.ChatDelta, error) {
	p, err := c.Owner(ctx, req.Model)
	if err != nil {
		return nil, err
	}
	return p.Stream(ctx, req)
}

// ChatOn sends the request to a specific registered provider (used when a
// tier names both a provider id and a model, so an ID collision cannot
// mis-route).
func (c *Collection) ChatOn(ctx context.Context, providerID string, req protocol.ChatRequest) (protocol.ChatResponse, error) {
	p, ok := c.Get(providerID)
	if !ok {
		return protocol.ChatResponse{}, fmt.Errorf("ai: unknown provider %q", providerID)
	}
	return p.Chat(ctx, req)
}

// StreamOn is Stream scoped to one registered provider.
func (c *Collection) StreamOn(ctx context.Context, providerID string, req protocol.ChatRequest) (<-chan protocol.ChatDelta, error) {
	p, ok := c.Get(providerID)
	if !ok {
		return nil, fmt.Errorf("ai: unknown provider %q", providerID)
	}
	return p.Stream(ctx, req)
}

func (c *Collection) Name() string        { return "collection" }
func (c *Collection) SupportsTools() bool { return true }

func (c *Collection) ModelCapabilities(model string) protocol.ModelCapabilities {
	c.mu.RLock()
	var listed Model
	var provider Provider
	for _, candidate := range c.last {
		if candidate.ID == model {
			listed = candidate
			provider = c.providers[candidate.ProviderID]
			break
		}
	}
	c.mu.RUnlock()
	if provider == nil {
		return protocol.ModelCapabilities{}
	}
	return protocol.CapabilitiesOf(provider, model).Overlay(listed.Capabilities)
}

var _ protocol.LLMProvider = (*Collection)(nil)
