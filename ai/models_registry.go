package ai

import (
	"context"
	"sync"

	"github.com/2found/2ai/jsonjs"
)

// ModelProvider carries Pi's concrete provider identity and catalog callbacks.
// GetAllModels and RefreshModels are optional. Callbacks retain their returned
// model/list identities and run without the registry lock. Provider mutation
// outside SetProvider is caller-synchronized, as are live model objects.
type ModelProvider struct {
	ID              string
	Name            string
	BaseURL         any
	Headers         any
	Auth            *ProviderAuth
	GetModels       func() (*Array, error)
	GetAllModels    func() (*Array, error)
	RefreshModels   func(ModelRefreshContext) error
	FilterModels    func(*Array, any) (*Array, error)
	FilterAllModels func(*Array, any) (*Array, error)
	Stream          ModelStreamFunc
	StreamSimple    ModelStreamFunc
	FetchDeferred   DeferredStreamFunc
	CancelDeferred  func(context.Context, any, any, *Object) error
	GenerateImages  func(context.Context, any, any, *Object) (any, error)
	Classify        func(context.Context, any, any, *Object) (any, error)
}

func (p *ModelProvider) authConfig() *ProviderAuth {
	if p.Auth != nil {
		return p.Auth
	}
	return &ProviderAuth{}
}

type ModelsOptions struct {
	Persistence *ModelsPersistence
	Credentials *CredentialPersistence
	AuthContext *AuthContext
}

// Models is Pi's native provider registry. It is distinct from the workspace
// Collection, whose live discovery and owner-by-model-ID rules differ from Pi.
// Native auth lifecycle, availability, catalog refresh and request dispatch
// share this registry. Concrete provider factories are wired separately.
type Models struct {
	mu          sync.Mutex
	providers   map[string]*modelProviderSlot
	order       []*modelProviderSlot
	iterators   int
	persistence ModelsPersistence
	publisher   *ModelsPublisher
	credentials CredentialPersistence
	authContext *AuthContext
}

type modelProviderSlot struct {
	provider *ModelProvider
}

func NewModels(options ...ModelsOptions) *Models {
	store := NewInMemoryModelsStore()
	persistence := ModelsPersistence{Read: store.Read, Write: store.Write, Delete: store.Delete}
	if len(options) > 0 && options[0].Persistence != nil {
		// Pi retains the supplied store object. Resolve its methods at each
		// call, so replacing a callback between operations remains observable.
		provided := options[0].Persistence
		persistence = ModelsPersistence{
			Read:   func(ctx context.Context, id string) (any, error) { return provided.Read(ctx, id) },
			Write:  func(ctx context.Context, id string, entry any) error { return provided.Write(ctx, id, entry) },
			Delete: func(ctx context.Context, id string) error { return provided.Delete(ctx, id) },
		}
	}
	credentialStore := NewInMemoryCredentialStore()
	credentials := CredentialPersistence{Read: credentialStore.Read, List: credentialStore.List, Modify: credentialStore.Modify, Delete: credentialStore.Delete}
	if len(options) > 0 && options[0].Credentials != nil {
		provided := options[0].Credentials
		credentials = CredentialPersistence{
			Read: func(ctx context.Context, id string) (any, error) { return provided.Read(ctx, id) },
			List: func(ctx context.Context) (*Array, error) { return provided.List(ctx) },
			Modify: func(ctx context.Context, id string, modify func(any) (any, error)) (any, error) {
				return provided.Modify(ctx, id, modify)
			},
			Delete: func(ctx context.Context, id string) error { return provided.Delete(ctx, id) },
		}
	}
	authContext := DefaultProviderAuthContext()
	if len(options) > 0 && options[0].AuthContext != nil {
		authContext = options[0].AuthContext
	}
	return &Models{providers: make(map[string]*modelProviderSlot), persistence: persistence, publisher: NewModelsPublisher(persistence), credentials: credentials, authContext: authContext}
}

func (m *Models) SetProvider(provider *ModelProvider) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.publisher.Invalidate(provider.ID)
	if slot := m.providers[provider.ID]; slot != nil {
		slot.provider = provider
		return
	}
	slot := &modelProviderSlot{provider: provider}
	m.providers[provider.ID] = slot
	m.order = append(m.order, slot)
}

func (m *Models) DeleteProvider(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.publisher.Invalidate(id)
	if slot := m.providers[id]; slot != nil {
		slot.provider = nil
		delete(m.providers, id)
	}
	m.compact()
}

func (m *Models) ClearProviders() {
	m.mu.Lock()
	defer m.mu.Unlock()
	ids := make(map[string]bool, len(m.providers))
	for id, slot := range m.providers {
		ids[id] = true
		slot.provider = nil
	}
	clear(m.providers)
	m.publisher.clearProviders(ids)
	m.compact()
}

// compact runs under mu. Active Map-style iterators retain deleted slots until
// they finish so deletion/reinsertion and Clear followed by Set remain visible.
func (m *Models) compact() {
	if m.iterators != 0 {
		return
	}
	order := m.order[:0]
	for _, slot := range m.order {
		if slot.provider != nil {
			order = append(order, slot)
		}
	}
	clear(m.order[len(order):])
	m.order = order
}

func (m *Models) GetProvider(id string) *ModelProvider {
	m.mu.Lock()
	defer m.mu.Unlock()
	if slot := m.providers[id]; slot != nil {
		return slot.provider
	}
	return nil
}

func (m *Models) GetProviders() []*ModelProvider {
	m.mu.Lock()
	defer m.mu.Unlock()
	providers := make([]*ModelProvider, 0, len(m.providers))
	for _, slot := range m.order {
		if slot.provider != nil {
			providers = append(providers, slot.provider)
		}
	}
	return providers
}

func readProviderModels(provider *ModelProvider, all bool) (models *Array) {
	// Pi swallows catalog errors, including a throwing getAllModels callback;
	// only a nullish result from that optional callback falls back to getModels.
	defer func() {
		if recover() != nil {
			models = NewArray()
		}
	}()
	var err error
	if all && provider.GetAllModels != nil {
		models, err = provider.GetAllModels()
		if err != nil {
			return NewArray()
		}
	}
	if models == nil {
		models, err = provider.GetModels()
	}
	if err != nil {
		return NewArray()
	}
	return models
}

func (m *Models) models(all bool, providerID []string) *Array {
	if len(providerID) > 0 {
		if provider := m.GetProvider(providerID[0]); provider != nil {
			return readProviderModels(provider, all)
		}
		return NewArray()
	}
	m.mu.Lock()
	m.iterators++
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		m.iterators--
		m.compact()
		m.mu.Unlock()
	}()
	result := NewArray()
	for index := 0; ; index++ {
		m.mu.Lock()
		if index >= len(m.order) {
			m.mu.Unlock()
			break
		}
		provider := m.order[index].provider
		m.mu.Unlock()
		if provider == nil {
			continue
		}
		models := readProviderModels(provider, all)
		for slot := 0; slot < models.Len(); slot++ {
			result.Append(models.Get(slot))
		}
	}
	return result
}

// GetModels/GetAllModels return the provider's original list when scoped and a
// fresh union when unscoped. Passing no argument differs from passing "".
func (m *Models) GetModels(providerID ...string) *Array    { return m.models(false, providerID) }
func (m *Models) GetAllModels(providerID ...string) *Array { return m.models(true, providerID) }

func (m *Models) GetModelsOfType(kind any, providerID ...string) (*Array, error) {
	all := m.GetAllModels(providerID...)
	if all == nil {
		return nil, jsonjs.PropertyReadError(true, "this.getAllModels(provider).filter")
	}
	result := NewArray()
	for _, index := range all.Keys() {
		model := all.Get(index)
		matches, err := IsModelType(model, kind)
		if err != nil {
			return nil, err
		}
		if matches {
			result.Append(model)
		}
	}
	return result, nil
}

func findCatalogModel(models *Array, id any) (any, error) {
	for index := 0; index < models.Len(); index++ {
		model := models.Get(index)
		if jsonjs.IsNullish(model) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(model), "model.id")
		}
		if catalogStrictEqual(catalogProperty(model, "id"), id) {
			return model, nil
		}
	}
	return Undefined, nil
}

func (m *Models) GetModel(providerID string, id any) (any, error) {
	models := m.GetModels(providerID)
	if models == nil {
		return nil, jsonjs.PropertyReadError(true, "this.getModels(provider).find")
	}
	return findCatalogModel(models, id)
}

func (m *Models) GetModelOfType(kind any, providerID string, id any) (any, error) {
	models, err := m.GetModelsOfType(kind, providerID)
	if err != nil {
		return nil, err
	}
	return findCatalogModel(models, id)
}
