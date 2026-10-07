package ai

import (
	"context"
	"errors"
	"sync"
	"testing"
)

func TestPiModelsRefreshSuperseded(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cache := NewInMemoryModelsStore()
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	late := make(chan any, 1)
	var mu sync.Mutex
	count := 0
	updates := NewArray()
	signals := []context.Context{}
	models := NewModels(ModelsOptions{Persistence: &ModelsPersistence{Read: cache.Read, Write: cache.Write, Delete: cache.Delete}})
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(APIKeyAuthInput) (any, error) {
		return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: "key"})}), nil
	}}}, RefreshModels: func(refresh ModelRefreshContext) error {
		if !refresh.AllowNetwork {
			return nil
		}
		mu.Lock()
		count++
		n := count
		signals = append(signals, refresh.Context)
		mu.Unlock()
		if n == 1 {
			close(started)
			<-release
		}
		accepted, err := refresh.Publish(ModelsPublication{Persist: NewObject(Property{Name: "models", Value: NewArray()}, Property{Name: "checkedAt", Value: n}), Update: func() error {
			mu.Lock()
			defer mu.Unlock()
			updates.Append(n)
			return nil
		}})
		if n == 1 {
			if err != nil {
				late <- map[string]any{"cancelled": refresh.Context.Err() != nil}
			} else {
				late <- map[string]any{"accepted": accepted}
			}
			return nil
		}
		return err
	}})
	first := make(chan ModelsRefreshResult, 1)
	go func() { first <- models.Refresh(ctx) }()
	publicationAwait(t, started)
	trace := map[string]any{"second": summarizeModelsRefresh(models.Refresh(ctx)), "first": summarizeModelsRefresh(publicationAwait(t, first))}
	once.Do(func() { close(release) })
	trace["late"] = publicationAwait(t, late)
	mu.Lock()
	trace["updates"] = updates
	aborted := NewArray()
	for _, signal := range signals {
		aborted.Append(signal.Err() != nil)
	}
	trace["signals"] = aborted
	mu.Unlock()
	var err error
	trace["cached"], err = cache.Read(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	catalogCompare(t, trace, readModelsRefreshFixture(t).Superseded)
}

func TestPiModelsRefreshCanceledReadStillRestores(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	phases := make(chan any, 1)
	models := NewModels(ModelsOptions{
		Credentials: &CredentialPersistence{Read: func(context.Context, string) (any, error) {
			close(started)
			<-release
			return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "stored"}), nil
		}},
		Persistence: &ModelsPersistence{Read: func(context.Context, string) (any, error) { return Undefined, nil }},
	})
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(APIKeyAuthInput) (any, error) { return nil, errors.New("must not resolve") }}}, RefreshModels: func(refresh ModelRefreshContext) error {
		phases <- NewObject(Property{Name: "network", Value: refresh.AllowNetwork}, Property{Name: "aborted", Value: refresh.Context.Err() != nil}, Property{Name: "credential", Value: refresh.Credential})
		return nil
	}})
	pending := make(chan ModelsRefreshResult, 1)
	go func() { pending <- models.Refresh(ctx) }()
	publicationAwait(t, started)
	cancel(errors.New("cancelled"))
	trace := map[string]any{"result": summarizeModelsRefresh(publicationAwait(t, pending))}
	once.Do(func() { close(release) })
	trace["phases"] = NewArray(publicationAwait(t, phases))
	catalogCompare(t, trace, readModelsRefreshFixture(t).Cancelled)
}

func TestModelsRefreshCatalogIntegrationAndIndependentProviders(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cache := NewInMemoryModelsStore()
	model := func(id string) *Object {
		return NewObject(Property{Name: "id", Value: id}, Property{Name: "provider", Value: "p"})
	}
	if err := cache.Write(ctx, "p", NewObject(Property{Name: "models", Value: NewArray(model("cached"), NewObject(Property{Name: "type", Value: "future"}, Property{Name: "id", Value: "future"}, Property{Name: "provider", Value: "p"}))})); err != nil {
		t.Fatal(err)
	}
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	catalog := NewProviderModelCatalog("p", NewArray(model("baseline")), func(refresh ModelRefreshContext) (*Array, error) {
		close(entered)
		<-release
		return NewArray(model("fetched")), nil
	}, func() int64 { return 1000 })
	models := NewModels(ModelsOptions{Persistence: &ModelsPersistence{Read: cache.Read, Write: cache.Write, Delete: cache.Delete}})
	models.SetProvider(&ModelProvider{ID: "p", GetModels: catalog.GetModels, GetAllModels: catalog.GetAllModels, RefreshModels: catalog.RefreshModels, Auth: &ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(APIKeyAuthInput) (any, error) {
		return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: "key"})}), nil
	}}}})
	cause := errors.New("independent provider failed")
	otherEntered := make(chan struct{})
	models.SetProvider(&ModelProvider{ID: "q", RefreshModels: func(ModelRefreshContext) error { close(otherEntered); return cause }})
	pending := make(chan ModelsRefreshResult, 1)
	go func() { pending <- models.Refresh(ctx) }()
	publicationAwait(t, entered)
	publicationAwait(t, otherEntered)
	visible := models.GetAllModels("p")
	if visible.Len() != 2 || catalogProperty(visible.Get(1), "id") != "cached" {
		t.Fatal("cache not restored before network or unknown type retained")
	}
	once.Do(func() { close(release) })
	result := publicationAwait(t, pending)
	if result.Aborted || len(result.Errors) != 1 || result.Errors[0].ProviderID != "q" || result.Errors[0].Error != cause {
		t.Fatalf("refresh lost provider error identity: %+v", result)
	}
	visible = models.GetAllModels("p")
	if visible.Len() != 2 || catalogProperty(visible.Get(1), "id") != "fetched" {
		t.Fatal("refreshed catalog not visible")
	}
	stored, err := cache.Read(ctx, "p")
	if err != nil || catalogProperty(stored, "checkedAt") != int64(1000) {
		t.Fatalf("refresh snapshot: %v, %v", stored, err)
	}
}
