package ai

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"
)

func availableCheck() any {
	return NewObject(Property{Name: "source", Value: "check"}, Property{Name: "type", Value: "api_key"})
}

func TestPiModelsAvailabilityConcurrentSnapshot(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pStarted, qStarted, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	models := NewModels()
	reads := 0
	p := NewObject(Property{Name: "id", Value: "p-model"}, Property{Name: "provider", Value: "p"})
	q := NewObject(Property{Name: "id", Value: "q-model"}, Property{Name: "provider", Value: "q"})
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Check: func(APIKeyAuthInput) (any, error) { close(pStarted); <-release; return availableCheck(), nil }}}, GetModels: func() (*Array, error) { reads++; return NewArray(p), nil }})
	models.SetProvider(&ModelProvider{ID: "q", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Check: func(APIKeyAuthInput) (any, error) { close(qStarted); return availableCheck(), nil }}}, GetModels: func() (*Array, error) { reads++; return NewArray(q), nil }})
	type outcome struct {
		models *Array
		err    error
	}
	pending := make(chan outcome, 1)
	go func() { value, err := models.GetAllAvailable(ctx, ""); pending <- outcome{value, err} }()
	publicationAwait(t, pStarted)
	publicationAwait(t, qStarted)
	trace := map[string]any{"readsBefore": reads}
	models.DeleteProvider("p")
	models.SetProvider(&ModelProvider{ID: "q", GetModels: func() (*Array, error) {
		return NewArray(NewObject(Property{Name: "id", Value: "replacement"}, Property{Name: "provider", Value: "q"})), nil
	}})
	once.Do(func() { close(release) })
	result := publicationAwait(t, pending)
	if result.err != nil {
		t.Fatal(result.err)
	}
	trace["result"] = result.models
	trace["sameModels"] = result.models.Get(0) == p && result.models.Get(1) == q
	registry := NewArray()
	for _, provider := range models.GetProviders() {
		registry.Append(provider.ID)
	}
	trace["registry"] = registry
	trace["readsAfter"] = reads
	catalogCompare(t, trace, readModelsAvailabilityFixture(t).Concurrent)
}

func TestPiModelsAvailabilityCancellationObservesRemainingWork(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	started, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	models := NewModels()
	reads := 0
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Check: func(APIKeyAuthInput) (any, error) { close(started); <-release; return availableCheck(), nil }}}, GetModels: func() (*Array, error) { reads++; close(finished); return NewArray(), nil }})
	pending := make(chan any, 1)
	go func() { value, err := models.GetAvailable(ctx, "p"); pending <- authResolveCapture(value, err) }()
	publicationAwait(t, started)
	cancel(errors.New("cancelled"))
	trace := map[string]any{"result": publicationAwait(t, pending), "readsBefore": reads}
	once.Do(func() { close(release) })
	publicationAwait(t, finished)
	trace["readsAfter"] = reads
	catalogCompare(t, trace, readModelsAvailabilityFixture(t).Cancelled)
}

func TestPiModelsAvailabilityIDSet(t *testing.T) {
	shared := NewObject(Property{Name: "key", Value: "shared"})
	different := NewObject(Property{Name: "key", Value: "shared"})
	model := func(label string, id any) *Object {
		return NewObject(Property{Name: "label", Value: label}, Property{Name: "id", Value: id}, Property{Name: "provider", Value: "p"})
	}
	all := NewArray(model("nan", math.NaN()), NewObject(Property{Name: "label", Value: "missing"}, Property{Name: "provider", Value: "p"}), model("undefined", Undefined), model("shared", shared), model("different", different), model("null", Null), model("zero", 0), model("one", 1), model("string-one", "1"), NewObject(Property{Name: "label", Value: "image"}, Property{Name: "id", Value: "image"}, Property{Name: "type", Value: "image"}, Property{Name: "provider", Value: "p"}))
	selected := NewArray(model("selected-nan", float32(math.NaN())), Undefined, model("selected-object", shared), model("selected-null", Null), model("selected-zero", math.Copysign(0, -1)), model("selected-one", 1))
	selected.Delete(1)
	models := NewModels()
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Check: func(APIKeyAuthInput) (any, error) { return availableCheck(), nil }}}, GetModels: func() (*Array, error) { return all, nil }, GetAllModels: func() (*Array, error) { return all, nil }, FilterModels: func(*Array, any) (*Array, error) { return selected, nil }})
	result, err := models.GetAllAvailable(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	labels := NewArray()
	for _, index := range result.Keys() {
		labels.Append(catalogProperty(result.Get(index), "label"))
	}
	catalogCompare(t, map[string]any{"labels": labels, "sharedModel": result.Get(3) == all.Get(3), "newUnion": result != all}, readModelsAvailabilityFixture(t).IDs)
}

func TestModelsAvailabilityFailureDoesNotWaitForOtherProvider(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	cause := errors.New("check failure")
	models := NewModels()
	models.SetProvider(&ModelProvider{ID: "slow", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Check: func(APIKeyAuthInput) (any, error) {
		close(started)
		<-release
		close(finished)
		return availableCheck(), nil
	}}}, GetModels: func() (*Array, error) { t.Error("catalog read after failed auth batch"); return NewArray(), nil }})
	models.SetProvider(&ModelProvider{ID: "fail", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Check: func(APIKeyAuthInput) (any, error) { <-started; return nil, cause }}}})
	pending := make(chan error, 1)
	go func() { _, err := models.GetAllAvailable(ctx); pending <- err }()
	err := publicationAwait(t, pending)
	var typed *ModelsError
	if !errors.Is(err, cause) || !errors.As(err, &typed) || typed.Code != "auth" {
		t.Fatalf("lost auth failure cause: %v", err)
	}
	once.Do(func() { close(release) })
	publicationAwait(t, finished)
}
