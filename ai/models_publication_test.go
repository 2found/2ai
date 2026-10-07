package ai

import (
	"context"
	"errors"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

func TestPiModelsPublication(t *testing.T) {
	log := NewArray()
	p := NewModelsPublisher(ModelsPersistence{
		Write: func(_ context.Context, id string, entry any) error {
			log.Append(NewArray("write", id, entry))
			if fail, ok := entry.(*Object).Get("fail").(bool); ok && fail {
				return errors.New("write failed")
			}
			return nil
		},
		Delete: func(_ context.Context, id string) error { log.Append(NewArray("delete", id)); return nil },
	})
	defer p.Clear()
	scope := p.Begin(context.Background(), "p")
	fixture := readProviderCatalogFixture(t)
	values := []any{Undefined, NewObject(Property{Name: "models", Value: NewArray()}, Property{Name: "v", Value: 1}), Null, NewObject(Property{Name: "fail", Value: true}), NewObject(Property{Name: "models", Value: NewArray()}, Property{Name: "v", Value: 2})}
	for index, persist := range values {
		accepted, err := scope.Publish(ModelsPublication{Persist: persist, Update: func() error { log.Append(NewArray("update")); return nil }})
		actual := map[string]any{"log": log}
		if err != nil {
			actual["error"] = err.Error()
		} else {
			actual["accepted"] = accepted
		}
		catalogCompare(t, actual, fixture.PublicationTrace[index])
	}
	p.Invalidate("p")
	_, err := scope.Publish(ModelsPublication{Persist: NewObject(Property{Name: "models", Value: NewArray()}), Update: func() error { log.Append(NewArray("stale")); return nil }})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("stale publication: %v", err)
	}
	catalogCompare(t, map[string]any{"cancelled": scope.Context.Err() != nil, "log": log}, fixture.PublicationTrace[5])
}

type publicationOutcome struct {
	accepted bool
	err      error
}

func publicationAwait[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-time.After(5 * time.Second):
		t.Fatal("publication did not settle")
		var zero T
		return zero
	}
}
func TestModelsPublicationCanceledWriteCannotOvertakeNewGeneration(t *testing.T) {
	store := NewInMemoryModelsStore()
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	var mu sync.Mutex
	writes := []string{}
	p := NewModelsPublisher(ModelsPersistence{Write: func(ctx context.Context, id string, entry any) error {
		version := entry.(*Object).Get("version").(string)
		if version == "old" {
			close(entered)
			<-release
		}
		mu.Lock()
		writes = append(writes, version)
		mu.Unlock()
		// Simulate a backend already past its cancellation boundary.
		return store.Write(context.Background(), id, entry)
	}, Delete: store.Delete})
	defer p.Clear()
	old := p.Begin(context.Background(), "p")
	oldResult := make(chan publicationOutcome, 1)
	updates := make(chan string, 2)
	go func() {
		ok, err := old.Publish(ModelsPublication{Persist: NewObject(Property{Name: "version", Value: "old"}), Update: func() error { updates <- "old"; return nil }})
		oldResult <- publicationOutcome{ok, err}
	}()
	publicationAwait(t, entered)
	p.queue.mu.Lock()
	oldTail := p.queue.tails["p"]
	p.queue.mu.Unlock()
	current := p.Begin(context.Background(), "p")
	if result := publicationAwait(t, oldResult); result.accepted || !errors.Is(result.err, context.Canceled) {
		t.Fatalf("old caller did not abort: %+v", result)
	}
	currentResult := make(chan publicationOutcome, 1)
	go func() {
		ok, err := current.Publish(ModelsPublication{Persist: NewObject(Property{Name: "version", Value: "new"}), Update: func() error { updates <- "new"; return nil }})
		currentResult <- publicationOutcome{ok, err}
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		p.queue.mu.Lock()
		queued := p.queue.tails["p"] != oldTail
		p.queue.mu.Unlock()
		if queued {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("new publication was not queued")
		}
		runtime.Gosched()
	}
	// A distinct provider is not held by p's stalled persistence callback.
	other := p.Begin(context.Background(), "q")
	otherResult := make(chan publicationOutcome, 1)
	go func() {
		ok, err := other.Publish(ModelsPublication{Persist: NewObject(Property{Name: "version", Value: "other"})})
		otherResult <- publicationOutcome{ok, err}
	}()
	if result := publicationAwait(t, otherResult); !result.accepted || result.err != nil {
		t.Fatalf("independent provider blocked: %+v", result)
	}
	releaseOnce.Do(func() { close(release) })
	if result := publicationAwait(t, currentResult); !result.accepted || result.err != nil {
		t.Fatalf("new publication failed: %+v", result)
	}
	if update := publicationAwait(t, updates); update != "new" {
		t.Fatalf("stale update ran: %s", update)
	}
	select {
	case update := <-updates:
		t.Fatalf("extra update %s", update)
	default:
	}
	value, err := store.Read(context.Background(), "p")
	if err != nil || value.(*Object).Get("version") != "new" {
		t.Fatalf("stale snapshot won: %v %v", value, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(writes, []string{"other", "old", "new"}) {
		t.Fatalf("publication order %v", writes)
	}
	expected := diagnosticValue(t, readProviderCatalogFixture(t).PublicationRace).(map[string]any)
	delete(expected, "selfCancelled") // Checked by the reentrant-update case below.
	actual := map[string]any{"old": map[string]any{"cancelled": old.Context.Err() != nil}, "other": true, "current": true, "writes": writes, "updates": []string{"new"}, "final": value}
	if !reflect.DeepEqual(diagnosticValue(t, []byte(catalogStringify(t, actual))), expected) {
		t.Fatalf("publication race differs from source: %s", catalogStringify(t, actual))
	}
}
func TestModelsPublicationCloneAndUpdateFailureDoNotPoisonQueue(t *testing.T) {
	store := NewInMemoryModelsStore()
	p := NewModelsPublisher(ModelsPersistence{Write: store.Write, Delete: store.Delete})
	defer p.Clear()
	scope := p.Begin(context.Background(), "p")
	called := false
	_, err := scope.Publish(ModelsPublication{Persist: NewObject(Property{Name: "toJSON", Value: JSONMethod(func(any, string) (any, error) { called = true; return nil, nil })})})
	var cloneError *DataCloneError
	if !errors.As(err, &cloneError) || called {
		t.Fatalf("clone boundary: %v, called=%v", err, called)
	}
	sentinel := errors.New("update failed")
	_, err = scope.Publish(ModelsPublication{Persist: NewObject(Property{Name: "version", Value: 1}), Update: func() error { return sentinel }})
	if !errors.Is(err, sentinel) {
		t.Fatalf("update error identity: %v", err)
	}
	value, err := store.Read(context.Background(), "p")
	if err != nil || value.(*Object).Get("version") != 1 {
		t.Fatal("persistence should precede failed update")
	}
	_, err = scope.Publish(ModelsPublication{Update: func() error { panic(sentinel) }})
	if !errors.Is(err, sentinel) {
		t.Fatalf("panic identity: %v", err)
	}
	accepted, err := scope.Publish(ModelsPublication{Persist: Null})
	if !accepted || err != nil {
		t.Fatalf("failed chain retained: %v", err)
	}
	value, err = store.Read(context.Background(), "p")
	if err != nil || !jsonjs.IsUndefined(value) {
		t.Fatal("delete publication failed")
	}
}
func TestModelsPublicationUpdateCanInvalidateItself(t *testing.T) {
	p := NewModelsPublisher(ModelsPersistence{})
	defer p.Clear()
	scope := p.Begin(context.Background(), "p")
	done := make(chan publicationOutcome, 1)
	go func() {
		ok, err := scope.Publish(ModelsPublication{Update: func() error { p.Invalidate("p"); return nil }})
		done <- publicationOutcome{ok, err}
	}()
	if result := publicationAwait(t, done); result.accepted || !errors.Is(result.err, context.Canceled) {
		t.Fatalf("reentrant update: %+v", result)
	}
	if expected := diagnosticValue(t, readProviderCatalogFixture(t).PublicationRace).(map[string]any)["selfCancelled"]; expected != (scope.Context.Err() != nil) {
		t.Fatal("reentrant cancellation differs from source")
	}
}

func TestProviderCatalogPersistencePrecedesVisibleRefresh(t *testing.T) {
	for _, mode := range []string{"success", "write-error", "superseded"} {
		t.Run(mode, func(t *testing.T) {
			ctx := context.Background()
			store := NewInMemoryModelsStore()
			model := func(id string) *Object {
				return NewObject(Property{Name: "id", Value: id}, Property{Name: "provider", Value: "p"})
			}
			if err := store.Write(ctx, "p", NewObject(Property{Name: "models", Value: NewArray(model("cached"))})); err != nil {
				t.Fatal(err)
			}
			entered := make(chan struct{})
			release := make(chan struct{})
			var releaseOnce sync.Once
			defer releaseOnce.Do(func() { close(release) })
			sentinel := errors.New("write failed")
			publisher := NewModelsPublisher(ModelsPersistence{Write: func(ctx context.Context, id string, entry any) error {
				close(entered)
				<-release
				if mode == "write-error" {
					return sentinel
				}
				return store.Write(ctx, id, entry)
			}, Delete: store.Delete})
			defer publisher.Clear()
			scope := publisher.Begin(ctx, "p")
			catalog := NewProviderModelCatalog("p", NewArray(model("base")), func(ModelRefreshContext) (*Array, error) { return NewArray(model("fresh")), nil }, func() int64 { return 100 })
			stored, err := store.Read(ctx, "p")
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() {
				done <- catalog.RefreshModels(ModelRefreshContext{Context: scope.Context, Stored: stored.(*Object), Publish: scope.Publish, AllowNetwork: true})
			}()
			publicationAwait(t, entered)
			checkVisible := func(want string) {
				t.Helper()
				models, err := catalog.GetModels()
				if err != nil || models.Len() != 2 || models.Get(1).(*Object).Get("id") != want {
					t.Fatalf("visible models before/after persistence: %v %v", models, err)
				}
			}
			checkVisible("cached")
			if mode == "superseded" {
				publisher.Invalidate("p")
				if err := publicationAwait(t, done); !errors.Is(err, context.Canceled) {
					t.Fatalf("superseded refresh: %v", err)
				}
			}
			releaseOnce.Do(func() { close(release) })
			switch mode {
			case "success":
				if err := publicationAwait(t, done); err != nil {
					t.Fatal(err)
				}
				checkVisible("fresh")
			case "write-error":
				if err := publicationAwait(t, done); !errors.Is(err, sentinel) {
					t.Fatalf("error identity: %v", err)
				}
				checkVisible("cached")
			case "superseded":
				barrier := publisher.Begin(ctx, "p")
				if _, err := barrier.Publish(ModelsPublication{}); err != nil {
					t.Fatal(err)
				}
				checkVisible("cached")
			}
			persisted, err := store.Read(ctx, "p")
			if err != nil {
				t.Fatal(err)
			}
			want := "cached"
			if mode == "success" {
				want = "fresh"
			}
			if id := persisted.(*Object).Get("models").(*Array).Get(0).(*Object).Get("id"); id != want {
				t.Fatalf("persisted model %v, expected %s", id, want)
			}
		})
	}
}
