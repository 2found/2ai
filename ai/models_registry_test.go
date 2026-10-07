package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

type modelsRegistryFixture struct {
	UpstreamCommit string
	Definitions    []struct {
		ID        string
		Chat, All json.RawMessage
	}
	Steps []struct {
		Op struct {
			Set    *int
			Delete *string
			Clear  bool
		}
		Providers []string
		Probes    []struct {
			Provider                   *string
			Chat, All, Find, FindImage json.RawMessage
			Types                      []json.RawMessage
		}
	}
	Mutations []struct {
		Mode             string
		All              bool
		Calls, Providers []string
		Models           json.RawMessage
	}
	References   json.RawMessage
	StoreBinding json.RawMessage
	Scopes       []struct {
		Mode       string
		Aborted    bool
		NewAborted *bool
		Updates    int
		Result     json.RawMessage
	}
}

func readModelsRegistryFixture(t *testing.T) modelsRegistryFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-models-registry.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture modelsRegistryFixture
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Steps) != 14 || len(fixture.Mutations) != 10 || len(fixture.Scopes) != 10 {
		t.Fatal("unexpected registry oracle coverage")
	}
	return fixture
}
func registryCapture(value any, err error) any {
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	if jsonjs.IsUndefined(value) {
		return map[string]any{"absent": true}
	}
	if array, ok := value.(*Array); ok && array == nil {
		value = nil // A null callback result, rather than an empty native array.
	}
	return map[string]any{"value": value}
}
func registryIDs(m *Models) []string {
	ids := []string{}
	for _, p := range m.GetProviders() {
		ids = append(ids, p.ID)
	}
	return ids
}
func TestPiModelsRegistry(t *testing.T) {
	fixture := readModelsRegistryFixture(t)
	m := NewModels()
	defer m.ClearProviders()
	providers := make([]*ModelProvider, len(fixture.Definitions))
	for i, d := range fixture.Definitions {
		callback := func(raw json.RawMessage) func() (*Array, error) {
			if string(raw) == `"throw"` {
				return func() (*Array, error) { return nil, errors.New("bad catalog") }
			}
			var models *Array
			if string(raw) != "null" {
				models = catalogDecode(t, raw).(*Array)
			}
			return func() (*Array, error) { return models, nil }
		}
		providers[i] = &ModelProvider{ID: d.ID, Name: d.ID, GetModels: callback(d.Chat)}
		if len(d.All) > 0 {
			providers[i].GetAllModels = callback(d.All)
		}
	}
	for i, step := range fixture.Steps {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			switch {
			case step.Op.Set != nil:
				m.SetProvider(providers[*step.Op.Set])
			case step.Op.Delete != nil:
				m.DeleteProvider(*step.Op.Delete)
			default:
				m.ClearProviders()
			}
			catalogCompare(t, registryIDs(m), mustRegistryJSON(t, step.Providers))
			for j, probe := range step.Probes {
				t.Run(fmt.Sprint(j), func(t *testing.T) {
					var scope []string
					findProvider := "a"
					if probe.Provider != nil {
						scope = []string{*probe.Provider}
						findProvider = *probe.Provider
					}
					catalogCompare(t, registryCapture(m.GetModels(scope...), nil), probe.Chat)
					catalogCompare(t, registryCapture(m.GetAllModels(scope...), nil), probe.All)
					for k, kind := range []string{"chat", "image", "classifier", "future"} {
						models, err := m.GetModelsOfType(kind, scope...)
						catalogCompare(t, registryCapture(models, err), probe.Types[k])
					}
					model, err := m.GetModel(findProvider, "same")
					catalogCompare(t, registryCapture(model, err), probe.Find)
					model, err = m.GetModelOfType("image", findProvider, "same")
					catalogCompare(t, registryCapture(model, err), probe.FindImage)
				})
			}
		})
	}
}
func mustRegistryJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
func TestPiModelsRegistryReentrantMutation(t *testing.T) {
	for _, tc := range readModelsRegistryFixture(t).Mutations {
		t.Run(fmt.Sprintf("%s/all=%t", tc.Mode, tc.All), func(t *testing.T) {
			m := NewModels()
			defer m.ClearProviders()
			calls := []string{}
			once := false
			list := func(id string) *Array { return NewArray(NewObject(Property{Name: "id", Value: id})) }
			next := &ModelProvider{ID: "c", GetModels: func() (*Array, error) { calls = append(calls, "c"); return list("c"), nil }}
			a := &ModelProvider{ID: "a"}
			a.GetModels = func() (*Array, error) {
				calls = append(calls, "a")
				if !once {
					once = true
					switch tc.Mode {
					case "append":
						m.SetProvider(next)
					case "delete-next":
						m.DeleteProvider("b")
					case "replace-next":
						m.SetProvider(&ModelProvider{ID: "b", GetModels: func() (*Array, error) { calls = append(calls, "b2"); return list("b2"), nil }})
					case "clear-add":
						m.ClearProviders()
						m.SetProvider(next)
					case "delete-self-add":
						m.DeleteProvider("a")
						m.SetProvider(a)
					}
				}
				return list("a"), nil
			}
			m.SetProvider(a)
			m.SetProvider(&ModelProvider{ID: "b", GetModels: func() (*Array, error) { calls = append(calls, "b"); return list("b"), nil }})
			var models *Array
			if tc.All {
				models = m.GetAllModels()
			} else {
				models = m.GetModels()
			}
			catalogCompare(t, models, tc.Models)
			catalogCompare(t, calls, mustRegistryJSON(t, tc.Calls))
			catalogCompare(t, registryIDs(m), mustRegistryJSON(t, tc.Providers))
			if m.iterators != 0 || len(m.order) != len(m.providers) {
				t.Fatal("completed iterator retained tombstones")
			}
		})
	}
}
func TestPiModelsRegistryReferences(t *testing.T) {
	m := NewModels()
	models := NewArray()
	models.SetLength(3)
	models.Set(1, NewObject(Property{Name: "id", Value: "a"}))
	p := &ModelProvider{ID: "p", GetModels: func() (*Array, error) { return models, nil }}
	m.SetProvider(p)
	scoped, union := m.GetModels("p"), m.GetModels()
	found, findErr := m.GetModel("p", "a")
	typed, typeErr := m.GetModelsOfType("chat", "p")
	actual := map[string]any{"scoped": scoped == models, "union": union == models, "scopedKeys": scoped.PropertyKeys(), "unionKeys": union.PropertyKeys(), "provider": m.GetProvider("p") == p, "find": registryCapture(found, findErr), "typed": registryCapture(typed, typeErr)}
	catalogCompare(t, actual, readModelsRegistryFixture(t).References)
}
func TestPiModelsRefreshScopeLifecycle(t *testing.T) {
	for _, tc := range readModelsRegistryFixture(t).Scopes {
		t.Run(tc.Mode, func(t *testing.T) {
			m := NewModels()
			defer m.ClearProviders()
			p := &ModelProvider{ID: "p", GetModels: func() (*Array, error) { return NewArray(), nil }}
			if !strings.Contains(tc.Mode, "unregistered") {
				m.SetProvider(p)
			}
			parent, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			old := m.publisher.Begin(parent, "p")
			updates := 0
			if strings.HasPrefix(tc.Mode, "finish") {
				old.Finish()
			}
			var newer *ModelsPublicationScope
			switch tc.Mode {
			case "finish-begin":
				newer = m.publisher.Begin(parent, "p")
			case "finish-replace", "active-replace":
				m.SetProvider(p)
			case "finish-delete":
				m.DeleteProvider("p")
			case "finish-clear", "finished-unregistered-clear", "active-unregistered-clear":
				m.ClearProviders()
			case "old-finish":
				newer = m.publisher.Begin(parent, "p")
				old.Finish()
				m.DeleteProvider("p")
			case "finish-parent-abort":
				cancel(errors.New("caller cancelled"))
			}
			accepted, err := old.Publish(ModelsPublication{Update: func() error { updates++; return nil }})
			var result any = map[string]any{"accepted": accepted}
			if err != nil {
				result = map[string]any{"cancelled": old.Context.Err() != nil}
			}
			catalogCompare(t, result, tc.Result)
			if aborted := old.Context.Err() != nil; aborted != tc.Aborted || updates != tc.Updates {
				t.Fatalf("aborted=%t updates=%d; Pi %t %d", aborted, updates, tc.Aborted, tc.Updates)
			}
			if newer != nil && (newer.Context.Err() != nil) != *tc.NewAborted {
				t.Fatal("old Finish retired replacement controller")
			}
		})
	}
}
func TestModelsRegistryConcurrentAndFailingProviders(t *testing.T) {
	m := NewModels()
	defer m.ClearProviders()
	p := &ModelProvider{ID: "panic", GetModels: func() (*Array, error) { panic(errors.New("bad catalog")) }}
	m.SetProvider(p)
	if m.GetModels("panic").Len() != 0 {
		t.Fatal("catalog panic escaped")
	}
	var workers sync.WaitGroup
	for worker := range 8 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			id := fmt.Sprint(worker)
			list := NewArray(NewObject(Property{Name: "id", Value: id}))
			provider := &ModelProvider{ID: id, GetModels: func() (*Array, error) { return list, nil }}
			for range 50 {
				m.SetProvider(provider)
				_ = m.GetModels()
				_ = m.GetAllModels()
				m.DeleteProvider(id)
			}
		}()
	}
	workers.Wait()
	if m.iterators != 0 || len(m.order) != 1 {
		t.Fatal("registry retained deleted providers")
	}
}

func TestPiModelsStoreCallbackReplacement(t *testing.T) {
	calls := NewArray()
	persistence := &ModelsPersistence{
		Write:  func(context.Context, string, any) error { calls.Append("old-write"); return nil },
		Delete: func(context.Context, string) error { calls.Append("old-delete"); return nil },
	}
	models := NewModels(ModelsOptions{Persistence: persistence})
	defer models.ClearProviders()
	scope := models.publisher.Begin(context.Background(), "p")
	persistence.Write = func(_ context.Context, id string, entry any) error {
		calls.Append(NewArray("new-write", id, entry))
		return nil
	}
	persistence.Delete = func(_ context.Context, id string) error { calls.Append(NewArray("new-delete", id)); return nil }
	for _, persist := range []any{NewObject(Property{Name: "models", Value: NewArray()}), Null} {
		if accepted, err := scope.Publish(ModelsPublication{Persist: persist}); !accepted || err != nil {
			t.Fatalf("publication %t: %v", accepted, err)
		}
	}
	catalogCompare(t, calls, readModelsRegistryFixture(t).StoreBinding)
}
