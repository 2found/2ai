package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

type providerCatalogFixture struct {
	UpstreamCommit string
	Cases          []struct {
		Input struct {
			Baseline, Stored, Fetched json.RawMessage
			Mode                      string
		}
		Before, Calls, After json.RawMessage
		Error                *string
	}
	HelperCases []struct {
		Model, Type, Chat, API json.RawMessage
		Equals                 []json.RawMessage
	}
	DynamicTypes []struct {
		Type, Persist, All, Chat json.RawMessage
		Error                    *string
	}
	PublicationTrace []json.RawMessage
	PublicationRace  json.RawMessage
}

func readProviderCatalogFixture(t *testing.T) providerCatalogFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-provider-model-catalog.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture providerCatalogFixture
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 160 || len(fixture.HelperCases) != 19 || len(fixture.DynamicTypes) != 14 || len(fixture.PublicationTrace) != 6 {
		t.Fatal("unexpected provider catalog oracle coverage")
	}
	return fixture
}
func catalogCapture(value any, err error) map[string]any {
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	return map[string]any{"value": value}
}
func catalogCompare(t *testing.T, value any, want json.RawMessage) {
	t.Helper()
	actual := diagnosticValue(t, []byte(catalogStringify(t, value)))
	expected := diagnosticValue(t, want)
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("Go %s\nPi %s", catalogStringify(t, value), want)
	}
}
func catalogSnapshot(t *testing.T, p *ProviderModelCatalog) any {
	t.Helper()
	all, allErr := p.GetAllModels()
	chat, chatErr := p.GetModels()
	return map[string]any{"all": catalogCapture(all, allErr), "chat": catalogCapture(chat, chatErr)}
}
func TestPiProviderModelCatalogRefresh(t *testing.T) {
	for index, tc := range readProviderCatalogFixture(t).Cases {
		t.Run(fmt.Sprint(index)+"/"+tc.Input.Mode, func(t *testing.T) {
			mode := tc.Input.Mode
			calls := NewArray()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if mode == "preabort" {
				cancel()
			}
			p := NewProviderModelCatalog("p", catalogDecode(t, tc.Input.Baseline).(*Array), func(r ModelRefreshContext) (*Array, error) {
				calls.Append(NewObject(Property{Name: "fetch", Value: true}, Property{Name: "force", Value: *r.Force}, Property{Name: "credential", Value: r.Credential}))
				if mode == "fetch-error" {
					return nil, errors.New("fetch failed")
				}
				if mode == "fetch-abort" {
					cancel()
				}
				return catalogDecode(t, tc.Input.Fetched).(*Array), nil
			}, func() int64 { return 100 })
			catalogCompare(t, catalogSnapshot(t, p), tc.Before)
			var stored *Object
			if string(tc.Input.Stored) != "null" {
				stored = catalogDecode(t, tc.Input.Stored).(*Object)
			}
			force := true
			err := p.RefreshModels(ModelRefreshContext{Context: ctx, Stored: stored, AllowNetwork: mode != "offline", Force: &force, Credential: NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "fixture"}), Publish: func(pub ModelsPublication) (bool, error) {
				persist := pub.Persist
				if persist == nil {
					persist = "absent"
				}
				calls.Append(NewObject(Property{Name: "persist", Value: persist}))
				if mode == "deny-restore" && pub.Persist == nil || mode == "deny-fetch" && pub.Persist != nil {
					return false, nil
				}
				if mode == "persist-error" && pub.Persist != nil {
					return false, errors.New("persist failed")
				}
				if pub.Update != nil {
					if err := pub.Update(); err != nil {
						return false, err
					}
				}
				return true, nil
			}})
			checkCatalogError(t, err, tc.Error)
			catalogCompare(t, calls, tc.Calls)
			catalogCompare(t, catalogSnapshot(t, p), tc.After)
		})
	}
}
func TestPiModelTypeAndIdentity(t *testing.T) {
	fixture := readProviderCatalogFixture(t)
	for index, tc := range fixture.HelperCases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			model := catalogDecode(t, tc.Model)
			value, err := GetModelType(model)
			catalogCompare(t, catalogCapture(value, err), tc.Type)
			chat, err := IsModelType(model, "chat")
			catalogCompare(t, catalogCapture(chat, err), tc.Chat)
			api, err := HasAPI(model, Undefined)
			catalogCompare(t, catalogCapture(api, err), tc.API)
			for j, expected := range tc.Equals {
				b := catalogDecode(t, fixture.HelperCases[j].Model)
				if index == j {
					b = model
				}
				equal, err := ModelsAreEqual(model, b)
				catalogCompare(t, catalogCapture(equal, err), expected)
			}
		})
	}
}
func TestPiProviderModelCatalogDynamicTypes(t *testing.T) {
	for index, tc := range readProviderCatalogFixture(t).DynamicTypes {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			fetched := NewArray(NewObject(Property{Name: "type", Value: catalogDecode(t, tc.Type)}, Property{Name: "id", Value: "a"}, Property{Name: "provider", Value: "p"}))
			p := NewProviderModelCatalog("p", NewArray(), func(ModelRefreshContext) (*Array, error) { return fetched, nil }, func() int64 { return 100 })
			var persisted any
			err := p.RefreshModels(ModelRefreshContext{Context: context.Background(), AllowNetwork: true, Publish: func(pub ModelsPublication) (bool, error) { persisted = pub.Persist; return true, pub.Update() }})
			checkCatalogError(t, err, tc.Error)
			catalogCompare(t, persisted, tc.Persist)
			all, allErr := p.GetAllModels()
			chat, chatErr := p.GetModels()
			catalogCompare(t, catalogCapture(all, allErr), tc.All)
			catalogCompare(t, catalogCapture(chat, chatErr), tc.Chat)
		})
	}
}
func TestProviderModelCatalogLiveReferences(t *testing.T) {
	model := NewObject(Property{Name: "id", Value: "a"})
	baseline := NewArray(model)
	p := NewProviderModelCatalog("p", baseline, nil, nil)
	if p.RefreshModels != nil {
		t.Fatal("static provider has refresh")
	}
	first, err := p.GetModels()
	if err != nil {
		t.Fatal(err)
	}
	model.Set("name", "changed")
	baseline.Append(NewObject(Property{Name: "id", Value: "b"}))
	second, err := p.GetModels()
	if err != nil {
		t.Fatal(err)
	}
	if first.Len() != 1 || second.Len() != 2 || first.Get(0) != model || second.Get(0) != model || first == second {
		t.Fatal("catalog copy/identity differs")
	}
	sparse := NewArray()
	sparse.SetLength(2)
	sparse.Set(1, model)
	p = NewProviderModelCatalog("p", sparse, nil, nil)
	all, err := p.GetAllModels()
	if err != nil || !all.Has(0) || !jsonjs.IsUndefined(all.Get(0)) {
		t.Fatal("spread did not materialize hole")
	}
	if _, err = p.GetModels(); err == nil {
		t.Fatal("explicit undefined model must fail filtering")
	}
}
