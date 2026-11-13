package ai

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"
	"time"
)

type providerFactoryFixture struct {
	UpstreamCommit string
	Constructors   []struct {
		Kind   string
		Name   *string
		Output json.RawMessage
	}
	Cases []struct {
		Method, Mode string
		Output, Log  json.RawMessage
	}
	References json.RawMessage
}

func readProviderFactoryFixture(t *testing.T) providerFactoryFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-provider-factory.json")
	if err != nil {
		t.Fatal(err)
	}
	var f providerFactoryFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Constructors) != 21 || len(f.Cases) != 90 {
		t.Fatal("unexpected factory fixture coverage")
	}
	return f
}
func factorySnapshot(t *testing.T, p *ModelProvider, auth *ProviderAuth) map[string]any {
	t.Helper()
	chat, err := p.GetModels()
	if err != nil {
		t.Fatal(err)
	}
	all, err := p.GetAllModels()
	if err != nil {
		t.Fatal(err)
	}
	return map[string]any{"id": p.ID, "name": p.Name, "authSame": p.Auth == auth, "chat": chat, "all": all, "fetch": p.FetchDeferred != nil, "cancel": p.CancelDeferred != nil, "images": p.GenerateImages != nil, "classify": p.Classify != nil}
}
func TestPiProviderFactoryConstruction(t *testing.T) {
	for _, tc := range readProviderFactoryFixture(t).Constructors {
		t.Run(tc.Kind+"/"+func() string {
			if tc.Name == nil {
				return "default"
			}
			return *tc.Name
		}(), func(t *testing.T) {
			input := &ProviderFactoryOptions{ID: "p", Name: tc.Name, Auth: &ProviderAuth{}, Models: NewArray(), BaseURL: "metadata", Headers: NewObject(Property{Name: "x", Value: "meta"})}
			basic := &ProviderStreams{Stream: func(context.Context, any, TranscriptContext, *Object) (*ProviderEventSource, error) { return nil, nil }}
			switch tc.Kind {
			case "undefined":
				input.APIs = map[string]*ProviderStreams{"a": nil}
				input.Images = map[string]*ProviderImages{"a": nil}
				input.Classifiers = map[string]*ProviderClassifier{"a": nil}
			case "single":
				input.API = basic
			case "map", "mixed":
				input.APIs = map[string]*ProviderStreams{"a": basic, "b": {Stream: basic.Stream, FetchDeferred: func(context.Context, any, any, *Object) (*ProviderEventSource, error) { return nil, nil }, CancelDeferred: func(context.Context, any, any, *Object) error { return nil }}}
			}
			if tc.Kind == "images" || tc.Kind == "mixed" {
				input.Images = map[string]*ProviderImages{"a": {}}
			}
			if tc.Kind == "classifiers" || tc.Kind == "mixed" {
				input.Classifiers = map[string]*ProviderClassifier{"a": {}}
			}
			p, err := NewModelProvider(input)
			if err != nil {
				catalogCompare(t, map[string]any{"error": err.Error()}, tc.Output)
				return
			}
			output := factorySnapshot(t, p, input.Auth)
			output["baseUrl"], output["headers"], output["refresh"] = p.BaseURL, p.Headers, p.RefreshModels != nil
			catalogCompare(t, output, tc.Output)
		})
	}
}
func TestPiProviderFactoryDispatch(t *testing.T) {
	for _, tc := range readProviderFactoryFixture(t).Cases {
		t.Run(tc.Method+"/"+tc.Mode, func(t *testing.T) {
			log := NewArray()
			message := fixtureStreamMessage()
			step := 0
			source := &ProviderEventSource{Next: func(context.Context) (AssistantMessageEvent, bool, error) {
				step++
				return AssistantMessageEvent{Type: "done", Reason: "stop", Message: message}, step == 1, nil
			}, Result: func(context.Context) (*Message, bool, error) { return message, true, nil }}
			model := NewObject(Property{Name: "id", Value: "m"}, Property{Name: "provider", Value: "p"}, Property{Name: "api", Value: "a"})
			transcript := NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent("identity")}}})
			request := NewObject()
			options := NewObject(Property{Name: "custom", Value: "same"})
			handle := NewObject(Property{Name: "id", Value: "h"})
			cause := errors.New("implementation failed")
			call := func(label, kind string, m, c any, o *Object) (any, error) {
				same := c == request
				if kind == "stream" || kind == "streamSimple" {
					same = reflect.ValueOf(c.(TranscriptContext).messages).Pointer() == reflect.ValueOf(transcript.messages).Pointer()
				}
				if kind == "fetchDeferred" || kind == "cancelDeferred" {
					same = c == handle
				}
				log.Append(NewObject(Property{Name: "label", Value: label}, Property{Name: "kind", Value: kind}, Property{Name: "modelSame", Value: m == model}, Property{Name: "contextSame", Value: same}, Property{Name: "optionsSame", Value: o == options}))
				if tc.Mode == "throw" {
					return nil, cause
				}
				return NewObject(Property{Name: "label", Value: label}), nil
			}
			impl := func(label string) *ProviderStreams {
				stream := func(kind string) ModelStreamFunc {
					return func(_ context.Context, m any, c TranscriptContext, o *Object) (*ProviderEventSource, error) {
						_, err := call(label, kind, m, c, o)
						return source, err
					}
				}
				return &ProviderStreams{Stream: stream("stream"), StreamSimple: stream("streamSimple"), FetchDeferred: func(_ context.Context, m, c any, o *Object) (*ProviderEventSource, error) {
					_, err := call(label, "fetchDeferred", m, c, o)
					return source, err
				}, CancelDeferred: func(_ context.Context, m, c any, o *Object) error {
					_, err := call(label, "cancelDeferred", m, c, o)
					return err
				}}
			}
			imageImpl := func(label string) *ProviderImages {
				return &ProviderImages{GenerateImages: func(_ context.Context, m, c any, o *Object) (any, error) {
					return call(label, "generateImages", m, c, o)
				}}
			}
			classifierImpl := func(label string) *ProviderClassifier {
				return &ProviderClassifier{Classify: func(_ context.Context, m, c any, o *Object) (any, error) { return call(label, "classify", m, c, o) }}
			}
			apis := map[string]*ProviderStreams{"a": impl("original")}
			images := map[string]*ProviderImages{"a": imageImpl("original")}
			classifiers := map[string]*ProviderClassifier{"a": classifierImpl("original")}
			input := &ProviderFactoryOptions{ID: "p", Auth: &ProviderAuth{}, Models: NewArray(), APIs: apis, Images: images, Classifiers: classifiers, Now: func() int64 { return 1000 }}
			if tc.Mode == "single" {
				input.API = apis["a"]
				input.APIs = nil
			}
			p, err := NewModelProvider(input)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch tc.Mode {
			case "missing", "aborted-missing", "live-id", "single":
				model.Set("api", "absent")
			case "nonstring":
				model.Set("api", 17)
			case "array-key":
				model.Set("api", NewArray("a"))
			case "object-key":
				model.Set("api", NewObject())
			case "null-key":
				model.Set("api", Null)
			case "undefined-key":
				model.Set("api", Undefined)
			case "replacement":
				apis["a"], images["a"], classifiers["a"] = impl("replaced"), imageImpl("replaced"), classifierImpl("replaced")
			case "replace-input-map":
				input.APIs = map[string]*ProviderStreams{"a": impl("unused")}
				input.Images = map[string]*ProviderImages{}
				input.Classifiers = map[string]*ProviderClassifier{}
			case "remove-capability":
				apis["a"].FetchDeferred = nil
				apis["a"].CancelDeferred = nil
			case "delete-entry":
				delete(apis, "a")
				delete(images, "a")
				delete(classifiers, "a")
			}
			if tc.Mode == "live-id" {
				input.ID = "changed"
			}
			if tc.Mode == "aborted-missing" {
				cancel()
			}
			var output any
			var returned *ProviderEventSource
			switch tc.Method {
			case "stream":
				returned, err = p.Stream(ctx, model, transcript, options)
			case "streamSimple":
				returned, err = p.StreamSimple(ctx, model, transcript, options)
			case "fetchDeferred":
				returned, err = p.FetchDeferred(ctx, model, handle, options)
			case "cancelDeferred":
				err = p.CancelDeferred(ctx, model, handle, options)
			case "generateImages":
				output, err = p.GenerateImages(ctx, model, request, options)
			case "classify":
				output, err = p.Classify(ctx, model, request, options)
			}
			if err != nil {
				var code any
				var me *ModelsError
				if errors.As(err, &me) {
					code = me.Code
				}
				if tc.Mode == "throw" && err != cause {
					t.Fatal("implementation error identity lost")
				}
				output = map[string]any{"error": err.Error(), "code": code}
			} else if returned != nil {
				if tc.Method != "fetchDeferred" && log.Len() > 0 && returned != source {
					t.Fatal("direct stream identity lost")
				}
				events := NewArray()
				readCtx, stop := context.WithTimeout(context.Background(), time.Second)
				defer stop()
				for {
					event, ok, e := returned.Next(readCtx)
					if e != nil {
						t.Fatal(e)
					}
					if !ok {
						break
					}
					events.Append(streamFixtureWire(event))
				}
				result, present, e := returned.Result(readCtx)
				if e != nil || !present {
					t.Fatalf("result: %v, %v", present, e)
				}
				output = map[string]any{"events": events, "result": streamFixtureWire(result)}
			}
			catalogCompare(t, output, tc.Output)
			catalogCompare(t, log, tc.Log)
		})
	}
}
func TestPiProviderFactoryReferences(t *testing.T) {
	model := func(id, provider string) *Object {
		return NewObject(Property{Name: "id", Value: id}, Property{Name: "provider", Value: provider}, Property{Name: "api", Value: "a"})
	}
	baseline := NewArray(model("base", "p"))
	auth := &ProviderAuth{}
	apis := map[string]*ProviderStreams{"a": {}}
	images := map[string]*ProviderImages{}
	classifiers := map[string]*ProviderClassifier{}
	fetches := 0
	input := &ProviderFactoryOptions{ID: "p", Auth: auth, Models: baseline, APIs: apis, Images: images, Classifiers: classifiers, Now: func() int64 { return 1000 }, FetchModels: func(ModelRefreshContext) (*Array, error) { fetches++; return NewArray(model("fresh", "changed")), nil }}
	p, err := NewModelProvider(input)
	if err != nil {
		t.Fatal(err)
	}
	input.Models = NewArray()
	image := model("image", "p")
	image.Set("type", "image")
	baseline.Append(image)
	input.Auth = &ProviderAuth{}
	auth.APIKey = &APIKeyAuth{Name: "live"}
	input.FetchModels = func(ModelRefreshContext) (*Array, error) { t.Error("replacement fetch called"); return nil, nil }
	input.ID = "changed"
	apis["a"] = &ProviderStreams{FetchDeferred: func(context.Context, any, any, *Object) (*ProviderEventSource, error) { return nil, nil }, CancelDeferred: func(context.Context, any, any, *Object) error { return nil }}
	images["a"] = &ProviderImages{}
	classifiers["a"] = &ProviderClassifier{}
	initial := factorySnapshot(t, p, auth)
	initial["authName"] = p.Auth.APIKey.Name
	publications := NewArray()
	publish := func(entry ModelsPublication) (bool, error) {
		var persisted any = entry.Persist
		publications.Append(persisted)
		return true, entry.Update()
	}
	err = p.RefreshModels(ModelRefreshContext{Context: context.Background(), Stored: NewObject(Property{Name: "models", Value: NewArray(model("old", "p"), model("restored", "changed"))}), Publish: publish})
	if err != nil {
		t.Fatal(err)
	}
	restored, err := p.GetAllModels()
	if err != nil {
		t.Fatal(err)
	}
	if err = p.RefreshModels(ModelRefreshContext{Context: context.Background(), AllowNetwork: true, Publish: publish}); err != nil {
		t.Fatal(err)
	}
	fetched, err := p.GetAllModels()
	if err != nil {
		t.Fatal(err)
	}
	catalogCompare(t, map[string]any{"initial": initial, "restored": restored, "fetched": fetched, "fetches": fetches, "publications": publications}, readProviderFactoryFixture(t).References)
}
