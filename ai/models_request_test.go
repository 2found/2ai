package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

type modelsRequestFixture struct {
	UpstreamCommit string
	Races          []struct {
		Operation, Stage           string
		Fail, SettledBeforeRelease bool
		Result, ProviderAborted    json.RawMessage
		DispatchCalls              int
	}
	Cases []struct {
		Input, Result, Log, OriginalModel json.RawMessage
		SameReturn                        bool
	}
}

func readModelsRequestFixture(t *testing.T) modelsRequestFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-models-request.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture modelsRequestFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 186 || len(fixture.Races) != 12 {
		t.Fatal("unexpected request coverage")
	}
	return fixture
}
func requestPropertyNames(value any) *Array {
	result := NewArray()
	if object, ok := value.(*Object); ok {
		for _, property := range object.Entries() {
			result.Append(property.Name)
		}
	}
	return result
}

func TestPiModelsRequestDispatch(t *testing.T) {
	for index, tc := range readModelsRequestFixture(t).Cases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			input := catalogDecode(t, tc.Input).(*Object)
			operation, mode := input.Get("operation"), input.Get("mode")
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			if mode == "pre-abort" {
				cancel(errors.New("cancelled"))
			}
			kind := "chat"
			if operation == "images" {
				kind = "image"
			}
			if operation == "classify" {
				kind = "classifier"
			}
			if mode == "wrong-type" {
				kind = "future"
			}
			var model any = NewObject(Property{Name: "id", Value: "m"}, Property{Name: "provider", Value: "p"}, Property{Name: "api", Value: "test"}, Property{Name: "type", Value: kind}, Property{Name: "baseUrl", Value: "https://model.invalid"})
			if input.Get("noModelHeaders") != true {
				model.(*Object).Set("headers", NewObject(Property{Name: "AUTHORIZATION", Value: "model"}, Property{Name: "X-Model", Value: "yes"}))
			}
			if boxed, ok := input.Get("model").(*Object); ok {
				if boxed.Get("absent") == true {
					model = Undefined
				} else {
					model = boxed.Get("value")
				}
			}
			payload := NewObject(Property{Name: "payload", Value: "same-input"})
			handle := NewObject(Property{Name: "id", Value: "handle"})
			returned := NewObject(Property{Name: "result", Value: "same-return"})
			values := authSpread(input.Get("request"))
			signal := NewObject()
			values.Set("signal", signal)
			settings := &ModelsRequestOptions{Values: values, Now: func() int64 { return 1000 }}
			log := NewArray()
			dispatch := func(callCtx context.Context, m, arg any, options *Object) (any, error) {
				sameInput := arg == any(payload)
				if operation == "cancel" {
					sameInput = arg == any(handle)
				}
				_, hasTransform := options.Lookup("transformHeaders")
				log.Append(NewObject(Property{Name: "dispatch", Value: operation}, Property{Name: "model", Value: m}, Property{Name: "options", Value: options}, Property{Name: "optionKeys", Value: requestPropertyNames(options)}, Property{Name: "headerKeys", Value: requestPropertyNames(options.Get("headers"))}, Property{Name: "sameModel", Value: catalogStrictEqual(m, model)}, Property{Name: "sameInput", Value: sameInput}, Property{Name: "signalIdentity", Value: callCtx == ctx && options.Get("signal") == signal}, Property{Name: "hasTransform", Value: hasTransform}))
				if mode == "provider-error" || mode == "abort-provider-error" {
					if mode == "abort-provider-error" {
						cancel(errors.New("cancelled"))
					}
					return nil, errors.New("provider failed")
				}
				return returned, nil
			}
			provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{}, Name: "Provider"}
			setDispatch := func(p *ModelProvider, fn func(context.Context, any, any, *Object) (any, error)) {
				if operation == "cancel" {
					p.CancelDeferred = func(ctx context.Context, m, h any, o *Object) error { _, err := fn(ctx, m, h, o); return err }
				} else if operation == "images" {
					p.GenerateImages = fn
				} else {
					p.Classify = fn
				}
			}
			setDispatch(provider, dispatch)
			if mode == "missing-handler" {
				provider.CancelDeferred = nil
				provider.GenerateImages = nil
				provider.Classify = nil
			}
			var models *Models
			provider.Auth.APIKey = &APIKeyAuth{Resolve: func(args APIKeyAuthInput) (any, error) {
				log.Append(NewObject(Property{Name: "resolve", Value: authResolveCapture(args.Credential, nil)}))
				if mode == "resolve-error" {
					return nil, errors.New("resolve failed")
				}
				if mode == "no-auth" {
					return Undefined, nil
				}
				if mode == "replace-method" {
					setDispatch(provider, func(context.Context, any, any, *Object) (any, error) {
						log.Append("replacement-method")
						return returned, nil
					})
				}
				if mode == "replace-provider" {
					replacement := *provider
					setDispatch(&replacement, func(context.Context, any, any, *Object) (any, error) {
						log.Append("replacement-provider")
						return returned, nil
					})
					models.SetProvider(&replacement)
				}
				if mode == "mutate-options" {
					values.Set("custom", "changed")
					values.Set("headers", NewObject(Property{Name: "X-Late", Value: "late"}))
					settings.TransformHeaders = func(h any) (any, error) { log.Append("replacement-transform"); return h, nil }
				}
				if boxed, ok := input.Get("resolution").(*Object); ok {
					if boxed.Get("absent") == true {
						return Undefined, nil
					}
					return boxed.Get("value"), nil
				}
				var cloner jsonjs.ValueCloner
				return NewObject(Property{Name: "auth", Value: cloner.Clone(input.Get("auth"))}, Property{Name: "env", Value: NewObject(Property{Name: "REGION", Value: "auth"}, Property{Name: "REMOVE", Value: "keep"})}), nil
			}}
			models = NewModels(ModelsOptions{Credentials: &CredentialPersistence{Read: func(context.Context, string) (any, error) {
				log.Append("read")
				if mode == "read-error" {
					return nil, errors.New("read failed")
				}
				return Undefined, nil
			}}})
			if mode != "unknown-provider" {
				models.SetProvider(provider)
			}
			if input.Get("transform") != "none" {
				values.Set("transformHeaders", Undefined)
				settings.TransformHeaders = func(headers any) (any, error) {
					var cloner jsonjs.ValueCloner
					log.Append(NewObject(Property{Name: "transform", Value: cloner.Clone(headers)}))
					if mode == "abort-transform" || mode == "abort-transform-error" {
						cancel(errors.New("cancelled"))
					}
					if mode == "transform-error" || mode == "abort-transform-error" {
						return nil, errors.New("transform failed")
					}
					if input.Get("transform") == "null" {
						return Null, nil
					}
					result := authSpread(headers)
					result.Set("X-Transformed", "yes")
					return result, nil
				}
			}
			var result any
			var err error
			if operation == "cancel" {
				err = models.CancelDeferred(ctx, model, handle, settings)
				result = Undefined
			} else if operation == "images" {
				result, err = models.GenerateImages(ctx, model, payload, settings)
			} else {
				result, err = models.Classify(ctx, model, payload, settings)
			}
			catalogCompare(t, authResolveCapture(result, err), tc.Result)
			catalogCompare(t, log, tc.Log)
			catalogCompare(t, model, tc.OriginalModel)
			if same := operation != "cancel" && result == any(returned); same != tc.SameReturn {
				t.Fatal("provider result identity differs")
			}
		})
	}
}
