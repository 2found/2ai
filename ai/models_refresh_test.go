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

type modelsRefreshFixture struct {
	UpstreamCommit        string
	Superseded, Cancelled json.RawMessage
	Cases                 []struct{ Input, Result, Log, Credential, Cached json.RawMessage }
}

func readModelsRefreshFixture(t *testing.T) modelsRefreshFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-models-refresh.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture modelsRefreshFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 253 {
		t.Fatal("unexpected refresh coverage")
	}
	return fixture
}

func summarizeModelsRefresh(result ModelsRefreshResult) any {
	errors := NewArray()
	for _, failure := range result.Errors {
		entry := NewObject(Property{Name: "providerId", Value: failure.ProviderID}, Property{Name: "message", Value: failure.Error.Error()})
		if typed, ok := failure.Error.(*ModelsError); ok {
			entry.Set("code", typed.Code)
		}
		errors.Append(entry)
	}
	return NewObject(Property{Name: "aborted", Value: result.Aborted}, Property{Name: "errors", Value: errors})
}

func TestPiModelsRefresh(t *testing.T) {
	for index, tc := range readModelsRefreshFixture(t).Cases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			input := catalogDecode(t, tc.Input).(*Object)
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			memory := NewInMemoryCredentialStore()
			cache := NewInMemoryModelsStore()
			credential := input.Get("credential").(*Object)
			if credential.Get("absent") != true {
				if _, err := memory.Modify(ctx, "p", func(any) (any, error) { return credential.Get("value"), nil }); err != nil {
					t.Fatal(err)
				}
			}
			cached := input.Get("cached").(*Object)
			if cached.Get("absent") != true {
				if err := cache.Write(ctx, "p", cached.Get("value")); err != nil {
					t.Fatal(err)
				}
			}
			if input.Get("abort") == true {
				cancel(errors.New("cancelled"))
			}
			log := NewArray()
			failure := input.Get("failure")
			credentials := &CredentialPersistence{Read: func(ctx context.Context, id string) (any, error) {
				log.Append("credential-read")
				if failure == "credential-read" {
					return nil, errors.New("credential read failed")
				}
				return memory.Read(ctx, id)
			}, Modify: func(ctx context.Context, id string, modify func(any) (any, error)) (any, error) {
				log.Append("modify")
				if failure == "modify" {
					return nil, errors.New("modify failed")
				}
				switch input.Get("swap") {
				case "logout":
					if err := memory.Delete(ctx, id); err != nil {
						return nil, err
					}
				case "fresh", "apikey":
					next := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "other"}, Property{Name: "expires", Value: 900000})
					if input.Get("swap") == "apikey" {
						next = NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "other"})
					}
					if _, err := memory.Modify(ctx, id, func(any) (any, error) { return next, nil }); err != nil {
						return nil, err
					}
				}
				return memory.Modify(ctx, id, modify)
			}}
			persistence := &ModelsPersistence{Read: func(ctx context.Context, id string) (any, error) {
				log.Append("catalog-read")
				if failure == "catalog-read" {
					return nil, errors.New("catalog read failed")
				}
				return cache.Read(ctx, id)
			}, Write: func(ctx context.Context, id string, entry any) error {
				log.Append("write")
				if failure == "write" {
					return errors.New("write failed")
				}
				return cache.Write(ctx, id, entry)
			}, Delete: cache.Delete}
			provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{}, Name: "Provider", GetModels: func() (*Array, error) { return NewArray(), nil }}
			handlers := input.Get("handlers")
			if handlers == "api" || handlers == "both" {
				provider.Auth.APIKey = &APIKeyAuth{Resolve: func(args APIKeyAuthInput) (any, error) {
					log.Append(NewObject(Property{Name: "api", Value: authResolveCapture(args.Credential, nil)}))
					if failure == "api" {
						return nil, errors.New("api failed")
					}
					if boxed, ok := input.Get("apiResult").(*Object); ok {
						if boxed.Get("absent") == true {
							return Undefined, nil
						}
						return boxed.Get("value"), nil
					}
					key := catalogProperty(args.Credential, "key")
					if jsonjs.IsNullish(key) {
						key = "ambient"
					}
					return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: key})}, Property{Name: "env", Value: catalogProperty(args.Credential, "env")}), nil
				}}
			}
			if handlers == "oauth" || handlers == "both" {
				provider.Auth.OAuth = &OAuthAuth{Refresh: func(ctx context.Context, credential any) (any, error) {
					log.Append(NewObject(Property{Name: "refresh", Value: credential}, Property{Name: "aborted", Value: ctx.Err() != nil}))
					if failure == "refresh" {
						return nil, errors.New("refresh failed")
					}
					return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "rotated"}, Property{Name: "expires", Value: 900000}), nil
				}}
			}
			provider.RefreshModels = func(refresh ModelRefreshContext) error {
				phase := "offline"
				if refresh.AllowNetwork {
					phase = "online"
				}
				var snapshot any = Undefined
				if refresh.Stored != nil {
					var cloner jsonjs.ValueCloner
					snapshot = cloner.Clone(refresh.Stored)
				}
				var force any = Undefined
				if refresh.Force != nil {
					force = *refresh.Force
				}
				log.Append(NewObject(Property{Name: "phase", Value: phase}, Property{Name: "credential", Value: authResolveCapture(refresh.Credential, nil)}, Property{Name: "stored", Value: authResolveCapture(snapshot, nil)}, Property{Name: "force", Value: authResolveCapture(force, nil)}, Property{Name: "aborted", Value: refresh.Context.Err() != nil}))
				if failure == phase {
					return errors.New(phase + " failed")
				}
				if refresh.Stored != nil {
					refresh.Stored.Set("etag", "mutated locally")
				}
				if refresh.AllowNetwork {
					accepted, err := refresh.Publish(ModelsPublication{Persist: NewObject(Property{Name: "models", Value: NewArray(NewObject(Property{Name: "id", Value: "fetched"}, Property{Name: "provider", Value: "p"}))}, Property{Name: "checkedAt", Value: 1000}), Update: func() error {
						log.Append("update")
						if failure == "update" {
							return errors.New("update failed")
						}
						return nil
					}})
					if err != nil {
						return err
					}
					log.Append(NewObject(Property{Name: "accepted", Value: accepted}))
				}
				return nil
			}
			models := NewModels(ModelsOptions{Credentials: credentials, Persistence: persistence})
			models.SetProvider(provider)
			models.SetProvider(&ModelProvider{ID: "static"})
			options := input.Get("options").(*Object)
			settings := ModelsRefreshOptions{Now: func() int64 { return 1000 }}
			if value, ok := options.Get("allowNetwork").(bool); ok {
				settings.AllowNetwork = &value
			}
			if value, ok := options.Get("force").(bool); ok {
				settings.Force = &value
			}
			if providers, ok := options.Get("providers").(*Array); ok {
				settings.Providers = []string{}
				for slot := 0; slot < providers.Len(); slot++ {
					settings.Providers = append(settings.Providers, providers.Get(slot).(string))
				}
			}
			catalogCompare(t, summarizeModelsRefresh(models.Refresh(ctx, settings)), tc.Result)
			catalogCompare(t, log, tc.Log)
			catalogCompare(t, authResolveCapture(memory.Read(context.Background(), "p")), tc.Credential)
			catalogCompare(t, authResolveCapture(cache.Read(context.Background(), "p")), tc.Cached)
		})
	}
}
