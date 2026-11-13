package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"
)

type modelsAvailabilityFixture struct {
	UpstreamCommit             string
	Concurrent, Cancelled, IDs json.RawMessage
	Cases                      []struct{ Input, Result, Log json.RawMessage }
}

func readModelsAvailabilityFixture(t *testing.T) modelsAvailabilityFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-models-availability.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture modelsAvailabilityFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 195 {
		t.Fatal("unexpected availability coverage")
	}
	return fixture
}
func TestPiModelsAvailability(t *testing.T) {
	for index, tc := range readModelsAvailabilityFixture(t).Cases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			input := catalogDecode(t, tc.Input).(*Object)
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			memory := NewInMemoryCredentialStore()
			credential := input.Get("credential").(*Object)
			if credential.Get("absent") != true {
				if _, err := memory.Modify(ctx, "p", func(any) (any, error) { return credential.Get("value"), nil }); err != nil {
					t.Fatal(err)
				}
			}
			log := NewArray()
			reads := 0
			failure := input.Get("failure")
			store := &CredentialPersistence{Read: func(ctx context.Context, id string) (any, error) {
				reads++
				log.Append(NewObject(Property{Name: "read", Value: id}))
				if failure == "read" || (reads == 2 && failure == "second-read") {
					return nil, errors.New("read failed")
				}
				if second, ok := input.Get("second").(*Object); reads == 2 && ok {
					if _, err := memory.Modify(ctx, id, func(any) (any, error) { return second, nil }); err != nil {
						return nil, err
					}
				}
				return memory.Read(ctx, id)
			}}
			if input.Get("abort") == true {
				cancel(errors.New("cancelled"))
			}
			provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{}}
			handlers := input.Get("handlers")
			if handlers == "check" || handlers == "resolve" || handlers == "both" {
				provider.Auth.APIKey = &APIKeyAuth{Resolve: func(args APIKeyAuthInput) (any, error) {
					log.Append(NewObject(Property{Name: "resolve", Value: authResolveCapture(args.Credential, nil)}))
					if failure == "resolve" {
						return nil, errors.New("resolve failed")
					}
					return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: "resolved"})}, Property{Name: "source", Value: "resolved"}), nil
				}}
			}
			if handlers == "check" || handlers == "both" {
				provider.Auth.APIKey.Check = func(args APIKeyAuthInput) (any, error) {
					log.Append(NewObject(Property{Name: "check", Value: authResolveCapture(args.Credential, nil)}, Property{Name: "aborted", Value: args.Context.Err() != nil}))
					if failure == "check" {
						return nil, errors.New("check failed")
					}
					if boxed, ok := input.Get("checkResult").(*Object); ok {
						if boxed.Get("absent") == true {
							return Undefined, nil
						}
						return boxed.Get("value"), nil
					}
					return NewObject(Property{Name: "source", Value: "checked"}, Property{Name: "type", Value: "api_key"}), nil
				}
			}
			if handlers == "oauth" || handlers == "both" || handlers == "resolve" {
				provider.Auth.OAuth = &OAuthAuth{Refresh: func(context.Context, any) (any, error) { return nil, errors.New("unexpected refresh") }, ToAuth: func(c any) (any, error) {
					log.Append(NewObject(Property{Name: "toAuth", Value: c}))
					return NewObject(Property{Name: "apiKey", Value: catalogProperty(c, "access")}), nil
				}}
			}
			model := func(id, kind string) *Object {
				v := NewObject(Property{Name: "id", Value: id}, Property{Name: "provider", Value: "p"})
				if kind != "" {
					v.Set("type", kind)
				}
				return v
			}
			c1, c2 := model("c1", ""), model("c2", "chat")
			chat := NewArray(c1, c2)
			all := NewArray(c1, model("i", "image"), model("k", "classifier"), model("f", "future"), c2)
			provider.GetModels = func() (*Array, error) {
				log.Append("models")
				if failure == "models" {
					return nil, errors.New("models failed")
				}
				return chat, nil
			}
			provider.GetAllModels = func() (*Array, error) {
				log.Append("all-models")
				if failure == "all-models" {
					return nil, errors.New("all models failed")
				}
				if raw, exists := input.Lookup("allModels"); exists {
					value, _ := raw.(*Array)
					return value, nil
				}
				return all, nil
			}
			policy := input.Get("policy")
			if policy == "chat" || policy == "both" || policy == "nil-chat" || policy == "sparse" {
				provider.FilterModels = func(models *Array, c any) (*Array, error) {
					log.Append(NewObject(Property{Name: "filter", Value: authResolveCapture(c, nil)}, Property{Name: "sameModels", Value: models == chat}))
					if failure == "filter" {
						return nil, errors.New("filter failed")
					}
					if policy == "nil-chat" {
						return nil, nil
					}
					if policy == "sparse" {
						result := NewArray()
						result.Set(2, c1)
						return result, nil
					}
					if catalogProperty(c, "type") == "oauth" {
						return NewArray(c2), nil
					}
					return NewArray(c1), nil
				}
			}
			if policy == "all" || policy == "both" || policy == "nil-all" {
				provider.FilterAllModels = func(models *Array, c any) (*Array, error) {
					log.Append(NewObject(Property{Name: "allFilter", Value: authResolveCapture(c, nil)}))
					if failure == "all-filter" {
						return nil, errors.New("all filter failed")
					}
					if policy == "nil-all" {
						return nil, nil
					}
					result := NewArray()
					start := models.Len() - 2
					if start < 0 {
						start = 0
					}
					for slot := start; slot < models.Len(); slot++ {
						result.Append(models.Get(slot))
					}
					return result, nil
				}
			}
			models := NewModels(ModelsOptions{Credentials: store})
			models.SetProvider(provider)
			targets := []string{"p"}
			if boxed, ok := input.Get("target").(*Object); ok {
				if boxed.Get("absent") == true {
					targets = nil
				} else {
					targets = []string{boxed.Get("value").(string)}
				}
			}
			var result any
			switch input.Get("operation") {
			case "check":
				target := "p"
				if len(targets) > 0 {
					target = targets[0]
				}
				result = authResolveCapture(models.CheckAuth(ctx, target))
			case "chat":
				value, err := models.GetAvailable(ctx, targets...)
				result = authResolveCapture(value, err)
			case "all":
				value, err := models.GetAllAvailable(ctx, targets...)
				result = authResolveCapture(value, err)
			case "image":
				value, err := models.GetAvailableOfType(ctx, "image", targets...)
				result = authResolveCapture(value, err)
			}
			catalogCompare(t, result, tc.Result)
			catalogCompare(t, log, tc.Log)
		})
	}
}
