package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

type authResolveFixture struct {
	UpstreamCommit        string
	Concurrent, Cancelled json.RawMessage
	CallbackMutation      json.RawMessage
	Cases                 []struct{ Input, Result, Log, Timeouts, Stored json.RawMessage }
	ErrorCases            []struct {
		Message, Detail, Expected, Code string
		SameCause                       bool
	}
	EnvCases  []struct{ Input, Result json.RawMessage }
	FileCases []struct {
		Path   string
		Result bool
	}
}

func readAuthResolveFixture(t *testing.T) authResolveFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-auth-resolve.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture authResolveFixture
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 434 || len(fixture.ErrorCases) != 20 || len(fixture.EnvCases) != 7 || len(fixture.FileCases) != 6 {
		t.Fatal("unexpected auth resolve coverage")
	}
	return fixture
}
func authResolveCapture(value any, err error) any {
	if err == nil {
		return registryCapture(value, nil)
	}
	details := NewObject(Property{Name: "message", Value: err.Error()})
	if modelsError, ok := err.(*ModelsError); ok {
		details.Set("code", modelsError.Code)
	}
	return NewObject(Property{Name: "error", Value: details})
}
func TestPiProviderAuthResolution(t *testing.T) {
	for index, tc := range readAuthResolveFixture(t).Cases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			input := catalogDecode(t, tc.Input).(*Object)
			log := NewArray()
			timeouts := NewArray()
			memory := NewInMemoryCredentialStore()
			initial := input.Get("stored").(*Object)
			if initial.Get("absent") != true {
				if _, err := memory.Modify(context.Background(), "p", func(any) (any, error) { return initial.Get("value"), nil }); err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			if input.Get("abort") == true {
				cancel(errors.New("cancelled by caller"))
			}
			store := &CredentialPersistence{Read: func(ctx context.Context, id string) (any, error) {
				log.Append("read")
				if input.Get("failure") == "read" {
					return nil, errors.New("read failed")
				}
				return memory.Read(ctx, id)
			}, Modify: func(ctx context.Context, id string, modify func(any) (any, error)) (any, error) {
				log.Append("modify")
				if input.Get("failure") == "modify" {
					return nil, errors.New("modify failed")
				}
				switch input.Get("swap") {
				case "logout":
					if err := memory.Delete(context.Background(), id); err != nil {
						return nil, err
					}
				case "fresh":
					_, err := memory.Modify(context.Background(), id, func(any) (any, error) {
						return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "other"}, Property{Name: "expires", Value: 900000}), nil
					})
					if err != nil {
						return nil, err
					}
				case "apikey":
					_, err := memory.Modify(context.Background(), id, func(any) (any, error) {
						return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "other"}), nil
					})
					if err != nil {
						return nil, err
					}
				}
				return memory.Modify(ctx, id, modify)
			}}
			authContext := &AuthContext{Env: func(_ context.Context, name string) (any, error) {
				log.Append(NewObject(Property{Name: "env", Value: name}))
				switch name {
				case "KEY":
					return "ambient", nil
				case "REGION":
					return "ambient-region", nil
				}
				return Undefined, nil
			}, FileExists: func(_ context.Context, path string) (bool, error) {
				log.Append(NewObject(Property{Name: "file", Value: path}))
				return true, nil
			}}
			provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{}, Name: "Provider", GetModels: func() (*Array, error) { return NewArray(), nil }}
			if handlers := input.Get("handlers"); handlers == "api" || handlers == "both" {
				helper := EnvAPIKeyAuth("Fixture key", NewArray("KEY"))
				provider.Auth.APIKey = &APIKeyAuth{Resolve: func(args APIKeyAuthInput) (any, error) {
					log.Append(NewObject(Property{Name: "api", Value: registryCapture(args.Credential, nil)}))
					if input.Get("failure") == "api" {
						return nil, errors.New("api failed")
					}
					region, err := args.AuthContext.Env(args.Context, "REGION")
					if err != nil {
						return nil, err
					}
					file, err := args.AuthContext.FileExists(args.Context, "/fixture/file")
					if err != nil {
						return nil, err
					}
					log.Append(NewObject(Property{Name: "region", Value: region}, Property{Name: "file", Value: file}))
					result, err := helper.Resolve(args)
					if err == nil && catalogEntryTruthy(result) && catalogEntryTruthy(input.Get("authHeaders")) {
						result.(*Object).Get("auth").(*Object).Set("headers", input.Get("authHeaders"))
					}
					if replacement, exists := input.Lookup("authResult"); exists {
						return replacement, err
					}
					return result, err
				}}
			}
			if handlers := input.Get("handlers"); handlers == "oauth" || handlers == "both" {
				provider.Auth.OAuth = &OAuthAuth{Refresh: func(refreshCtx context.Context, credential any) (any, error) {
					deadline, present := refreshCtx.Deadline()
					if !present || time.Until(deadline) > OAuthRefreshTimeout || time.Until(deadline) < OAuthRefreshTimeout-time.Second {
						t.Error("missing 15-second refresh deadline")
					}
					timeouts.Append(15000)
					log.Append(NewObject(Property{Name: "refresh", Value: credential}, Property{Name: "aborted", Value: refreshCtx.Err() != nil}))
					if input.Get("failure") == "refresh" {
						return nil, errors.New("refresh failed")
					}
					expires, exists := input.Lookup("refreshExpires")
					if !exists {
						expires = 900000
					}
					return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "rotated"}, Property{Name: "refresh", Value: "new-r"}, Property{Name: "expires", Value: expires}), nil
				}, ToAuth: func(credential any) (any, error) {
					log.Append(NewObject(Property{Name: "toAuth", Value: credential}))
					if input.Get("failure") == "toAuth" {
						return nil, errors.New("derive failed")
					}
					return NewObject(Property{Name: "apiKey", Value: catalogProperty(credential, "access")}, Property{Name: "baseUrl", Value: "https://example.invalid/oauth"}), nil
				}}
			}
			models := NewModels(ModelsOptions{Credentials: store, AuthContext: authContext})
			models.SetProvider(provider)
			rawOverrides := input.Get("overrides").(*Object)
			optional := func(name string) any {
				value, exists := rawOverrides.Lookup(name)
				if !exists {
					return Undefined
				}
				if value == nil {
					return Null
				}
				return value
			}
			overrides := AuthResolutionOverrides{APIKey: optional("apiKey"), Env: optional("env"), MinOAuthValidityMS: optional("minOAuthValidityMs"), Now: func() int64 { return 1000 }}
			target, exists := input.Lookup("target")
			if !exists {
				target, exists = input.Lookup("model")
				if !exists {
					target = "p"
				}
			}
			result, err := models.GetAuth(ctx, target, overrides)
			catalogCompare(t, authResolveCapture(result, err), tc.Result)
			catalogCompare(t, log, tc.Log)
			catalogCompare(t, timeouts, tc.Timeouts)
			stored, err := memory.Read(context.Background(), "p")
			if err != nil {
				t.Fatal(err)
			}
			catalogCompare(t, registryCapture(stored, nil), tc.Stored)
		})
	}
}
func TestPiModelsErrorCauseDetail(t *testing.T) {
	for i, tc := range readAuthResolveFixture(t).ErrorCases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			cause := errors.New(tc.Detail)
			err := NewModelsError("auth", tc.Message, cause)
			if err.Error() != tc.Expected || err.Code != tc.Code || errors.Is(err, cause) != tc.SameCause {
				t.Fatalf("Go %q (%s), Pi %q (%s)", err.Error(), err.Code, tc.Expected, tc.Code)
			}
		})
	}
}
func TestPiDefaultProviderAuthContext(t *testing.T) {
	fixture := readAuthResolveFixture(t)
	auth := DefaultProviderAuthContext()
	const name = "AGENTRAY_PI_AUTH_RESOLVE_FIXTURE"
	for index, tc := range fixture.EnvCases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			value := credentialUnbox(t, tc.Input)
			if jsonjs.IsUndefined(value) {
				t.Setenv(name, "")
				if err := os.Unsetenv(name); err != nil {
					t.Fatal(err)
				}
			} else {
				t.Setenv(name, value.(string))
			}
			result, err := auth.Env(context.Background(), name)
			if err != nil {
				t.Fatal(err)
			}
			catalogCompare(t, registryCapture(result, nil), tc.Result)
		})
	}
	for _, tc := range fixture.FileCases {
		result, err := auth.FileExists(context.Background(), tc.Path)
		if err != nil || result != tc.Result {
			t.Fatalf("FileExists(%q)=%t,%v; Pi %t", tc.Path, result, err, tc.Result)
		}
	}
}
