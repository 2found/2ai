package ai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

func TestPiEnvAPIKeyAuth(t *testing.T) {
	for index, tc := range readCredentialFixture(t).AuthCases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			calls := NewArray()
			env := catalogDecode(t, tc.Env).(*Object)
			auth := EnvAPIKeyAuth("Example API key", NewArray("PRIMARY", "SECONDARY"))
			result, err := auth.Resolve(APIKeyAuthInput{Context: context.Background(), Credential: credentialUnbox(t, tc.Credential), AuthContext: &AuthContext{Env: func(_ context.Context, name string) (any, error) {
				calls.Append(name)
				value, exists := env.Lookup(name)
				if !exists {
					return Undefined, nil
				}
				return value, nil
			}}})
			catalogCompare(t, calls, tc.Calls)
			catalogCompare(t, registryCapture(result, err), tc.Result)
			if auth.Name != "Example API key" || auth.Login == nil || auth.Check != nil {
				t.Fatal("env auth declaration differs")
			}
		})
	}
}
func TestPiEnvAPIKeyLogin(t *testing.T) {
	for index, tc := range readCredentialFixture(t).LoginCases {
		t.Run(fmt.Sprint(index)+"/"+tc.Mode, func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			cause := errors.New("cancelled by caller")
			if tc.Mode == "preabort" {
				cancel(cause)
			}
			prompts := NewArray()
			auth := EnvAPIKeyAuth("Example API key", NewArray())
			result, err := auth.Login(ProviderAuthInteraction{Context: ctx, Prompt: func(promptCtx context.Context, prompt *Object) (any, error) {
				if promptCtx != ctx {
					t.Error("prompt context changed")
				}
				prompts.Append(prompt)
				if tc.Mode == "prompt-abort" {
					cancel(cause)
				}
				if tc.Mode == "prompt-error" {
					return nil, errors.New("prompt failed")
				}
				return credentialUnbox(t, tc.Key), nil
			}})
			catalogCompare(t, prompts, tc.Prompts)
			catalogCompare(t, registryCapture(result, err), tc.Result)
			if tc.Mode == "preabort" || tc.Mode == "prompt-abort" {
				if !errors.Is(err, cause) {
					t.Fatal("login changed cancellation cause")
				}
			}
		})
	}
}
func TestPiEnvAPIKeyFailureAndLiveVariableList(t *testing.T) {
	for _, tc := range readCredentialFixture(t).EnvFailures {
		t.Run(tc.Mode, func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			cause := errors.New("cancelled by caller")
			if tc.Mode == "preabort" {
				cancel(cause)
			}
			names := NewArray("PRIMARY")
			calls := NewArray()
			failure := errors.New("env failed")
			result, err := EnvAPIKeyAuth("Example API key", names).Resolve(APIKeyAuthInput{Context: ctx, AuthContext: &AuthContext{Env: func(_ context.Context, name string) (any, error) {
				calls.Append(name)
				if tc.Mode == "append" {
					if name == "PRIMARY" {
						names.Append("ADDED")
						return "", nil
					}
					return "added-key", nil
				}
				if strings.Contains(tc.Mode, "abort") {
					cancel(cause)
				}
				if strings.Contains(tc.Mode, "error") {
					return nil, failure
				}
				return "key", nil
			}}})
			catalogCompare(t, calls, tc.Calls)
			catalogCompare(t, registryCapture(result, err), tc.Result)
			if strings.Contains(tc.Mode, "error") && !errors.Is(err, failure) {
				t.Fatal("environment rejection identity was replaced")
			}
		})
	}
}
func TestEnvAPIKeyStoredMetadataIdentity(t *testing.T) {
	scoped := NewObject(Property{Name: "ZONE", Value: "fixture"})
	credential := NewObject(Property{Name: "key", Value: "fixture"}, Property{Name: "env", Value: scoped})
	auth := EnvAPIKeyAuth("Key", NewArray("ENV"))
	resolved, err := auth.Resolve(APIKeyAuthInput{Context: context.Background(), Credential: credential, AuthContext: &AuthContext{Env: func(context.Context, string) (any, error) {
		t.Error("stored key should skip environment")
		return Undefined, nil
	}}})
	if err != nil || resolved.(*Object).Get("env") != scoped {
		t.Fatal("auth copied scoped metadata")
	}
	credential.Delete("env")
	resolved, err = auth.Resolve(APIKeyAuthInput{Context: context.Background(), Credential: credential})
	if err != nil {
		t.Fatal(err)
	}
	value, present := resolved.(*Object).Lookup("env")
	if !present || !jsonjs.IsUndefined(value) {
		t.Fatal("missing env must retain an own undefined property")
	}
}
func TestModelsCredentialStorageOwnership(t *testing.T) {
	ctx := context.Background()
	first, second := NewModels(), NewModels()
	credential := NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "fixture"})
	if _, err := first.credentials.Modify(ctx, "p", func(any) (any, error) { return credential, nil }); err != nil {
		t.Fatal(err)
	}
	value, err := second.credentials.Read(ctx, "p")
	if err != nil || !jsonjs.IsUndefined(value) {
		t.Fatal("default credential stores are shared")
	}
	first.DeleteProvider("p")
	first.ClearProviders()
	value, err = first.credentials.Read(ctx, "p")
	if err != nil || value != credential {
		t.Fatal("provider removal deleted credentials")
	}
	store := NewInMemoryCredentialStore()
	provided := &CredentialPersistence{Read: store.Read, List: store.List, Modify: store.Modify, Delete: store.Delete}
	first = NewModels(ModelsOptions{Credentials: provided})
	second = NewModels(ModelsOptions{Credentials: provided})
	if _, err = first.credentials.Modify(ctx, "p", func(any) (any, error) { return credential, nil }); err != nil {
		t.Fatal(err)
	}
	value, err = second.credentials.Read(ctx, "p")
	if err != nil || value != credential {
		t.Fatal("injected store identity changed")
	}
	replacement := NewObject(Property{Name: "type", Value: "oauth"})
	provided.Read = func(context.Context, string) (any, error) { return replacement, nil }
	value, err = first.credentials.Read(ctx, "p")
	if err != nil || value != replacement {
		t.Fatal("injected callback replacement ignored")
	}
}
