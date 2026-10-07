package ai

import "context"

// AuthContext is the concrete environment boundary for provider-owned auth.
// Env returns Undefined when absent. The callback is read on every resolution.
type AuthContext struct {
	Env        func(context.Context, string) (any, error)
	FileExists func(context.Context, string) (bool, error)
}

type ProviderAuthInteraction struct {
	Context context.Context
	Prompt  func(context.Context, *Object) (any, error)
	Notify  func(*Object)
}

type APIKeyAuthInput struct {
	Context     context.Context
	AuthContext *AuthContext
	Credential  any
}

type APIKeyAuth struct {
	Name    string
	Login   func(ProviderAuthInteraction, ...*OAuthLoginOptions) (any, error)
	Check   func(APIKeyAuthInput) (any, error)
	Resolve func(APIKeyAuthInput) (any, error)
}

// EnvAPIKeyAuth retains Pi's stored-key-first resolution and interactive login.
// envVars is a live array of string variable names. Environment values and
// credential fields remain unmodified, including whitespace and scoped config.
func EnvAPIKeyAuth(name string, envVars *Array) *APIKeyAuth {
	return &APIKeyAuth{
		Name: name,
		Login: func(interaction ProviderAuthInteraction, _ ...*OAuthLoginOptions) (any, error) {
			if err := context.Cause(interaction.Context); err != nil {
				return nil, err
			}
			key, err := interaction.Prompt(interaction.Context, NewObject(Property{Name: "type", Value: "secret"}, Property{Name: "message", Value: "Enter " + name}))
			if err != nil {
				return nil, err
			}
			if err := context.Cause(interaction.Context); err != nil {
				return nil, err
			}
			return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: key}), nil
		},
		Resolve: func(input APIKeyAuthInput) (any, error) {
			if err := context.Cause(input.Context); err != nil {
				return nil, err
			}
			if key := catalogProperty(input.Credential, "key"); catalogEntryTruthy(key) {
				return NewObject(
					Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: key})},
					Property{Name: "env", Value: catalogProperty(input.Credential, "env")},
					Property{Name: "source", Value: "stored credential"},
				), nil
			}
			for index := 0; index < envVars.Len(); index++ {
				name := envVars.Get(index).(string)
				value, err := input.AuthContext.Env(input.Context, name)
				if err != nil {
					return nil, err
				}
				if err := context.Cause(input.Context); err != nil {
					return nil, err
				}
				if catalogEntryTruthy(value) {
					return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: value})}, Property{Name: "source", Value: name}), nil
				}
			}
			return Undefined, nil
		},
	}
}
