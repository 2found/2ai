package ai

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"time"

	"github.com/2found/2ai/jsonjs"
)

type OAuthLoginOptions struct{ GetDeviceID func() string }

type OAuthAuth struct {
	Name           string
	IsSubscription *bool
	LoginLabel     *string
	Login          func(ProviderAuthInteraction, *OAuthLoginOptions) (any, error)
	Refresh        func(context.Context, any) (any, error)
	ToAuth         func(any) (any, error)
}

type ProviderAuth struct {
	APIKey *APIKeyAuth
	OAuth  *OAuthAuth
}

// nil/Undefined mean omitted overrides; Null retains an explicit null. Now is
// a native clock dependency and is not forwarded to provider callbacks.
type AuthResolutionOverrides struct {
	APIKey             any
	Env                any
	MinOAuthValidityMS any
	Now                func() int64
}

const OAuthMinimumValidityMS = 5 * 60 * 1000
const OAuthRefreshTimeout = 15 * time.Second

var oauthRefreshTimeoutCause = errors.New("The operation timed out.")

func invokeAuth(callback func() (any, error)) (value any, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			var ok bool
			err, ok = recovered.(error)
			if !ok {
				err = fmt.Errorf("%v", recovered)
			}
		}
	}()
	return callback()
}

// raceAuthOperation stops waiting on cancellation while continuing to observe
// the callback. A canceled waiter does not grant permission to commit storage.
func raceAuthOperation(ctx context.Context, operation func() (any, error)) (any, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	type outcome struct {
		value any
		err   error
	}
	ready := make(chan outcome, 1)
	go func() {
		defer func() {
			if value := recover(); value != nil {
				err, ok := value.(error)
				if !ok {
					err = fmt.Errorf("%v", value)
				}
				ready <- outcome{err: err}
			}
		}()
		value, err := operation()
		if cause := context.Cause(ctx); cause != nil {
			err = cause
		}
		ready <- outcome{value, err}
	}()
	select {
	case result := <-ready:
		return result.value, result.err
	case <-ctx.Done():
		return nil, context.Cause(ctx)
	}
}

func ResolveProviderAuth(ctx context.Context, provider *ModelProvider, credentials *CredentialPersistence, authContext *AuthContext, overrides AuthResolutionOverrides) (any, error) {
	return raceAuthOperation(ctx, func() (any, error) {
		requestContext := authContext
		if catalogEntryTruthy(overrides.Env) {
			requestContext = &AuthContext{
				Env: func(ctx context.Context, name string) (any, error) {
					if value := catalogProperty(overrides.Env, name); catalogEntryTruthy(value) {
						return value, nil
					}
					return authContext.Env(ctx, name)
				},
				FileExists: func(ctx context.Context, path string) (bool, error) { return authContext.FileExists(ctx, path) },
			}
		}
		if overrides.APIKey != nil && !jsonjs.IsUndefined(overrides.APIKey) && provider.authConfig().APIKey != nil {
			env := overrides.Env
			if env == nil {
				env = Undefined
			}
			credential := NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: overrides.APIKey}, Property{Name: "env", Value: env})
			return resolveAPIKey(ctx, provider, requestContext, credential)
		}
		readID := provider.ID
		stored, err := invokeAuth(func() (any, error) { return credentials.Read(ctx, readID) })
		if err != nil {
			return nil, NewModelsError("auth", "Credential store read failed for "+readID, err)
		}
		if catalogEntryTruthy(stored) {
			switch catalogProperty(stored, "type") {
			case "oauth":
				if provider.authConfig().OAuth != nil {
					return resolveStoredOAuth(ctx, provider, credentials, stored, overrides)
				}
			case "api_key":
				if provider.authConfig().APIKey != nil {
					if catalogEntryTruthy(overrides.Env) {
						credential := authSpread(stored)
						env := authSpread(catalogProperty(stored, "env"))
						authSpreadInto(env, overrides.Env)
						credential.Set("env", env)
						stored = credential
					}
					return resolveAPIKey(ctx, provider, requestContext, stored)
				}
			}
			return Undefined, nil
		}
		if provider.authConfig().APIKey != nil {
			return resolveAPIKey(ctx, provider, requestContext, Undefined)
		}
		return Undefined, nil
	})
}

func resolveAPIKey(ctx context.Context, provider *ModelProvider, authContext *AuthContext, credential any) (any, error) {
	providerID, apiKey := provider.ID, provider.authConfig().APIKey
	result, err := invokeAuth(func() (any, error) {
		return apiKey.Resolve(APIKeyAuthInput{Context: ctx, AuthContext: authContext, Credential: credential})
	})
	if err != nil {
		return nil, NewModelsError("auth", "API key auth failed for provider "+providerID, err)
	}
	return result, nil
}

func resolveStoredOAuth(ctx context.Context, provider *ModelProvider, credentials *CredentialPersistence, stored any, overrides AuthResolutionOverrides) (any, error) {
	providerID, oauth := provider.ID, provider.authConfig().OAuth
	minimum := float64(OAuthMinimumValidityMS)
	if !jsonjs.IsNullish(overrides.MinOAuthValidityMS) {
		value, err := authNumber(overrides.MinOAuthValidityMS)
		if err != nil {
			return nil, err
		}
		minimum = math.Max(minimum, value)
	}
	now := overrides.Now
	if now == nil {
		now = func() int64 { return time.Now().UnixMilli() }
	}
	expiresSoon := func(credential any) (bool, error) {
		expires, err := authNumber(catalogProperty(credential, "expires"))
		return float64(now())+minimum >= expires, err
	}
	credential := stored
	soon, err := expiresSoon(credential)
	if err != nil {
		return nil, err
	}
	if soon {
		post, err := invokeAuth(func() (any, error) {
			return credentials.Modify(ctx, providerID, func(current any) (any, error) {
				if !catalogStrictEqual(catalogProperty(current, "type"), "oauth") {
					return Undefined, nil
				}
				soon, err := expiresSoon(current)
				if err != nil {
					return nil, err
				}
				if !soon {
					return Undefined, nil
				}
				refreshContext, cancel := context.WithTimeoutCause(ctx, OAuthRefreshTimeout, oauthRefreshTimeoutCause)
				// Pi's timeout signal remains live after refresh returns. Retained
				// callback signals must not become aborted merely because it finished.
				context.AfterFunc(refreshContext, cancel)
				refreshed, err := invokeAuth(func() (any, error) { return oauth.Refresh(refreshContext, current) })
				if err != nil {
					return nil, NewModelsError("oauth", "OAuth refresh failed for "+providerID, err)
				}
				return refreshed, nil
			})
		})
		if err != nil {
			if _, ok := err.(*ModelsError); ok {
				return nil, err
			}
			return nil, NewModelsError("auth", "Credential store modify failed for "+providerID, err)
		}
		if !catalogStrictEqual(catalogProperty(post, "type"), "oauth") {
			return Undefined, nil
		}
		credential = post
		if overrides.MinOAuthValidityMS != nil && !jsonjs.IsUndefined(overrides.MinOAuthValidityMS) {
			soon, err := expiresSoon(credential)
			if err != nil {
				return nil, err
			}
			if soon {
				return nil, NewModelsError("oauth", "OAuth refresh returned a token that expires too soon for "+providerID, nil)
			}
		}
	}
	auth, err := invokeAuth(func() (any, error) { return oauth.ToAuth(credential) })
	if err != nil {
		return nil, NewModelsError("oauth", "OAuth auth derivation failed for "+providerID, err)
	}
	return NewObject(Property{Name: "auth", Value: auth}, Property{Name: "source", Value: "OAuth"}), nil
}

func authNumber(value any) (float64, error) {
	if jsonjs.IsUndefined(value) {
		return math.NaN(), nil
	}
	if value == nil || jsonjs.IsNull(value) {
		return 0, nil
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Bool:
		if v.Bool() {
			return 1, nil
		}
		return 0, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(v.Int()), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return float64(v.Uint()), nil
	case reflect.Float32, reflect.Float64:
		return v.Float(), nil
	}
	text, err := catalogKey(value, make(map[*Array]bool))
	return ParseJSNumber(text), err
}

func authSpread(value any) *Object {
	object := NewObject()
	authSpreadInto(object, value)
	return object
}

func authSpreadInto(target *Object, value any) {
	switch value := value.(type) {
	case *Object:
		for _, property := range value.Entries() {
			target.Set(property.Name, property.Value)
		}
	case *Array:
		for _, key := range value.PropertyKeys() {
			item, _ := value.GetProperty(key)
			target.Set(key, item)
		}
	case string:
		values, _ := catalogValues(value)
		for index, item := range values {
			target.Set(fmt.Sprint(index), item)
		}
	}
}
