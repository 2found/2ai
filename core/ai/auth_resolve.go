package ai

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/2found/2ai/core/ai/jscompat"

	"github.com/2found/2ai/core/jsonjs"
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

// raceAuthOperation stops waiting on cancellation while continuing to observe
// the callback. Started OAuth refreshes still persist rotated credentials;
// other operations retain their own cancellation and storage rules.
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
		if jscompat.Truthy(overrides.Env) {
			requestContext = &AuthContext{
				Env: func(ctx context.Context, name string) (any, error) {
					if value := jscompat.ReadProperty(overrides.Env, name); jscompat.Truthy(value) {
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
		stored, err := jscompat.Invoke(func() (any, error) { return credentials.Read(ctx, readID) })
		if err != nil {
			return nil, NewModelsError("auth", "Credential store read failed for "+readID, err)
		}
		if jscompat.Truthy(stored) {
			switch jscompat.ReadProperty(stored, "type") {
			case "oauth":
				if provider.authConfig().OAuth != nil {
					return resolveStoredOAuth(ctx, provider, credentials, stored, overrides)
				}
			case "api_key":
				if provider.authConfig().APIKey != nil {
					if jscompat.Truthy(overrides.Env) {
						credential := jscompat.Spread(stored)
						env := jscompat.Spread(jscompat.ReadProperty(stored, "env"))
						jscompat.SpreadInto(env, overrides.Env)
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
	result, err := jscompat.Invoke(func() (any, error) {
		return apiKey.Resolve(APIKeyAuthInput{Context: ctx, AuthContext: authContext, Credential: credential})
	})
	if err != nil {
		return nil, NewModelsError("auth", "API key auth failed for provider "+providerID, err)
	}
	return result, nil
}

// refreshStoredOAuthCredential keeps cancellation active while waiting for the
// credential transaction. Once its callback starts, caller cancellation must
// not discard a refresh token the provider may already have rotated. The
// provider refresh has an independent timeout; Modify keeps its lock through
// persistence. Callers race the operation with cancellation to stop waiting.
func refreshStoredOAuthCredential(ctx context.Context, credentials *CredentialPersistence, providerID string, needsRefresh func(any) (bool, error), refresh func(context.Context, any) (any, error)) (any, error) {
	lockContext, cancelLock := context.WithCancelCause(context.WithoutCancel(ctx))
	stopWait := context.AfterFunc(ctx, func() { cancelLock(context.Cause(ctx)) })
	defer stopWait()
	defer cancelLock(nil)
	return jscompat.Invoke(func() (any, error) {
		return credentials.Modify(lockContext, providerID, func(current any) (any, error) {
			stopWait()
			if cause := context.Cause(ctx); cause != nil {
				return nil, cause
			}
			if !jscompat.StrictEqual(jscompat.ReadProperty(current, "type"), "oauth") {
				return Undefined, nil
			}
			needed, err := needsRefresh(current)
			if !needed || err != nil {
				return Undefined, err
			}
			refreshContext, cancel := context.WithTimeoutCause(context.WithoutCancel(ctx), OAuthRefreshTimeout, oauthRefreshTimeoutCause)
			// Retained provider contexts stay live until their timeout, even
			// when the callback returns or its original caller is cancelled.
			context.AfterFunc(refreshContext, cancel)
			return jscompat.Invoke(func() (any, error) { return refresh(refreshContext, current) })
		})
	})
}

func resolveStoredOAuth(ctx context.Context, provider *ModelProvider, credentials *CredentialPersistence, stored any, overrides AuthResolutionOverrides) (any, error) {
	providerID, oauth := provider.ID, provider.authConfig().OAuth
	minimum := float64(OAuthMinimumValidityMS)
	if !jsonjs.IsNullish(overrides.MinOAuthValidityMS) {
		value, err := jscompat.ToNumber(overrides.MinOAuthValidityMS)
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
		expires, err := jscompat.ToNumber(jscompat.ReadProperty(credential, "expires"))
		return float64(now())+minimum >= expires, err
	}
	credential := stored
	soon, err := expiresSoon(credential)
	if err != nil {
		return nil, err
	}
	if soon {
		post, err := refreshStoredOAuthCredential(ctx, credentials, providerID, expiresSoon, func(refreshContext context.Context, current any) (any, error) {
			refreshed, err := jscompat.Invoke(func() (any, error) { return oauth.Refresh(refreshContext, current) })
			if err != nil {
				return nil, NewModelsError("oauth", "OAuth refresh failed for "+providerID, err)
			}
			return refreshed, nil
		})
		if err != nil {
			if _, ok := err.(*ModelsError); ok {
				return nil, err
			}
			return nil, NewModelsError("auth", "Credential store modify failed for "+providerID, err)
		}
		if !jscompat.StrictEqual(jscompat.ReadProperty(post, "type"), "oauth") {
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
	if cause := context.Cause(ctx); cause != nil {
		return nil, cause
	}
	auth, err := jscompat.Invoke(func() (any, error) { return oauth.ToAuth(credential) })
	if err != nil {
		return nil, NewModelsError("oauth", "OAuth auth derivation failed for "+providerID, err)
	}
	return NewObject(Property{Name: "auth", Value: auth}, Property{Name: "source", Value: "OAuth"}), nil
}
