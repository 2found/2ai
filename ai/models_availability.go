package ai

import (
	"context"
	"math"
	"reflect"

	"github.com/2found/2ai/jsonjs"
)

// CheckAuth checks configuration without refreshing stored OAuth credentials.
// API-key providers may supply a side-effect-free Check; otherwise resolution
// performs its own second credential read, as in Pi.
func (m *Models) CheckAuth(ctx context.Context, providerID string) (any, error) {
	return raceAuthOperation(ctx, func() (any, error) {
		provider := m.GetProvider(providerID)
		if provider == nil {
			return Undefined, nil
		}
		credential, err := m.readCredential(ctx, providerID)
		if err != nil {
			return nil, err
		}
		return m.checkProviderAuth(ctx, provider, credential)
	})
}

func (m *Models) checkProviderAuth(ctx context.Context, provider *ModelProvider, credential any) (any, error) {
	if catalogStrictEqual(catalogProperty(credential, "type"), "oauth") {
		if provider.authConfig().OAuth == nil {
			return Undefined, nil
		}
		return NewObject(Property{Name: "source", Value: "OAuth"}, Property{Name: "type", Value: "oauth"}), nil
	}
	apiKey := provider.authConfig().APIKey
	if apiKey == nil {
		return Undefined, nil
	}
	if apiKey.Check != nil {
		if !catalogStrictEqual(catalogProperty(credential, "type"), "api_key") {
			credential = Undefined
		}
		result, err := invokeAuth(func() (any, error) {
			return apiKey.Check(APIKeyAuthInput{Context: ctx, AuthContext: m.authContext, Credential: credential})
		})
		if err != nil {
			return nil, NewModelsError("auth", "API key auth check failed for provider "+provider.ID, err)
		}
		return result, nil
	}
	resolution, err := ResolveProviderAuth(ctx, provider, &m.credentials, m.authContext, AuthResolutionOverrides{})
	if err != nil || !catalogEntryTruthy(resolution) {
		return Undefined, err
	}
	return NewObject(Property{Name: "source", Value: catalogProperty(resolution, "source")}, Property{Name: "type", Value: "api_key"}), nil
}

type authenticatedProvider struct {
	provider   *ModelProvider
	credential any
	auth       any
}

func (m *Models) authenticatedProviders(ctx context.Context, providerID []string) ([]authenticatedProvider, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	providers := m.GetProviders()
	if len(providerID) > 0 && providerID[0] != "" {
		providers = nil
		if provider := m.GetProvider(providerID[0]); provider != nil {
			providers = append(providers, provider)
		}
	}
	type outcome struct {
		index int
		entry authenticatedProvider
		err   error
	}
	completed := make(chan outcome, len(providers))
	for index, provider := range providers {
		go func(index int, provider *ModelProvider) {
			entry := authenticatedProvider{provider: provider}
			_, err := invokeAuth(func() (any, error) {
				credential, err := m.readCredential(ctx, provider.ID)
				if err != nil {
					return nil, err
				}
				entry.credential = credential
				entry.auth, err = m.checkProviderAuth(ctx, provider, credential)
				return nil, err
			})
			completed <- outcome{index, entry, err}
		}(index, provider)
	}
	entries := make([]authenticatedProvider, len(providers))
	for range providers {
		result := <-completed
		if result.err != nil {
			return nil, result.err
		}
		entries[result.index] = result.entry
	}
	available := entries[:0]
	for _, entry := range entries {
		if !jsonjs.IsUndefined(entry.auth) {
			available = append(available, entry)
		}
	}
	return available, nil
}

// Availability reads propagate provider errors. Passing "" selects every
// provider, unlike the synchronous catalog accessors. Returned unions skip
// sparse holes and preserve model identity.
func (m *Models) GetAvailable(ctx context.Context, providerID ...string) (*Array, error) {
	return m.available(ctx, false, providerID)
}

func (m *Models) GetAllAvailable(ctx context.Context, providerID ...string) (*Array, error) {
	return m.available(ctx, true, providerID)
}

func (m *Models) available(ctx context.Context, all bool, providerID []string) (*Array, error) {
	value, err := raceAuthOperation(ctx, func() (any, error) {
		providers, err := m.authenticatedProviders(ctx, providerID)
		if err != nil {
			return nil, err
		}
		result := NewArray()
		for _, entry := range providers {
			models, err := availableProviderModels(entry, all)
			if err != nil {
				return nil, err
			}
			if models == nil {
				result.Append(Null)
				continue
			}
			for _, index := range models.Keys() {
				result.Append(models.Get(index))
			}
		}
		return result, nil
	})
	if err != nil {
		return nil, err
	}
	return value.(*Array), nil
}

func availableProviderModels(entry authenticatedProvider, all bool) (*Array, error) {
	provider := entry.provider
	var models *Array
	var err error
	if all && provider.GetAllModels != nil {
		models, err = provider.GetAllModels()
		if err != nil {
			return nil, err
		}
	}
	if models == nil {
		models, err = provider.GetModels()
		if err != nil {
			return nil, err
		}
	}
	if !all {
		if provider.FilterModels != nil {
			filtered, err := provider.FilterModels(models, entry.credential)
			if err != nil || filtered != nil {
				return filtered, err
			}
		}
		return models, nil
	}
	if provider.FilterAllModels != nil {
		return provider.FilterAllModels(models, entry.credential)
	}
	if provider.FilterModels == nil {
		return models, nil
	}
	chat, err := provider.GetModels()
	if err != nil {
		return nil, err
	}
	chat, err = provider.FilterModels(chat, entry.credential)
	if err != nil {
		return nil, err
	}
	if chat == nil {
		return nil, jsonjs.PropertyReadError(true, "provider.filterModels(provider.getModels(), credential).map")
	}
	ids := make([]any, chat.Len())
	for index := range ids {
		ids[index] = Undefined
		if !chat.Has(index) {
			continue
		}
		model := chat.Get(index)
		if jsonjs.IsNullish(model) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(model), "model.id")
		}
		ids[index] = catalogProperty(model, "id")
	}
	if models == nil {
		return nil, jsonjs.PropertyReadError(true, "models.filter")
	}
	result := NewArray()
	for _, index := range models.Keys() {
		model := models.Get(index)
		chat, err := IsModelType(model, "chat")
		if err != nil {
			return nil, err
		}
		keep := !chat
		if chat {
			for _, id := range ids {
				if catalogSameValueZero(id, catalogProperty(model, "id")) {
					keep = true
					break
				}
			}
		}
		if keep {
			result.Append(model)
		}
	}
	return result, nil
}

func (m *Models) GetAvailableOfType(ctx context.Context, kind any, providerID ...string) (*Array, error) {
	models, err := m.GetAllAvailable(ctx, providerID...)
	if err != nil {
		return nil, err
	}
	result := NewArray()
	for _, index := range models.Keys() {
		model := models.Get(index)
		matches, err := IsModelType(model, kind)
		if err != nil {
			return nil, err
		}
		if matches {
			result.Append(model)
		}
	}
	return result, nil
}

func catalogSameValueZero(a, b any) bool {
	if catalogStrictEqual(a, b) {
		return true
	}
	isNaN := func(value any) bool {
		if value == nil {
			return false
		}
		v := reflect.ValueOf(value)
		return (v.Kind() == reflect.Float64 || v.Kind() == reflect.Float32) && math.IsNaN(v.Float())
	}
	return isNaN(a) && isNaN(b)
}
