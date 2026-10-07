package ai

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/2found/2ai/jsonjs"
)

type RadiusProviderOptions struct{ ID, Name, Gateway *string }

// The pinned tree omits generated Radius model data. Supply that catalog
// explicitly; no current external catalog is silently substituted. Streams
// defaults to the native Go pi-messages implementation.
type RadiusProviderDependencies struct {
	BaselineModels *Object
	Streams        *ProviderStreams
	Client         *http.Client
	Now            func() float64
	LoadOAuth      func(*RadiusOAuthOptions) (*OAuthAuth, error)
}

func RadiusProvider(input RadiusProviderOptions, dependencies RadiusProviderDependencies) (*ModelProvider, error) {
	id, name, gateway := "radius", "Radius", DefaultRadiusGateway
	if input.ID != nil {
		id = *input.ID
	}
	if input.Name != nil {
		name = *input.Name
	}
	if input.Gateway != nil {
		gateway = *input.Gateway
	}
	gateway = NormalizeRadiusGatewayURL(gateway)
	streams := dependencies.Streams
	if streams == nil {
		streams = PiMessagesAPI(PiMessagesStreamOptions{Client: dependencies.Client, Now: dependencies.Now})
	}
	baseline := NewArray()
	if gateway == NormalizeRadiusGatewayURL(DefaultRadiusGateway) {
		if dependencies.BaselineModels == nil {
			return nil, errors.New("Radius provider requires the pinned baseline model catalog for the default gateway")
		}
		for _, entry := range dependencies.BaselineModels.Entries() {
			model := authSpread(entry.Value)
			model.Set("provider", id)
			baseline.Append(model)
		}
	}
	now := dependencies.Now
	if now == nil {
		now = func() float64 { return float64(time.Now().UnixMilli()) }
	}
	load := dependencies.LoadOAuth
	if load == nil {
		load = func(options *RadiusOAuthOptions) (*OAuthAuth, error) {
			return RadiusOAuth(options, RadiusOAuthRuntimeOptions{Client: dependencies.Client, Now: now}), nil
		}
	}
	var mu sync.RWMutex
	dynamic := NewArray()
	update := func(models *Array) func() error {
		return func() error {
			mu.Lock()
			dynamic = models
			mu.Unlock()
			return nil
		}
	}
	provider := &ModelProvider{ID: id, Name: name, Auth: &ProviderAuth{
		APIKey: EnvAPIKeyAuth("Radius API key", NewArray("RADIUS_API_KEY")),
		OAuth: LazyOAuth(&LazyOAuthOptions{Name: name, Load: func() (*OAuthAuth, error) {
			return load(&RadiusOAuthOptions{Name: name, Gateway: gateway})
		}}),
	}}
	provider.GetModels = func() (*Array, error) {
		mu.RLock()
		defer mu.RUnlock()
		merged := NewArray(baseline.Values()...)
		for _, model := range dynamic.Values() {
			index := -1
			for candidate, entry := range merged.Values() {
				if jsonjs.IsNullish(entry) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(entry), "entry.id")
				}
				if jsonjs.IsNullish(model) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(model), "model.id")
				}
				if catalogStrictEqual(catalogProperty(entry, "id"), catalogProperty(model, "id")) {
					index = candidate
					break
				}
			}
			if index < 0 {
				merged.Append(model)
			} else {
				merged.Set(index, model)
			}
		}
		return merged, nil
	}
	provider.RefreshModels = func(refresh ModelRefreshContext) error {
		_, err := invokeAuth(func() (any, error) {
			if refresh.Stored != nil {
				value := catalogProperty(refresh.Stored, "models")
				models, ok := value.(*Array)
				if !ok {
					if jsonjs.IsNullish(value) {
						return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(value), "stored.models.filter")
					}
					return nil, errors.New("stored.models.filter is not a function")
				}
				restored := NewArray()
				for _, index := range models.Keys() {
					model := models.Get(index)
					if jsonjs.IsNullish(model) {
						return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(model), "model.provider")
					}
					if catalogStrictEqual(catalogProperty(model, "provider"), id) {
						restored.Append(model)
					}
				}
				accepted, err := refresh.Publish(ModelsPublication{Update: update(restored)})
				if err != nil || !accepted {
					return nil, err
				}
			} else if catalogStrictEqual(catalogProperty(refresh.Credential, "type"), "oauth") {
				legacy := GetRadiusModels(id, refresh.Credential)
				if legacy.Len() > 0 {
					accepted, err := refresh.Publish(ModelsPublication{Persist: NewObject(Property{Name: "models", Value: legacy}, Property{Name: "checkedAt", Value: now()}), Update: update(legacy)})
					if err != nil || !accepted {
						return nil, err
					}
				}
			}
			ctx := refresh.Context
			if ctx == nil {
				ctx = context.Background()
			}
			if !refresh.AllowNetwork || ctx.Err() != nil {
				return nil, nil
			}
			key := catalogProperty(refresh.Credential, "key")
			if catalogStrictEqual(catalogProperty(refresh.Credential, "type"), "oauth") {
				key = catalogProperty(refresh.Credential, "access")
			}
			config, err := LoadRadiusGatewayConfig(ctx, gateway, key, dependencies.Client)
			if err != nil || ctx.Err() != nil {
				return nil, err
			}
			models, err := GetRadiusModelsFromConfig(id, config)
			if err != nil {
				return nil, err
			}
			_, err = refresh.Publish(ModelsPublication{Persist: NewObject(Property{Name: "models", Value: models}, Property{Name: "checkedAt", Value: now()}), Update: update(models)})
			return nil, err
		})
		return err
	}
	provider.Stream = func(ctx context.Context, model any, transcript TranscriptContext, options *Object) (*ProviderEventSource, error) {
		return streams.Stream(ctx, model, transcript, options)
	}
	provider.StreamSimple = func(ctx context.Context, model any, transcript TranscriptContext, options *Object) (*ProviderEventSource, error) {
		return streams.StreamSimple(ctx, model, transcript, options)
	}
	return provider, nil
}
