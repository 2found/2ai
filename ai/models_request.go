package ai

import (
	"context"
	"time"

	"github.com/2found/2ai/jsonjs"
)

// Values retains provider-specific options and their field presence. The
// Models-only header callback runs after auth/model/request headers are merged.
// Now supplies the clock for failure results, never a provider request option.
type ModelsRequestOptions struct {
	Values           any
	TransformHeaders func(any) (any, error)
	Now              func() int64
}

func modelRequestSettings(options []*ModelsRequestOptions) *ModelsRequestOptions {
	if len(options) > 0 && options[0] != nil {
		return options[0]
	}
	return &ModelsRequestOptions{}
}

func modelProperty(model any, key string) (any, error) {
	if jsonjs.IsNullish(model) {
		return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(model), "model."+key)
	}
	return catalogProperty(model, key), nil
}

func (m *Models) requireProvider(model any) (*ModelProvider, error) {
	id, err := modelProperty(model, "provider")
	if err != nil {
		return nil, err
	}
	if key, ok := id.(string); ok {
		if provider := m.GetProvider(key); provider != nil {
			return provider, nil
		}
	}
	key, err := catalogKey(id, make(map[*Array]bool))
	if err != nil {
		return nil, err
	}
	return nil, NewModelsError("provider", "Unknown provider: "+key, nil)
}

func assertModelType(model any, kind, article string) error {
	matches, err := IsModelType(model, kind)
	if matches || err != nil {
		return err
	}
	provider, err := catalogKey(catalogProperty(model, "provider"), make(map[*Array]bool))
	if err != nil {
		return err
	}
	id, err := catalogKey(catalogProperty(model, "id"), make(map[*Array]bool))
	if err != nil {
		return err
	}
	return NewModelsError("provider", "Model "+provider+"/"+id+" is not "+article+" "+kind+" model", nil)
}

func AssertChatModel(model any) error       { return assertModelType(model, "chat", "a") }
func AssertImageModel(model any) error      { return assertModelType(model, "image", "an") }
func AssertClassifierModel(model any) error { return assertModelType(model, "classifier", "a") }

func (m *Models) applyAuth(ctx context.Context, model any, settings *ModelsRequestOptions) (any, *Object, error) {
	prepared, err := m.prepareModelAuth(model, settings)
	if err != nil {
		return nil, nil, err
	}
	return m.applyPreparedAuth(ctx, model, settings, prepared)
}

type preparedModelAuth struct {
	provider  *ModelProvider
	options   any
	overrides AuthResolutionOverrides
}

func (m *Models) prepareModelAuth(model any, settings *ModelsRequestOptions) (*preparedModelAuth, error) {
	provider, err := m.requireProvider(model)
	if err != nil {
		return nil, err
	}
	options := settings.Values
	apiKeyOverride, envOverride := catalogProperty(options, "apiKey"), catalogProperty(options, "env")
	// Decoded JSON null is nil, while nil in the native override struct means
	// omitted. Preserve the explicit property value across that boundary.
	if apiKeyOverride == nil {
		apiKeyOverride = Null
	}
	if envOverride == nil {
		envOverride = Null
	}
	return &preparedModelAuth{provider, options, AuthResolutionOverrides{APIKey: apiKeyOverride, Env: envOverride}}, nil
}

func (m *Models) applyPreparedAuth(ctx context.Context, model any, settings *ModelsRequestOptions, prepared *preparedModelAuth) (any, *Object, error) {
	options := prepared.options
	resolution, err := m.authForProvider(ctx, prepared.provider, model, false, prepared.overrides)
	if err != nil {
		return nil, nil, err
	}
	if !catalogEntryTruthy(resolution) {
		id, err := catalogKey(catalogProperty(model, "provider"), make(map[*Array]bool))
		if err != nil {
			return nil, nil, err
		}
		return nil, nil, NewModelsError("auth", "Provider is not configured: "+id, nil)
	}
	auth := catalogProperty(resolution, "auth")
	apiKey := catalogProperty(options, "apiKey")
	if jsonjs.IsNullish(apiKey) {
		if jsonjs.IsNullish(auth) {
			return nil, nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(auth), "auth.apiKey")
		}
		apiKey = catalogProperty(auth, "apiKey")
	}
	if jsonjs.IsNullish(auth) {
		return nil, nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(auth), "auth.headers")
	}
	headers := mergeAuthHeaders(catalogProperty(auth, "headers"), catalogProperty(options, "headers"))
	if settings.TransformHeaders != nil {
		if jsonjs.IsNullish(headers) {
			headers = NewObject()
		}
		headers, err = settings.TransformHeaders(headers)
		if err != nil {
			return nil, nil, err
		}
	}
	var env any = Undefined
	if catalogEntryTruthy(catalogProperty(resolution, "env")) || catalogEntryTruthy(catalogProperty(options, "env")) {
		env = authSpread(catalogProperty(resolution, "env"))
		authSpreadInto(env.(*Object), catalogProperty(options, "env"))
	}
	requestModel := model
	if baseURL := catalogProperty(auth, "baseUrl"); catalogEntryTruthy(baseURL) {
		copy := authSpread(model)
		copy.Set("baseUrl", baseURL)
		requestModel = copy
	}
	requestOptions := authSpread(options)
	requestOptions.Delete("transformHeaders")
	requestOptions.Set("apiKey", apiKey)
	requestOptions.Set("headers", headers)
	requestOptions.Set("env", env)
	return requestModel, requestOptions, nil
}

func (m *Models) CancelDeferred(ctx context.Context, model, handle any, options ...*ModelsRequestOptions) error {
	_, err := invokeAuth(func() (any, error) {
		if err := AssertChatModel(model); err != nil {
			return nil, err
		}
		provider, err := m.requireProvider(model)
		if err != nil {
			return nil, err
		}
		if provider.CancelDeferred == nil {
			id, err := catalogKey(catalogProperty(model, "provider"), make(map[*Array]bool))
			if err != nil {
				return nil, err
			}
			return nil, NewModelsError("provider", "Provider "+id+" does not support deferred responses", nil)
		}
		settings := modelRequestSettings(options)
		requestModel, requestOptions, err := m.applyAuth(ctx, model, settings)
		if err != nil {
			return nil, err
		}
		return Undefined, provider.CancelDeferred(ctx, requestModel, handle, requestOptions)
	})
	return err
}

func (m *Models) GenerateImages(ctx context.Context, model, input any, options ...*ModelsRequestOptions) (any, error) {
	return m.runModelOperation(ctx, model, input, false, options)
}

func (m *Models) Classify(ctx context.Context, model, input any, options ...*ModelsRequestOptions) (any, error) {
	return m.runModelOperation(ctx, model, input, true, options)
}

func (m *Models) runModelOperation(ctx context.Context, model, input any, classify bool, options []*ModelsRequestOptions) (any, error) {
	settings := modelRequestSettings(options)
	result, err := invokeAuth(func() (any, error) {
		assert := AssertImageModel
		if classify {
			assert = AssertClassifierModel
		}
		if err := assert(model); err != nil {
			return nil, err
		}
		provider, err := m.requireProvider(model)
		if err != nil {
			return nil, err
		}
		operation, name := provider.GenerateImages, "image generation"
		if classify {
			operation, name = provider.Classify, "classification"
		}
		if operation == nil {
			id, err := catalogKey(catalogProperty(model, "provider"), make(map[*Array]bool))
			if err != nil {
				return nil, err
			}
			return nil, NewModelsError("provider", "Provider "+id+" does not support "+name, nil)
		}
		requestModel, requestOptions, err := m.applyAuth(ctx, model, settings)
		if err != nil {
			return nil, err
		}
		// Resolve the method again after auth: callbacks may replace it while
		// auth is pending, while registry replacement retains this provider.
		if classify {
			return provider.Classify(ctx, requestModel, input, requestOptions)
		}
		return provider.GenerateImages(ctx, requestModel, input, requestOptions)
	})
	if err == nil {
		return result, nil
	}
	return modelOperationErrorResult(model, err, ctx.Err() != nil, classify, settings.Now)
}

func modelOperationErrorResult(model any, failure error, aborted, classify bool, now func() int64) (*Object, error) {
	api, err := modelProperty(model, "api")
	if err != nil {
		return nil, err
	}
	if now == nil {
		now = func() int64 { return time.Now().UnixMilli() }
	}
	result := NewObject(Property{Name: "api", Value: api}, Property{Name: "provider", Value: catalogProperty(model, "provider")}, Property{Name: "model", Value: catalogProperty(model, "id")})
	if classify {
		result.Set("answers", NewObject())
	} else {
		result.Set("output", NewArray())
	}
	stopReason := "error"
	if aborted {
		stopReason = "aborted"
	}
	result.Set("stopReason", stopReason)
	result.Set("errorMessage", failure.Error())
	result.Set("timestamp", now())
	return result, nil
}
