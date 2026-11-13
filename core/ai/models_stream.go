package ai

import (
	"context"
	"errors"
)

func (m *Models) Stream(ctx context.Context, model any, input Context, options ...*ModelsRequestOptions) *AssistantMessageEventStream {
	return m.stream(ctx, model, NormalizeContext(input), nil, "stream", modelRequestSettings(options))
}

func (m *Models) StreamSimple(ctx context.Context, model any, input Context, options ...*ModelsRequestOptions) *AssistantMessageEventStream {
	return m.stream(ctx, model, NormalizeContext(input), nil, "streamSimple", modelRequestSettings(options))
}

func (m *Models) StreamDeferred(ctx context.Context, model, handle any, options ...*ModelsRequestOptions) *AssistantMessageEventStream {
	return m.stream(ctx, model, TranscriptContext{}, handle, "fetchDeferred", modelRequestSettings(options))
}

func (m *Models) stream(ctx context.Context, model any, transcript TranscriptContext, handle any, method string, settings *ModelsRequestOptions) *AssistantMessageEventStream {
	var provider *ModelProvider
	var prepared *preparedModelAuth
	_, admission := invokeAuth(func() (any, error) {
		if err := AssertChatModel(model); err != nil {
			return nil, err
		}
		var err error
		provider, err = m.requireProvider(model)
		if err != nil {
			return nil, err
		}
		if method == "fetchDeferred" && provider.FetchDeferred == nil {
			id, err := catalogKey(catalogProperty(model, "provider"), make(map[*Array]bool))
			if err != nil {
				return nil, err
			}
			return nil, NewModelsError("provider", "Provider "+id+" does not support deferred responses", nil)
		}
		prepared, err = m.prepareModelAuth(model, settings)
		return nil, err
	})
	return LazyStream(ctx, model, func(streamCtx context.Context) (*ProviderEventSource, error) {
		if admission != nil {
			return nil, admission
		}
		requestModel, requestOptions, err := m.applyPreparedAuth(ctx, model, settings, prepared)
		if err != nil {
			return nil, err
		}
		if method == "fetchDeferred" {
			return provider.FetchDeferred(streamCtx, requestModel, handle, requestOptions)
		}
		callback := provider.Stream
		if method == "streamSimple" {
			callback = provider.StreamSimple
		}
		if callback == nil {
			return nil, errors.New("provider." + method + " is not a function. (In 'provider." + method + "(requestModel, transcript, requestOptions)', 'provider." + method + "' is undefined)")
		}
		return callback(streamCtx, requestModel, transcript, requestOptions)
	}, settings.Now)
}

// Completion waits for the stream result; request cancellation is handled by
// auth/provider callbacks and normally becomes an error message, not a Go wait
// error. Callers needing independent reader cancellation can use Stream.Result.
func (m *Models) Complete(ctx context.Context, model any, input Context, options ...*ModelsRequestOptions) (*Message, error) {
	return m.Stream(ctx, model, input, options...).Result(context.Background())
}

func (m *Models) CompleteSimple(ctx context.Context, model any, input Context, options ...*ModelsRequestOptions) (*Message, error) {
	return m.StreamSimple(ctx, model, input, options...).Result(context.Background())
}

func (m *Models) FetchDeferred(ctx context.Context, model, handle any, options ...*ModelsRequestOptions) (*Message, error) {
	return m.StreamDeferred(ctx, model, handle, options...).Result(context.Background())
}
