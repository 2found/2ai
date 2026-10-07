package ai

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/2found/2ai/jsonjs"
)

// ProviderEventSource is a concrete async-iteration boundary. Result is
// optional; its bool distinguishes undefined (no result) from explicit null.
// Iteration and result callbacks remain live until forwarding finishes.
type ProviderEventSource struct {
	Next   func(context.Context) (AssistantMessageEvent, bool, error)
	Result func(context.Context) (*Message, bool, error)
}

func SourceFromAssistantStream(stream *AssistantMessageEventStream) *ProviderEventSource {
	return &ProviderEventSource{Next: stream.Next, Result: func(ctx context.Context) (*Message, bool, error) {
		message, err := stream.Result(ctx)
		return message, true, err
	}}
}

type ModelStreamFunc func(context.Context, any, TranscriptContext, *Object) (*ProviderEventSource, error)
type DeferredStreamFunc func(context.Context, any, any, *Object) (*ProviderEventSource, error)

type ProviderStreams struct {
	Stream         ModelStreamFunc
	StreamSimple   ModelStreamFunc
	FetchDeferred  DeferredStreamFunc
	CancelDeferred func(context.Context, any, any, *Object) error
}

// LazyStream returns immediately. Only setup/provider callbacks own request
// cancellation; forwarding drains the source independently of reader waits.
// A terminal event settles Result before forwarding and source.Result finish.
func LazyStream(ctx context.Context, model any, setup func(context.Context) (*ProviderEventSource, error), now func() int64) *AssistantMessageEventStream {
	outer := NewAssistantMessageEventStreamFor(ctx)
	setupContext := WithAssistantStreamSynchronization(ctx, outer)
	go func() {
		_, err := invokeAuth(func() (any, error) {
			source, err := setup(setupContext)
			if err != nil {
				return nil, err
			}
			if source == nil {
				return nil, errors.New("undefined is not an object (evaluating 'source')")
			}
			for {
				event, present, err := source.Next(context.Background())
				if err != nil {
					return nil, err
				}
				if !present {
					break
				}
				outer.Push(event)
			}
			if source.Result != nil {
				result, present, err := source.Result(context.Background())
				if err != nil {
					return nil, err
				}
				if present {
					outer.End(result)
				} else {
					outer.End()
				}
			} else {
				outer.End()
			}
			return nil, nil
		})
		if err != nil {
			message, failure := lazySetupErrorMessage(model, err, now)
			if failure != nil {
				// Pi's failure-message construction can itself reject for a null
				// model, leaving the outer result unsettled. Preserve that boundary.
				return
			}
			outer.Push(AssistantMessageEvent{Type: "error", Reason: "error", Error: message})
			outer.End(message)
		}
	}()
	return outer
}

func lazySetupErrorMessage(model any, failure error, now func() int64) (*Message, error) {
	api, err := modelProperty(model, "api")
	if err != nil {
		return nil, err
	}
	if now == nil {
		now = func() int64 { return time.Now().UnixMilli() }
	}
	usage := NewObject(Property{Name: "input", Value: 0}, Property{Name: "output", Value: 0}, Property{Name: "cacheRead", Value: 0}, Property{Name: "cacheWrite", Value: 0}, Property{Name: "totalTokens", Value: 0}, Property{Name: "cost", Value: NewObject(Property{Name: "input", Value: 0}, Property{Name: "output", Value: 0}, Property{Name: "cacheRead", Value: 0}, Property{Name: "cacheWrite", Value: 0}, Property{Name: "total", Value: 0})})
	value := NewObject(Property{Name: "role", Value: "assistant"}, Property{Name: "content", Value: NewArray()}, Property{Name: "api", Value: api}, Property{Name: "provider", Value: catalogProperty(model, "provider")}, Property{Name: "model", Value: catalogProperty(model, "id")}, Property{Name: "usage", Value: usage}, Property{Name: "stopReason", Value: "error"}, Property{Name: "errorMessage", Value: failure.Error()}, Property{Name: "timestamp", Value: now()})
	raw, err := jsonjs.MarshalValue(value)
	if err != nil {
		return nil, err
	}
	var message Message
	if err := json.Unmarshal(raw, &message); err != nil {
		return nil, err
	}
	return &message, nil
}

type LazyAPICapabilities struct{ FetchDeferred, CancelDeferred bool }

// LazyAPI invokes load per request. Import/module caching belongs to the
// caller's loader, as in Pi; this adapter does not cache load failures.
func LazyAPI(load func(context.Context) (*ProviderStreams, error), capabilities LazyAPICapabilities, now func() int64) *ProviderStreams {
	stream := func(simple bool) ModelStreamFunc {
		return func(ctx context.Context, model any, transcript TranscriptContext, options *Object) (*ProviderEventSource, error) {
			return SourceFromAssistantStream(LazyStream(ctx, model, func(ctx context.Context) (*ProviderEventSource, error) {
				implementation, err := load(ctx)
				if err != nil {
					return nil, err
				}
				if simple {
					return implementation.StreamSimple(ctx, model, transcript, options)
				}
				return implementation.Stream(ctx, model, transcript, options)
			}, now)), nil
		}
	}
	api := &ProviderStreams{Stream: stream(false), StreamSimple: stream(true)}
	if capabilities.FetchDeferred {
		api.FetchDeferred = func(ctx context.Context, model, handle any, options *Object) (*ProviderEventSource, error) {
			return SourceFromAssistantStream(LazyStream(ctx, model, func(ctx context.Context) (*ProviderEventSource, error) {
				implementation, err := load(ctx)
				if err != nil {
					return nil, err
				}
				if implementation.FetchDeferred == nil {
					return nil, errors.New("API does not support deferred responses")
				}
				return implementation.FetchDeferred(ctx, model, handle, options)
			}, now)), nil
		}
	}
	if capabilities.CancelDeferred {
		api.CancelDeferred = func(ctx context.Context, model, handle any, options *Object) error {
			_, err := invokeAuth(func() (any, error) {
				implementation, err := load(ctx)
				if err != nil {
					return nil, err
				}
				if implementation.CancelDeferred == nil {
					return nil, errors.New("API cannot cancel deferred responses")
				}
				return Undefined, implementation.CancelDeferred(ctx, model, handle, options)
			})
			return err
		}
	}
	return api
}
