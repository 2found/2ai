package ai

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/2found/2ai/jsonjs"
)

// StreamPiMessagesJSON adapts the host's serialized model/options and callback
// boundary to pi-messages. It owns decoded input objects; callbacks can replace
// raw events (including primitive JSON values) before protocol conversion.
// Nil payload output preserves the payload, while JSON null replaces it.
func StreamPiMessagesJSON(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, input OpenAICompletionsStreamOptions) (*AssistantMessageEventStream, error) {
	value, err := jsonjs.DecodeValue(rawModel)
	if err != nil {
		return nil, err
	}
	model, ok := value.(*Object)
	if !ok {
		return nil, errors.New("pi-messages requires a model object")
	}
	values := NewObject()
	if len(input.Options) > 0 {
		value, err = jsonjs.DecodeValue(input.Options)
		if err != nil {
			return nil, err
		}
		if !jsonjs.IsNullish(value) {
			values, ok = value.(*Object)
			if !ok {
				return nil, errors.New("pi-messages options must be an object")
			}
		}
	}
	options := PiMessagesStreamOptions{Values: values, Client: input.Client}
	if input.Now != nil {
		options.Now = func() float64 { return float64(input.Now()) }
	}
	if input.OnPayload != nil {
		options.OnPayload = func(ctx context.Context, payload any, model *Object) (any, error) {
			encoded, err := jsonjs.MarshalValue(payload)
			if err != nil {
				return nil, err
			}
			raw, err := jsonjs.MarshalValue(model)
			if err != nil {
				return nil, err
			}
			next, err := input.OnPayload(ctx, encoded, raw)
			if err != nil || next == nil {
				return Undefined, err
			}
			return jsonjs.DecodeValue(next)
		}
	}
	if input.OnResponse != nil {
		options.OnResponse = func(ctx context.Context, response CompletionsResponse, model *Object) error {
			raw, err := jsonjs.MarshalValue(model)
			if err != nil {
				return err
			}
			return input.OnResponse(ctx, response, raw)
		}
	}
	if input.OnProviderStreamEvent != nil {
		options.transformEvent = func(ctx context.Context, event any, model *Object) (any, error) {
			encoded, err := jsonjs.MarshalValue(event)
			if err != nil {
				return nil, err
			}
			raw, err := jsonjs.MarshalValue(model)
			if err != nil {
				return nil, err
			}
			if err = input.OnProviderStreamEvent(ctx, (*json.RawMessage)(&encoded), raw); err != nil {
				return nil, err
			}
			return jsonjs.DecodeValue(encoded)
		}
	}
	return StreamSimplePiMessages(ctx, model, transcript, options), nil
}
