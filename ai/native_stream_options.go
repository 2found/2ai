package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// BindNativeStreamOptions binds the host's concrete callbacks and HTTP client
// separately from serializable controls. Nil callback entries preserve defaults.
// Provider-specific validation and request cancellation belong to the stream.
func BindNativeStreamOptions(raw json.RawMessage, options map[string]any, defaults OpenAICompletionsStreamOptions) (OpenAICompletionsStreamOptions, error) {
	provider := defaults
	provider.Options = append(json.RawMessage(nil), raw...)
	if client, exists := options["fetch"]; exists && client != nil {
		var ok bool
		provider.Client, ok = client.(*http.Client)
		if !ok {
			return provider, fmt.Errorf("native fetch must be an *http.Client")
		}
	}
	switch callback := options["onPayload"].(type) {
	case nil:
	case func(json.RawMessage, json.RawMessage) (json.RawMessage, error):
		provider.OnPayload = func(_ context.Context, payload, model json.RawMessage) (json.RawMessage, error) {
			return callback(payload, model)
		}
	case func(context.Context, json.RawMessage, json.RawMessage) (json.RawMessage, error):
		provider.OnPayload = callback
	default:
		return provider, fmt.Errorf("unsupported native onPayload callback %T", callback)
	}
	switch callback := options["onResponse"].(type) {
	case nil:
	case func(CompletionsResponse, json.RawMessage) error:
		provider.OnResponse = func(_ context.Context, response CompletionsResponse, model json.RawMessage) error {
			return callback(response, model)
		}
	case func(context.Context, CompletionsResponse, json.RawMessage) error:
		provider.OnResponse = callback
	default:
		return provider, fmt.Errorf("unsupported native onResponse callback %T", callback)
	}
	switch callback := options["onProviderStreamEvent"].(type) {
	case nil:
	case func(*json.RawMessage, json.RawMessage) error:
		provider.OnProviderStreamEvent = func(_ context.Context, event *json.RawMessage, model json.RawMessage) error {
			return callback(event, model)
		}
	case func(context.Context, *json.RawMessage, json.RawMessage) error:
		provider.OnProviderStreamEvent = callback
	default:
		return provider, fmt.Errorf("unsupported native onProviderStreamEvent callback %T", callback)
	}
	return provider, nil
}
