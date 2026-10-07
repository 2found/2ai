package ai

import (
	"context"
	"encoding/json"

	"github.com/2found/2ai/jsonjs"
)

// Native API factories compose the existing stream implementations with Pi's
// lazy provider boundary. Defaults carry concrete HTTP/clock/callback hooks;
// request controls and callbacks override them for that request.
func OpenAICompletionsAPI(defaults OpenAICompletionsStreamOptions) *ProviderStreams {
	return nativeJSONAPI(defaults, StreamOpenAICompletions, StreamOpenAICompletionsSimple)
}
func OpenAIResponsesAPI(defaults OpenAICompletionsStreamOptions) *ProviderStreams {
	return nativeJSONAPI(defaults, StreamOpenAIResponses, StreamOpenAIResponsesSimple)
}
func AnthropicMessagesAPI(defaults OpenAICompletionsStreamOptions) *ProviderStreams {
	return nativeJSONAPI(defaults, StreamAnthropic, StreamAnthropicSimple)
}
func AzureOpenAIResponsesAPI(defaults OpenAICompletionsStreamOptions) *ProviderStreams {
	return nativeJSONAPI(defaults, StreamAzureResponses, StreamAzureResponsesSimple)
}
func OpenAICodexResponsesAPI(defaults OpenAICompletionsStreamOptions) *ProviderStreams {
	return nativeJSONAPI(defaults, StreamCodexResponses, StreamCodexResponsesSimple)
}

func nativeJSONAPI(defaults OpenAICompletionsStreamOptions,
	stream func(context.Context, json.RawMessage, TranscriptContext, OpenAICompletionsStreamOptions) *AssistantMessageEventStream,
	simple func(context.Context, json.RawMessage, TranscriptContext, OpenAICompletionsStreamOptions) (*AssistantMessageEventStream, error),
) *ProviderStreams {
	call := func(useSimple bool) ModelStreamFunc {
		return func(ctx context.Context, model any, transcript TranscriptContext, values *Object) (*ProviderEventSource, error) {
			rawModel, err := jsonjs.MarshalValue(model)
			if err != nil {
				return nil, err
			}
			controls := NewObject()
			bindings := map[string]any{}
			for _, entry := range values.Entries() {
				switch entry.Name {
				case "signal", "telemetryContext":
					continue
				case "fetch", "onPayload", "onResponse", "onProviderStreamEvent":
					bindings[entry.Name] = entry.Value
				default:
					controls.Set(entry.Name, entry.Value)
				}
			}
			raw, err := jsonjs.MarshalValue(controls)
			if err != nil {
				return nil, err
			}
			options, err := BindNativeStreamOptions(raw, bindings, defaults)
			if err != nil {
				return nil, err
			}
			var result *AssistantMessageEventStream
			if useSimple {
				result, err = simple(ctx, rawModel, transcript, options)
			} else {
				result = stream(ctx, rawModel, transcript, options)
			}
			if err != nil {
				return nil, err
			}
			return SourceFromAssistantStream(result), nil
		}
	}
	implementation := &ProviderStreams{Stream: call(false), StreamSimple: call(true)}
	return LazyAPI(func(context.Context) (*ProviderStreams, error) { return implementation, nil }, LazyAPICapabilities{}, defaults.Now)
}
