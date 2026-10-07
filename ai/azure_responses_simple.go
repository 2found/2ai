package ai

import (
	"context"
	"encoding/json"
	"fmt"
)

// BuildAzureResponsesSimpleOptions retains only Pi's shared simple controls;
// Azure-specific settings must come through the scoped/process environment.
func BuildAzureResponsesSimpleOptions(model json.RawMessage, transcript TranscriptContext, options json.RawMessage) (json.RawMessage, error) {
	return buildOpenAISimpleOptions(model, transcript, options, false)
}

// StreamAzureResponsesSimple rejects missing API keys synchronously. Unlike
// the OpenAI providers, an authorization header cannot satisfy this admission.
func StreamAzureResponsesSimple(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options AzureResponsesStreamOptions) (*AssistantMessageEventStream, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	controls := map[string]json.RawMessage{}
	if len(options.Options) > 0 {
		if err := json.Unmarshal(options.Options, &controls); err != nil {
			return nil, err
		}
	}
	if samplingString(controls["apiKey"]) == "" {
		return nil, fmt.Errorf("No API key for provider: %s", model.Provider)
	}
	prepared, err := BuildAzureResponsesSimpleOptions(rawModel, transcript, options.Options)
	if err != nil {
		return nil, err
	}
	options.Options = prepared
	return StreamAzureResponses(ctx, rawModel, transcript, options), nil
}
