package ai

import (
	"context"
	"encoding/json"
	"fmt"
)

// BuildCodexResponsesSimpleOptions uses the source's shared base-option and
// thinking-level projection. Provider-only effort/summary/tier options do not
// bypass the simple-mode reasoning contract.
func BuildCodexResponsesSimpleOptions(rawModel json.RawMessage, transcript TranscriptContext, rawOptions json.RawMessage) (json.RawMessage, error) {
	return buildOpenAISimpleOptions(rawModel, transcript, rawOptions, false)
}

// StreamCodexResponsesSimpleSSE preserves streamSimple's synchronous API-key
// check while executing the explicit SSE transport. It never obtains a token
// from environment fallback or a different account's cached credential.
func StreamCodexResponsesSimpleSSE(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions) (*AssistantMessageEventStream, error) {
	return streamCodexResponsesSimple(ctx, rawModel, transcript, options, true)
}

// StreamCodexResponsesSimple applies the simple-mode contract with native transport selection.
func StreamCodexResponsesSimple(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions) (*AssistantMessageEventStream, error) {
	return streamCodexResponsesSimple(ctx, rawModel, transcript, options, false)
}
func streamCodexResponsesSimple(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions, sseOnly bool) (*AssistantMessageEventStream, error) {
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
	if !samplingTruthy(controls["apiKey"]) {
		return nil, fmt.Errorf("No API key for provider: %s", model.Provider)
	}
	prepared, err := BuildCodexResponsesSimpleOptions(rawModel, transcript, options.Options)
	if err != nil {
		return nil, err
	}
	options.Options = prepared
	return streamCodexResponses(ctx, rawModel, transcript, options, sseOnly), nil
}
