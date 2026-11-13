package ai

import (
	"context"
	"encoding/json"
	"errors"
)

// StreamClaudeCodePooled uses the native Anthropic transcript and SSE machinery.
// It never calls the legacy Chat/Stream adapter or serializes the selected token
// into the model, transcript or returned events.
func StreamClaudeCodePooled(ctx context.Context, model json.RawMessage, transcript TranscriptContext, options AnthropicStreamOptions, source TokenSource) (*AssistantMessageEventStream, error) {
	var identity struct{ API, Provider string }
	if err := json.Unmarshal(model, &identity); err != nil {
		return nil, err
	}
	if identity.API != "anthropic-messages" || identity.Provider != VendorClaudeCode {
		return nil, errors.New("Claude Code pool requires its bound anthropic-messages model")
	}
	return nativeOAuthPoolStream(ctx, model, transcript, options, source, "Claude Code", func(ctx context.Context, model json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions, _ OAuthToken) (*AssistantMessageEventStream, error) {
		return StreamAnthropicSimple(ctx, model, transcript, options)
	})
}
