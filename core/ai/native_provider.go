package ai

import (
	"context"
	"encoding/json"
	"fmt"
)

// ValidateNativeModel rejects APIs without a native Go transport.
func ValidateNativeModel(raw json.RawMessage) error {
	var model struct{ API string }
	if err := json.Unmarshal(raw, &model); err != nil {
		return err
	}
	switch model.API {
	case "openai-completions", "openai-responses", "anthropic-messages", "openai-codex-responses", "azure-openai-responses", "pi-messages", VendorGoogleAntigravity, "devin-agent":
		return nil
	}
	return fmt.Errorf("native Go provider for API %q is not ported", model.API)
}

// NativeProvider selects a native Go transport from a model's declared API.
// Tokens optionally binds account rotation for Codex, Antigravity or Claude Code.
// It implements the same StreamFn contract as FallbackProvider.
type NativeProvider struct{ Tokens TokenSource }

func (p NativeProvider) Stream(ctx context.Context, model json.RawMessage, transcript TranscriptContext, options map[string]any) (*AssistantMessageEventStream, error) {
	if err := ValidateNativeModel(model); err != nil {
		return nil, err
	}
	controls := map[string]any{}
	for key, value := range options {
		switch key {
		case "signal", "telemetryContext", "fetch", "onPayload", "onResponse", "onProviderStreamEvent":
			continue
		}
		controls[key] = value
	}
	raw, err := json.Marshal(controls)
	if err != nil {
		return nil, err
	}
	provider, err := BindNativeStreamOptions(raw, options, OpenAICompletionsStreamOptions{})
	if err != nil {
		return nil, err
	}
	var selected struct{ API, Provider string }
	_ = json.Unmarshal(model, &selected) // Validated before callback binding.
	if p.Tokens != nil {
		if selected.API == VendorGoogleAntigravity && selected.Provider == VendorGoogleAntigravity {
			return StreamAntigravityPooled(ctx, model, transcript, provider, p.Tokens)
		}
		if selected.API == "anthropic-messages" && selected.Provider == VendorClaudeCode {
			return StreamClaudeCodePooled(ctx, model, transcript, provider, p.Tokens)
		}
		if selected.API == "devin-agent" && selected.Provider == VendorDevin {
			return StreamDevinPooled(ctx, model, transcript, provider, p.Tokens)
		}
		if !IsOAuthVendor(selected.Provider) || selected.Provider == VendorXaiOAuth {
			return nativeOAuthPoolStream(ctx, model, transcript, provider, p.Tokens, selected.Provider, func(ctx context.Context, model json.RawMessage, transcript TranscriptContext, opts OpenAICompletionsStreamOptions, _ OAuthToken) (*AssistantMessageEventStream, error) {
				if selected.API == "anthropic-messages" {
					return StreamAnthropicSimple(ctx, model, transcript, opts)
				}
				if selected.API == "openai-responses" {
					return StreamOpenAIResponsesSimple(ctx, model, transcript, opts)
				}
				return StreamOpenAICompletionsSimple(ctx, model, transcript, opts)
			})
		}
		if selected.API != "openai-codex-responses" {
			return nil, fmt.Errorf("native Codex account pool cannot serve API %q", selected.API)
		}
		return StreamCodexResponsesPooled(ctx, model, transcript, provider, p.Tokens)
	}
	if selected.API == VendorGoogleAntigravity {
		return nil, fmt.Errorf("Antigravity requires an account pool")
	}
	if selected.API == "devin-agent" {
		return nil, fmt.Errorf("Devin requires an account pool")
	}
	if selected.API == VendorPiMessages {
		return StreamPiMessagesJSON(ctx, model, transcript, provider)
	}
	if selected.API == "openai-codex-responses" {
		return StreamCodexResponsesSimple(ctx, model, transcript, provider)
	}
	if selected.API == "azure-openai-responses" {
		return StreamAzureResponsesSimple(ctx, model, transcript, provider)
	}
	if selected.API == "openai-responses" {
		return StreamOpenAIResponsesSimple(ctx, model, transcript, provider)
	}
	if selected.API == "anthropic-messages" {
		return StreamAnthropicSimple(ctx, model, transcript, provider)
	}
	return StreamOpenAICompletionsSimple(ctx, model, transcript, provider)
}
