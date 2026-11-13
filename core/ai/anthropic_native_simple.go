package ai

import (
	"context"
	"encoding/json"
	"math"
)

// BuildAnthropicSimpleOptions projects Pi's simple controls, including adaptive
// effort and the two context clamps around budget-based thinking.
func BuildAnthropicSimpleOptions(rawModel json.RawMessage, transcript TranscriptContext, rawOptions json.RawMessage) (json.RawMessage, error) {
	prepared, _, err := buildAnthropicSimpleOptions(rawModel, transcript, rawOptions)
	return prepared, err
}

func buildAnthropicSimpleOptions(rawModel json.RawMessage, transcript TranscriptContext, rawOptions json.RawMessage) (json.RawMessage, *ThinkingTokenBudget, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, nil, err
	}
	options := map[string]json.RawMessage{}
	if len(rawOptions) > 0 {
		if err := json.Unmarshal(rawOptions, &options); err != nil {
			return nil, nil, err
		}
	}
	base := nativeBaseOptions(model, transcript, options)
	var numeric *ThinkingTokenBudget
	if !samplingTruthy(options["reasoning"]) {
		base["thinkingEnabled"] = json.RawMessage(`false`)
	} else {
		base["thinkingEnabled"] = json.RawMessage(`true`)
		level := samplingString(options["reasoning"])
		if string(model.Compat["forceAdaptiveThinking"]) == "true" {
			effort := "high"
			if level == "minimal" || level == "low" {
				effort = "low"
			} else if level == "medium" {
				effort = "medium"
			}
			if mapped := model.ThinkingLevelMap[level]; len(mapped) > 0 && mapped[0] == '"' {
				effort = samplingString(mapped)
			}
			base["effort"], _ = json.Marshal(effort)
		} else {
			var ceiling float64
			_ = json.Unmarshal(base["maxTokens"], &ceiling)
			custom, _ := samplingObject(options["thinkingBudgets"])
			adjusted := AdjustMaxTokensForThinking(&ceiling, model.MaxTokens, level, custom)
			adjusted.MaxTokens = ClampMaxTokensToContext(model.ContextWindow, transcript, adjusted.MaxTokens)
			adjusted.ThinkingBudget = math.Min(adjusted.ThinkingBudget, math.Max(0, adjusted.MaxTokens-1024))
			base["maxTokens"] = anthropicJSONNumber(adjusted.MaxTokens)
			base["thinkingBudgetTokens"] = anthropicJSONNumber(adjusted.ThinkingBudget)
			// JSON encodes NaN as null, but nullish defaults in buildParams must
			// not turn the original numeric NaN back into the model's token cap.
			if math.IsNaN(adjusted.MaxTokens) || math.IsNaN(adjusted.ThinkingBudget) {
				numeric = &adjusted
			}
		}
	}
	encoded, err := json.Marshal(base)
	return encoded, numeric, err
}

func anthropicJSONNumber(value float64) json.RawMessage {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return json.RawMessage(`null`)
	}
	encoded, _ := json.Marshal(value)
	return encoded
}

// StreamAnthropicSimple rejects missing credentials before admitting a stream.
// Federation is admitted when the provider's scoped/process env is complete.
func StreamAnthropicSimple(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options AnthropicStreamOptions) (*AssistantMessageEventStream, error) {
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
	if err := anthropicRequestAuth(model.Provider, controls); err != nil && anthropicFederation(model.Provider, controls) == nil {
		return nil, err
	}
	prepared, numeric, err := buildAnthropicSimpleOptions(rawModel, transcript, options.Options)
	if err != nil {
		return nil, err
	}
	options.Options = prepared
	return streamAnthropic(ctx, rawModel, transcript, options, numeric, nil), nil
}
