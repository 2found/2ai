package ai

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"slices"
)

const MinAnswerTokens = 1024

func ClampReasoning(effort string) string {
	if effort == "xhigh" || effort == "max" {
		return "high"
	}
	return effort
}

func ThinkingBudgetForLevel(level string, custom map[string]json.RawMessage) float64 {
	level = ClampReasoning(level)
	budget := math.NaN()
	switch level {
	case "minimal":
		budget = 1024
	case "low":
		budget = 2048
	case "medium":
		budget = 8192
	case "high":
		budget = 16384
	}
	if override, exists := custom[level]; exists {
		if !samplingNonNull(override) {
			budget = 0
		} else {
			_ = json.Unmarshal(override, &budget)
		}
	}
	return budget
}

func ClampThinkingBudgetToAnswerRoom(budget, ceiling float64) float64 {
	return math.Min(budget, math.Max(0, ceiling-MinAnswerTokens))
}

type ThinkingTokenBudget struct {
	MaxTokens      float64 `json:"maxTokens"`
	ThinkingBudget float64 `json:"thinkingBudget"`
}

func AdjustMaxTokensForThinking(base *float64, modelMax float64, level string, custom map[string]json.RawMessage) ThinkingTokenBudget {
	budget := ThinkingBudgetForLevel(level, custom)
	ceiling := modelMax
	if base != nil {
		ceiling = math.Min(*base+budget, modelMax)
	}
	if ceiling <= budget {
		budget = ClampThinkingBudgetToAnswerRoom(budget, ceiling)
	}
	return ThinkingTokenBudget{MaxTokens: ceiling, ThinkingBudget: budget}
}

var extendedThinkingLevels = []string{"off", "minimal", "low", "medium", "high", "xhigh", "max"}

func supportedThinkingLevels(model completionsModel) []string {
	if !model.Reasoning {
		return []string{"off"}
	}
	levels := []string{}
	for _, level := range extendedThinkingLevels {
		mapped, exists := model.ThinkingLevelMap[level]
		if exists && !samplingNonNull(mapped) {
			continue
		}
		if (level == "xhigh" || level == "max") && !exists {
			continue
		}
		levels = append(levels, level)
	}
	return levels
}

func clampThinkingLevel(model completionsModel, level string) string {
	available := supportedThinkingLevels(model)
	if slices.Contains(available, level) {
		return level
	}
	index := slices.Index(extendedThinkingLevels, level)
	if index >= 0 {
		for _, candidate := range extendedThinkingLevels[index:] {
			if slices.Contains(available, candidate) {
				return candidate
			}
		}
		for i := index - 1; i >= 0; i-- {
			if slices.Contains(available, extendedThinkingLevels[i]) {
				return extendedThinkingLevels[i]
			}
		}
	}
	if len(available) > 0 {
		return available[0]
	}
	return "off"
}

func GetSupportedThinkingLevels(rawModel json.RawMessage) ([]string, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	return supportedThinkingLevels(model), nil
}

func ClampThinkingLevel(rawModel json.RawMessage, level string) (string, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return "", err
	}
	return clampThinkingLevel(model, level), nil
}

func ClampMaxTokensToContext(contextWindow float64, transcript TranscriptContext, maxTokens float64) float64 {
	if contextWindow <= 0 {
		return math.Max(1, maxTokens)
	}
	available := contextWindow - EstimateContextTokens(transcript.Messages()).Tokens - 4096
	return math.Min(maxTokens, math.Max(1, available))
}

// BuildOpenAICompletionsSimpleOptions ports streamSimple's serializable option
// projection. Concrete Go callbacks, HTTP client and signal remain outside JSON.
func BuildOpenAICompletionsSimpleOptions(rawModel json.RawMessage, transcript TranscriptContext, rawOptions json.RawMessage) (json.RawMessage, error) {
	return buildOpenAISimpleOptions(rawModel, transcript, rawOptions, true)
}

// BuildOpenAIResponsesSimpleOptions follows Responses' streamSimple projection:
// thinkingBudgets, reasoningSummary and serviceTier are not forwarded.
func BuildOpenAIResponsesSimpleOptions(rawModel json.RawMessage, transcript TranscriptContext, rawOptions json.RawMessage) (json.RawMessage, error) {
	return buildOpenAISimpleOptions(rawModel, transcript, rawOptions, false)
}

func buildOpenAISimpleOptions(rawModel json.RawMessage, transcript TranscriptContext, rawOptions json.RawMessage, includeThinkingBudgets bool) (json.RawMessage, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	options := map[string]json.RawMessage{}
	if len(rawOptions) > 0 {
		if err := json.Unmarshal(rawOptions, &options); err != nil {
			return nil, err
		}
	}
	base := nativeBaseOptions(model, transcript, options)
	if includeThinkingBudgets {
		if value, exists := options["thinkingBudgets"]; exists {
			base["thinkingBudgets"] = value
		}
	}
	if level := samplingString(options["reasoning"]); level != "" {
		level = clampThinkingLevel(model, level)
		if level != "off" {
			base["reasoningEffort"], _ = json.Marshal(level)
		}
	}
	return json.Marshal(base)
}

// ResolveSamplingParams applies model defaults, effective-thinking-level defaults,
// then request overrides. A nil result means no sampling object was supplied.
func ResolveSamplingParams(rawModel json.RawMessage, thinkingLevel string, requestParams json.RawMessage) (json.RawMessage, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	if len(requestParams) > 0 && !json.Valid(requestParams) {
		return nil, errors.New("invalid sampling parameters JSON")
	}
	params := resolveSamplingParams(model, thinkingLevel, requestParams)
	if params == nil {
		return nil, nil
	}
	return json.Marshal(params)
}

func resolveSamplingParams(model completionsModel, thinkingLevel string, requestParams json.RawMessage) map[string]json.RawMessage {
	level := clampThinkingLevel(model, thinkingLevel)
	levelParams, _ := samplingObject(model.SamplingParamsByThinkingLevel[level])
	request, _ := samplingObject(requestParams)
	if model.SamplingParams == nil && levelParams == nil && request == nil {
		return nil
	}
	result := make(map[string]json.RawMessage)
	for _, params := range []map[string]json.RawMessage{model.SamplingParams, levelParams, request} {
		for key, value := range params {
			result[key] = value
		}
	}
	return result
}

func nativeBaseOptions(model completionsModel, transcript TranscriptContext, options map[string]json.RawMessage) map[string]json.RawMessage {
	base := map[string]json.RawMessage{}
	for _, name := range []string{"temperature", "signal", "telemetryContext", "apiKey", "fetch", "transport", "cacheRetention", "sessionId", "headers", "onPayload", "onResponse", "onProviderStreamEvent", "timeoutMs", "websocketConnectTimeoutMs", "maxRetries", "maxRetryDelayMs", "metadata", "env", "toolChoice"} {
		if value, exists := options[name]; exists {
			base[name] = value
		}
	}
	if sampling := resolveSamplingParams(model, samplingString(options["reasoning"]), options["samplingParams"]); sampling != nil {
		base["samplingParams"], _ = json.Marshal(sampling)
	}
	maxTokens := model.MaxTokens
	if samplingNonNull(options["maxTokens"]) {
		_ = json.Unmarshal(options["maxTokens"], &maxTokens)
	}
	base["maxTokens"], _ = json.Marshal(ClampMaxTokensToContext(model.ContextWindow, transcript, maxTokens))
	return base
}

// StreamOpenAICompletionsSimple retains the original synchronous credential
// check. Provider/request failures after admission settle the returned stream.
func StreamOpenAICompletionsSimple(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions) (*AssistantMessageEventStream, error) {
	prepared, err := prepareOpenAISimpleOptions(rawModel, transcript, options.Options, true)
	if err != nil {
		return nil, err
	}
	options.Options = prepared
	return StreamOpenAICompletions(ctx, rawModel, transcript, options), nil
}

// StreamOpenAIResponsesSimple validates credentials before admitting the stream,
// then runs the native provider with Pi's clamped simple options.
func StreamOpenAIResponsesSimple(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options OpenAIResponsesStreamOptions) (*AssistantMessageEventStream, error) {
	prepared, err := prepareOpenAISimpleOptions(rawModel, transcript, options.Options, false)
	if err != nil {
		return nil, err
	}
	options.Options = prepared
	return StreamOpenAIResponses(ctx, rawModel, transcript, options), nil
}

func prepareOpenAISimpleOptions(rawModel json.RawMessage, transcript TranscriptContext, rawOptions json.RawMessage, includeThinkingBudgets bool) (json.RawMessage, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	controls := map[string]json.RawMessage{}
	if len(rawOptions) > 0 {
		if err := json.Unmarshal(rawOptions, &controls); err != nil {
			return nil, err
		}
	}
	if _, err := completionsAPIKey(model.Provider, controls); err != nil {
		return nil, err
	}
	return buildOpenAISimpleOptions(rawModel, transcript, rawOptions, includeThinkingBudgets)
}
