package ai

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"strings"
)

type OpenAIResponsesCompat struct {
	SupportsDeveloperRole           bool   `json:"supportsDeveloperRole"`
	SupportsMidConvoSystemMessages  bool   `json:"supportsMidConvoSystemMessages"`
	SessionAffinityFormat           string `json:"sessionAffinityFormat"`
	SupportsLongCacheRetention      bool   `json:"supportsLongCacheRetention"`
	SupportsStrictMode              bool   `json:"supportsStrictMode"`
	SupportsOpenAIGrammarTools      bool   `json:"supportsOpenAIGrammarTools"`
	SupportsAdditionalTools         bool   `json:"supportsAdditionalTools"`
	SupportsToolSearch              bool   `json:"supportsToolSearch"`
	SupportsExplicitPromptCacheMode bool   `json:"supportsExplicitPromptCacheMode"`
	SupportsMaxOutputTokens         bool   `json:"supportsMaxOutputTokens"`
}

func ResolveOpenAIResponsesCompat(raw json.RawMessage) (OpenAIResponsesCompat, error) {
	var model completionsModel
	if err := json.Unmarshal(raw, &model); err != nil {
		return OpenAIResponsesCompat{}, err
	}
	compat := OpenAIResponsesCompat{SupportsDeveloperRole: true, SessionAffinityFormat: "openai", SupportsLongCacheRetention: true, SupportsMaxOutputTokens: true}
	if model.Provider == "openrouter" || strings.Contains(model.BaseURL, "openrouter.ai") {
		compat.SessionAffinityFormat = "openrouter"
	}
	overrides := map[string]json.RawMessage{}
	for key, value := range model.Compat {
		if samplingNonNull(value) {
			overrides[key] = value
		}
	}
	encoded, err := json.Marshal(overrides)
	if err != nil {
		return compat, err
	}
	err = json.Unmarshal(encoded, &compat)
	return compat, err
}

// BuildOpenAIResponsesParams builds the native Responses request body. Callers
// supply the resolved transcript, as Pi's stream does before invoking its builder.
func BuildOpenAIResponsesParams(rawModel json.RawMessage, context TranscriptContext, rawOptions json.RawMessage) (json.RawMessage, error) {
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
	compat, err := ResolveOpenAIResponsesCompat(rawModel)
	if err != nil {
		return nil, err
	}
	grammar, err := CreateGrammarToolInputProperties(GetDeclaredTools(context.Messages()), compat.SupportsOpenAIGrammarTools)
	if err != nil {
		return nil, err
	}
	toolOptions := ResponsesToolsOptions{SupportsStrictMode: &compat.SupportsStrictMode, SupportsOpenAIGrammarTools: compat.SupportsOpenAIGrammarTools}
	messages, err := ConvertResponsesMessages(rawModel, context, []string{"openai", "openai-codex", "opencode"}, ResponsesMessagesOptions{
		GrammarToolInputProperties: grammar, SupportsMidConvoSystemMessages: compat.SupportsMidConvoSystemMessages, SupportsAdditionalTools: compat.SupportsAdditionalTools, SupportsToolSearch: compat.SupportsToolSearch, ToolOptions: toolOptions,
	})
	if err != nil {
		return nil, err
	}
	state := ResolveTranscriptTools(context.Messages(), compat.SupportsAdditionalTools || compat.SupportsToolSearch)
	cache := samplingString(options["cacheRetention"])
	if cache == "" {
		env, _ := samplingObject(options["env"])
		value := samplingString(env["PI_CACHE_RETENTION"])
		if value == "" {
			value = os.Getenv("PI_CACHE_RETENTION")
		}
		cache = "short"
		if value == "long" {
			cache = "long"
		}
	}
	apiKey, hasKey := options["apiKey"]
	// Pi checks undefined, then calls startsWith on the original option. A
	// nullable key admitted through an authorization header still fails here.
	if model.Provider == "openai" && model.BaseURL == "https://api.openai.com/v1" && hasKey && !samplingNonNull(apiKey) {
		return nil, errors.New("null is not an object (evaluating 'apiKey.startsWith')")
	}
	chatGPT := model.Provider == "openai" && model.BaseURL == "https://api.openai.com/v1" && hasKey && !strings.HasPrefix(samplingString(apiKey), "sk-")
	params := map[string]json.RawMessage{}
	set := func(key string, value any) { params[key], _ = json.Marshal(value) }
	set("model", model.ID)
	params["input"] = messages
	set("stream", true)
	set("store", false)
	if cache != "none" {
		if session, exists := options["sessionId"]; exists {
			params["prompt_cache_key"], err = ClampOpenAIPromptCacheKey(session)
			if err != nil {
				return nil, err
			}
		}
	}
	if !chatGPT {
		if cache == "long" && compat.SupportsLongCacheRetention && !compat.SupportsExplicitPromptCacheMode {
			set("prompt_cache_retention", "24h")
		}
		if compat.SupportsExplicitPromptCacheMode {
			if cache == "none" {
				set("prompt_cache_options", map[string]string{"mode": "explicit"})
			} else if cache == "long" && compat.SupportsLongCacheRetention {
				set("prompt_cache_options", map[string]string{"ttl": "30m"})
			}
		}
		if samplingTruthy(options["maxTokens"]) && compat.SupportsMaxOutputTokens {
			var tokens float64
			if err := json.Unmarshal(options["maxTokens"], &tokens); err != nil {
				return nil, err
			}
			set("max_output_tokens", math.Max(tokens, 16))
		}
		if value, exists := options["temperature"]; exists {
			params["temperature"] = value
		}
	}
	if value, exists := options["serviceTier"]; exists {
		params["service_tier"] = value
	}
	if len(state.RequestTools) > 0 {
		params["tools"], err = ConvertResponsesTools(state.RequestTools, toolOptions)
		if err != nil {
			return nil, err
		}
	}
	if value, exists := options["toolChoice"]; exists {
		params["tool_choice"] = value
	}
	reasoningEffort := options["reasoningEffort"]
	if !samplingNonNull(reasoningEffort) && samplingTruthy(options["reasoningSummary"]) {
		reasoningEffort = json.RawMessage(`"medium"`)
	}
	if model.Reasoning {
		if samplingTruthy(reasoningEffort) {
			effort := reasoningEffort
			if samplingTruthy(options["reasoningEffort"]) {
				effort = options["reasoningEffort"]
				if mapped := model.ThinkingLevelMap[samplingString(effort)]; samplingNonNull(mapped) {
					effort = mapped
				}
			}
			summary := options["reasoningSummary"]
			if !samplingTruthy(summary) {
				summary = json.RawMessage(`"auto"`)
			}
			set("reasoning", map[string]json.RawMessage{"effort": effort, "summary": summary})
			set("include", []string{"reasoning.encrypted_content"})
		} else if model.Provider != "github-copilot" && strings.TrimSpace(string(model.ThinkingLevelMap["off"])) != "null" {
			effort := model.ThinkingLevelMap["off"]
			if !samplingNonNull(effort) {
				effort = json.RawMessage(`"none"`)
			}
			set("reasoning", map[string]json.RawMessage{"effort": effort})
		}
		if model.Provider == "xai" {
			set("include", []string{"reasoning.encrypted_content"})
		}
	}
	for key, value := range resolveSamplingParams(model, samplingString(reasoningEffort), options["samplingParams"]) {
		params[key] = value
	}
	return json.Marshal(params)
}
