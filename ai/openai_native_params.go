package ai

import (
	"encoding/json"
	"os"
	"strings"
)

// BuildOpenAICompletionsParams ports Pi's request-body builder. Options are the
// provider-specific (not streamSimple) options. Credentials, callbacks and HTTP
// configuration never enter the body unless explicitly set by samplingParams,
// whose final override semantics match the original provider.
func BuildOpenAICompletionsParams(rawModel json.RawMessage, context TranscriptContext, rawOptions json.RawMessage) (json.RawMessage, error) {
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
	compat, err := ResolveOpenAICompletionsCompat(rawModel)
	if err != nil {
		return nil, err
	}
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
	grammar, err := CreateGrammarToolInputProperties(GetDeclaredTools(context.Messages()), compat.SupportsOpenAIGrammarTools)
	if err != nil {
		return nil, err
	}
	messages, err := ConvertOpenAICompletionsMessages(rawModel, context, compat, grammar)
	if err != nil {
		return nil, err
	}
	toolState := ResolveTranscriptTools(context.Messages(), compat.SupportsMidConvoSystemMessages && compat.SupportsMidConvoToolAdditions)
	params := map[string]json.RawMessage{}
	set := func(key string, value any) { params[key], _ = json.Marshal(value) }
	set("model", model.ID)
	params["messages"] = messages
	set("stream", true)
	if strings.Contains(model.BaseURL, "api.openai.com") && cache != "none" || cache == "long" && compat.SupportsLongCacheRetention {
		if value, exists := options["sessionId"]; exists {
			key := []rune(samplingString(value))
			if len(key) > 64 {
				key = key[:64]
			}
			set("prompt_cache_key", string(key))
		}
	}
	if cache == "long" && compat.SupportsLongCacheRetention {
		set("prompt_cache_retention", "24h")
	}
	if compat.SupportsUsageInStreaming {
		set("stream_options", map[string]bool{"include_usage": true})
	}
	if compat.SupportsStore {
		set("store", false)
	}
	if samplingTruthy(options["maxTokens"]) {
		field := "max_completion_tokens"
		if compat.MaxTokensField == "max_tokens" {
			field = "max_tokens"
		}
		params[field] = options["maxTokens"]
	}
	if value, exists := options["temperature"]; exists {
		params["temperature"] = value
	}
	if len(toolState.RequestTools) > 0 {
		params["tools"], err = ConvertOpenAICompletionsTools(toolState.RequestTools, compat)
		if err != nil {
			return nil, err
		}
		if compat.ZaiToolStream {
			set("tool_stream", true)
		}
	} else if completionsHasToolHistory(context.Messages()) {
		params["tools"] = json.RawMessage(`[]`)
	}
	if compat.CacheControlFormat == "anthropic" && cache != "none" {
		control := map[string]string{"type": "ephemeral"}
		if cache == "long" && compat.SupportsLongCacheRetention {
			control["ttl"] = "1h"
		}
		params["messages"], params["tools"] = completionsCacheControl(params["messages"], params["tools"], control)
		if params["tools"] == nil {
			delete(params, "tools")
		}
	}
	if samplingTruthy(options["toolChoice"]) {
		params["tool_choice"] = options["toolChoice"]
	}
	if compat.VLLMPriority != nil {
		params["priority"] = compat.VLLMPriority
	}
	effort := samplingString(options["reasoningEffort"])
	var budget *float64
	if effort != "" && model.Reasoning {
		ceiling := model.MaxTokens
		for _, field := range []string{"max_tokens", "max_completion_tokens"} {
			if value, exists := params[field]; exists && string(value) != "null" {
				_ = json.Unmarshal(value, &ceiling)
				break
			}
		}
		custom, _ := samplingObject(options["thinkingBudgets"])
		value := ClampThinkingBudgetToAnswerRoom(ThinkingBudgetForLevel(effort, custom), ceiling)
		if value > 0 {
			budget = &value
		}
	}
	mapped := func(level, fallback string) json.RawMessage {
		value, exists := model.ThinkingLevelMap[level]
		if !exists || string(value) == "null" {
			value, _ = json.Marshal(fallback)
		}
		return value
	}
	mappedString := func(level, fallback string) (string, bool) {
		value, exists := model.ThinkingLevelMap[level]
		if !exists {
			return fallback, fallback != ""
		}
		var text string
		if len(value) == 0 || value[0] != '"' || json.Unmarshal(value, &text) != nil {
			return "", false
		}
		return text, true
	}
	offAllowed := string(model.ThinkingLevelMap["off"]) != "null"
	if model.Reasoning {
		switch compat.ThinkingFormat {
		case "zai":
			if effort != "" {
				set("thinking", map[string]any{"type": "enabled", "clear_thinking": false})
			} else {
				set("thinking", map[string]string{"type": "disabled"})
			}
			if effort != "" && compat.SupportsReasoningEffort {
				if value, ok := mappedString(effort, effort); ok {
					set("reasoning_effort", value)
				}
			}
		case "qwen":
			set("enable_thinking", effort != "")
			if effort != "" && compat.SupportsReasoningEffort {
				params["reasoning_effort"] = mapped(effort, effort)
			}
		case "qwen-chat-template":
			set("chat_template_kwargs", map[string]bool{"enable_thinking": effort != "", "preserve_thinking": true})
		case "chat-template":
			if values := completionsTemplateValues(model, effort, compat.ChatTemplateKwargs, budget); len(values) > 0 {
				set("chat_template_kwargs", values)
			}
		case "baseten":
			if values := completionsTemplateValues(model, effort, compat.ChatTemplateArgs, budget); len(values) > 0 {
				set("chat_template_args", values)
			}
			if compat.SupportsReasoningEffort {
				level := effort
				if level == "" {
					level = "off"
				}
				if value, ok := mappedString(level, effort); ok {
					set("reasoning_effort", value)
				}
			}
		case "deepseek":
			if effort != "" {
				set("thinking", map[string]string{"type": "enabled"})
			} else if offAllowed {
				set("thinking", map[string]string{"type": "disabled"})
			}
			if effort != "" && compat.SupportsReasoningEffort {
				params["reasoning_effort"] = mapped(effort, effort)
			}
		case "openrouter":
			if effort != "" {
				set("reasoning", map[string]json.RawMessage{"effort": mapped(effort, effort)})
			} else if offAllowed {
				set("reasoning", map[string]json.RawMessage{"effort": mapped("off", "none")})
			}
		case "ant-ling":
			if effort != "" {
				if value, ok := mappedString(effort, ""); ok {
					set("reasoning", map[string]string{"effort": value})
				}
			} else if compat.SupportsReasoningEffort {
				if value, ok := mappedString("off", ""); ok {
					set("reasoning_effort", value)
				}
			}
		case "together":
			set("reasoning", map[string]bool{"enabled": effort != ""})
			if effort != "" && compat.SupportsReasoningEffort {
				params["reasoning_effort"] = mapped(effort, effort)
			}
		case "string-thinking":
			if effort != "" {
				params["thinking"] = mapped(effort, effort)
			} else if offAllowed {
				params["thinking"] = mapped("off", "none")
			}
		default:
			if compat.SupportsReasoningEffort {
				if effort != "" {
					params["reasoning_effort"] = mapped(effort, effort)
				} else if value, ok := mappedString("off", ""); ok {
					set("reasoning_effort", value)
				}
			}
		}
	}
	budgetField := compat.ThinkingTokenBudgetField
	if budgetField == "" && compat.SupportsThinkingTokenBudget {
		budgetField = "thinking_token_budget"
	}
	if budgetField != "" && budget != nil {
		set(budgetField, *budget)
	}
	if value := model.Compat["openRouterRouting"]; samplingTruthy(value) {
		params["provider"] = value
	}
	if routing, ok := samplingObject(model.Compat["vercelGatewayRouting"]); ok {
		gateway := map[string]json.RawMessage{}
		for _, key := range []string{"only", "order"} {
			if samplingTruthy(routing[key]) {
				gateway[key] = routing[key]
			}
		}
		if len(gateway) > 0 {
			set("providerOptions", map[string]any{"gateway": gateway})
		}
	}
	for key, value := range resolveSamplingParams(model, samplingString(options["reasoningEffort"]), options["samplingParams"]) {
		params[key] = value
	}
	return json.Marshal(params)
}

func completionsTemplateValues(model completionsModel, effort string, raw json.RawMessage, budget *float64) map[string]json.RawMessage {
	values, _ := samplingObject(raw)
	result := map[string]json.RawMessage{}
	for key, value := range values {
		variable, isObject := samplingObject(value)
		if !isObject {
			result[key] = value
			continue
		}
		if effort == "" && samplingTruthy(variable["omitWhenOff"]) {
			continue
		}
		switch samplingString(variable["$var"]) {
		case "thinking.enabled":
			result[key], _ = json.Marshal(effort != "")
		case "thinking.budget":
			if budget != nil {
				result[key], _ = json.Marshal(*budget)
			}
		default:
			level := effort
			if level == "" {
				level = "off"
			}
			mapped, exists := model.ThinkingLevelMap[level]
			if !exists {
				if effort != "" {
					result[key], _ = json.Marshal(effort)
				}
			} else if len(mapped) > 0 && mapped[0] == '"' {
				result[key] = mapped
			}
		}
	}
	return result
}
func samplingTruthy(raw json.RawMessage) bool {
	value := strings.TrimSpace(string(raw))
	if value == "" || value == "null" || value == "false" || value == `""` {
		return false
	}
	if value[0] == '-' || value[0] >= '0' && value[0] <= '9' {
		var number float64
		if json.Unmarshal(raw, &number) == nil {
			return number != 0
		}
	}
	return true
}
func completionsHasToolHistory(messages []Message) bool {
	for _, message := range messages {
		if message.Role == "toolResult" {
			return true
		}
		if message.Role == "assistant" {
			for _, block := range message.Content.Blocks.Values() {
				if block.Type == "toolCall" {
					return true
				}
			}
		}
	}
	return false
}
func completionsCacheControl(messagesRaw, toolsRaw json.RawMessage, control map[string]string) (json.RawMessage, json.RawMessage) {
	var messages, tools []map[string]json.RawMessage
	_ = json.Unmarshal(messagesRaw, &messages)
	_ = json.Unmarshal(toolsRaw, &tools)
	cache, _ := json.Marshal(control)
	add := func(message map[string]json.RawMessage) bool {
		content := message["content"]
		if len(content) > 0 && content[0] == '"' {
			text := samplingString(content)
			if text == "" {
				return false
			}
			message["content"], _ = json.Marshal([]any{map[string]any{"type": "text", "text": text, "cache_control": control}})
			return true
		}
		var parts []map[string]json.RawMessage
		if json.Unmarshal(content, &parts) != nil {
			return false
		}
		for i := len(parts) - 1; i >= 0; i-- {
			if samplingString(parts[i]["type"]) == "text" {
				parts[i]["cache_control"] = cache
				message["content"], _ = json.Marshal(parts)
				return true
			}
		}
		return false
	}
	for _, message := range messages {
		role := samplingString(message["role"])
		if role == "system" || role == "developer" {
			add(message)
			break
		}
	}
	if len(tools) > 0 {
		tools[len(tools)-1]["cache_control"] = cache
	}
	for i := len(messages) - 1; i >= 0; i-- {
		role := samplingString(messages[i]["role"])
		if role == "user" || role == "assistant" || role == "tool" {
			if add(messages[i]) {
				break
			}
		}
	}
	result, _ := json.Marshal(messages)
	if toolsRaw == nil {
		return result, nil
	}
	toolResult, _ := json.Marshal(tools)
	return result, toolResult
}
