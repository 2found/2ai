package ai

import (
	"encoding/json"
	"os"
	"strings"
)

type AnthropicCompat struct {
	SupportsEagerToolInputStreaming bool    `json:"supportsEagerToolInputStreaming"`
	SupportsLongCacheRetention      bool    `json:"supportsLongCacheRetention"`
	SendSessionAffinityHeaders      bool    `json:"sendSessionAffinityHeaders"`
	SessionAffinityFormat           *string `json:"sessionAffinityFormat,omitempty"`
	SupportsCacheControlOnTools     bool    `json:"supportsCacheControlOnTools"`
	SupportsTemperature             bool    `json:"supportsTemperature"`
	AllowEmptySignature             bool    `json:"allowEmptySignature"`
	SupportsStrictTools             bool    `json:"supportsStrictTools"`
	SupportsMidConvoSystemMessages  bool    `json:"supportsMidConvoSystemMessages"`
	SupportsMidConvoToolChanges     bool    `json:"supportsMidConvoToolChanges"`
}

func ResolveAnthropicCompat(rawModel json.RawMessage) (AnthropicCompat, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return AnthropicCompat{}, err
	}
	compat := AnthropicCompat{SupportsEagerToolInputStreaming: true, SupportsLongCacheRetention: true, SupportsCacheControlOnTools: true, SupportsTemperature: true}
	if model.Provider == "openrouter" || strings.Contains(model.BaseURL, "openrouter.ai") {
		value := "openrouter"
		compat.SendSessionAffinityHeaders = true
		compat.SessionAffinityFormat = &value
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
func anthropicCacheControl(compat AnthropicCompat, options map[string]json.RawMessage) json.RawMessage {
	retention := samplingString(options["cacheRetention"])
	if retention == "" {
		env, _ := samplingObject(options["env"])
		value := samplingString(env["PI_CACHE_RETENTION"])
		if value == "" {
			value = os.Getenv("PI_CACHE_RETENTION")
		}
		retention = "short"
		if value == "long" {
			retention = "long"
		}
	}
	if retention == "none" {
		return nil
	}
	if retention == "long" && compat.SupportsLongCacheRetention {
		return json.RawMessage(`{"type":"ephemeral","ttl":"1h"}`)
	}
	return json.RawMessage(`{"type":"ephemeral"}`)
}
func anthropicBetaFeatures(rawModel json.RawMessage, model completionsModel, context TranscriptContext, oauth, nativeTools bool, options map[string]json.RawMessage) []string {
	fields, _ := samplingObject(rawModel)
	var configured json.RawMessage
	for _, raw := range []json.RawMessage{fields["headers"], options["headers"]} {
		headers, _ := samplingObject(raw)
		for _, name := range samplingObjectKeys(raw) {
			if strings.EqualFold(name, "anthropic-beta") {
				configured = headers[name]
			}
		}
	}
	result := []string{}
	seen := map[string]bool{}
	add := func(value string) {
		if !seen[value] {
			seen[value] = true
			result = append(result, value)
		}
	}
	if len(configured) > 0 {
		if !samplingNonNull(configured) {
			return result
		}
		for _, feature := range strings.Split(samplingString(configured), ",") {
			if value := strings.TrimFunc(feature, jsWhitespace); value != "" {
				add(value)
			}
		}
		return result
	}
	if oauth {
		add("claude-code-20250219")
		add("oauth-2025-04-20")
	}
	compat, _ := ResolveAnthropicCompat(rawModel)
	if len(GetCurrentTools(context.Messages())) > 0 && !compat.SupportsEagerToolInputStreaming {
		add("fine-grained-tool-streaming-2025-05-14")
	}
	if model.Reasoning && string(options["thinkingEnabled"]) == "true" && (!samplingNonNull(options["interleavedThinking"]) || samplingTruthy(options["interleavedThinking"])) && string(model.Compat["forceAdaptiveThinking"]) != "true" {
		add("interleaved-thinking-2025-05-14")
	}
	var fallbacks []json.RawMessage
	_ = json.Unmarshal(model.Compat["allowedFallbackModels"], &fallbacks)
	if len(fallbacks) > 0 {
		add("server-side-fallback-2026-07-01")
	}
	if string(model.Compat["supportsMidConvoEffort"]) == "true" {
		add("mid-conversation-output-config-2026-07-01")
		add("thinking-binding-controls-2026-08-01")
	}
	if nativeTools {
		add("inline-tools-2026-09-15")
	}
	return result
}

// BuildAnthropicParams ports Pi's request builder. OAuth is supplied by the
// client/auth selection; this function does not resolve tokens or federation.
// The caller resolves mid-conversation system-message support beforehand.
func BuildAnthropicParams(rawModel json.RawMessage, context TranscriptContext, oauth bool, rawOptions json.RawMessage) (json.RawMessage, error) {
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
	compat, err := ResolveAnthropicCompat(rawModel)
	if err != nil {
		return nil, err
	}
	cache := anthropicCacheControl(compat, options)
	initial := GetInitialSystemMessage(context.Messages())
	system := ""
	var initialTools []Tool
	if initial != nil {
		system = GetSystemMessageText(*initial)
		initialTools = initial.ToolsAdded
	}
	transformed := TransformMessages(context.Messages(), Model{ID: model.ID, API: model.API, Provider: model.Provider, Input: model.Input}, func(id string, _ *Model, _ *Message) string { return normalizeAnthropicToolID(id) })
	if initial != nil {
		transformed = transformed[1:]
	}
	nativeTools := compat.SupportsMidConvoSystemMessages && compat.SupportsMidConvoToolChanges && len(initialTools) > 0
	toolOptions := AnthropicToolsOptions{OAuth: oauth, EagerInputStreaming: compat.SupportsEagerToolInputStreaming, StrictTools: compat.SupportsStrictTools}
	conversion := AnthropicMessagesOptions{OAuth: oauth, CacheControl: cache, AllowEmptySignature: compat.AllowEmptySignature}
	managed := string(model.Compat["supportsMidConvoEffort"]) == "true"
	if managed {
		conversion.ManagedProvider = &model.Provider
	}
	if nativeTools {
		conversion.ConvertToolDefinitions = func(tools []Tool) (json.RawMessage, error) { return ConvertAnthropicTools(tools, toolOptions) }
	}
	converted, err := ConvertAnthropicMessages(transformed, conversion)
	if err != nil {
		return nil, err
	}
	active := options["effort"]
	if !samplingNonNull(active) {
		active = json.RawMessage(`"high"`)
	}
	messages := converted.Messages
	if managed {
		messages, err = insertAnthropicThinkingLevelMessages(converted, active)
		if err != nil {
			return nil, err
		}
	}
	params := map[string]json.RawMessage{}
	set := func(key string, value any) { params[key], _ = json.Marshal(value) }
	set("model", model.ID)
	params["messages"] = messages
	set("stream", true)
	params["max_tokens"] = options["maxTokens"]
	if !samplingNonNull(params["max_tokens"]) {
		set("max_tokens", model.MaxTokens)
	}
	betas := anthropicBetaFeatures(rawModel, model, context, oauth, nativeTools, options)
	if len(betas) > 0 {
		set("betas", betas)
	}
	systemBlocks := []map[string]any{}
	systemBlock := func(text string) map[string]any {
		block := map[string]any{"type": "text", "text": text}
		if samplingTruthy(cache) {
			block["cache_control"] = cache
		}
		return block
	}
	if oauth {
		systemBlocks = append(systemBlocks, systemBlock("You are Claude Code, Anthropic's official CLI for Claude."))
	}
	if system != "" {
		systemBlocks = append(systemBlocks, systemBlock(SanitizeSurrogates(system)))
	}
	if len(systemBlocks) > 0 {
		set("system", systemBlocks)
	}
	if value, exists := options["temperature"]; exists && !samplingTruthy(options["thinkingEnabled"]) && !managed && compat.SupportsTemperature {
		params["temperature"] = value
	}
	if compat.SupportsCacheControlOnTools {
		toolOptions.CacheControl = cache
	}
	tools := GetCurrentTools(context.Messages())
	if nativeTools {
		tools = initialTools
	}
	if len(tools) > 0 {
		declarations, err := ConvertAnthropicTools(tools, toolOptions)
		if err != nil {
			return nil, err
		}
		if nativeTools {
			var list []json.RawMessage
			_ = json.Unmarshal(declarations, &list)
			list = append(list, json.RawMessage(`{"name":"__pi_deferred_placeholder__","description":"Reserved placeholder. Never available. Never call this.","input_schema":{"type":"object","properties":{},"required":[]},"defer_loading":true}`))
			declarations, _ = json.Marshal(list)
		}
		params["tools"] = declarations
	}
	display := options["thinkingDisplay"]
	if !samplingNonNull(display) {
		display = json.RawMessage(`"summarized"`)
	}
	if managed {
		set("thinking", map[string]any{"type": "adaptive", "display": display, "block_binding": map[string]string{"prefix_mismatch_behavior": "drop_block"}})
		set("output_config", map[string]string{"effort": "high"})
	} else if model.Reasoning {
		if samplingTruthy(options["thinkingEnabled"]) {
			if string(model.Compat["forceAdaptiveThinking"]) == "true" {
				set("thinking", map[string]any{"type": "adaptive", "display": display})
				if samplingTruthy(options["effort"]) {
					set("output_config", map[string]json.RawMessage{"effort": options["effort"]})
				}
			} else {
				budget := options["thinkingBudgetTokens"]
				if !samplingTruthy(budget) {
					budget = json.RawMessage(`1024`)
				}
				set("thinking", map[string]any{"type": "enabled", "budget_tokens": budget, "display": display})
			}
		} else if string(options["thinkingEnabled"]) == "false" && strings.TrimSpace(string(model.ThinkingLevelMap["off"])) != "null" {
			set("thinking", map[string]string{"type": "disabled"})
		}
	}
	metadata, _ := samplingObject(options["metadata"])
	if value := metadata["user_id"]; len(value) > 0 && value[0] == '"' {
		set("metadata", map[string]json.RawMessage{"user_id": value})
	}
	if choice := options["toolChoice"]; samplingTruthy(choice) {
		if choice[0] == '"' {
			set("tool_choice", map[string]json.RawMessage{"type": choice})
		} else {
			params["tool_choice"] = choice
		}
	}
	var fallbacks []map[string]json.RawMessage
	_ = json.Unmarshal(model.Compat["allowedFallbackModels"], &fallbacks)
	if len(fallbacks) > 0 {
		values := make([]map[string]json.RawMessage, len(fallbacks))
		for i, fallback := range fallbacks {
			values[i] = map[string]json.RawMessage{}
			if id, exists := fallback["model"]; exists {
				values[i]["model"] = id
			}
		}
		set("fallbacks", values)
	}
	return json.Marshal(params)
}
