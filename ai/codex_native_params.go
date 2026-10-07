package ai

import (
	"encoding/json"
	"strings"

	"github.com/2found/2ai/jsonjs"
)

// BuildCodexResponsesParams ports Pi's Codex request builder. The transcript
// must already be normalized/resolved, and cacheSessionID is the transport's
// chosen cache identity (nil preserves an absent prompt_cache_key).
func BuildCodexResponsesParams(rawModel json.RawMessage, context TranscriptContext, rawOptions json.RawMessage, cacheSessionID *string) (json.RawMessage, error) {
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
	compat := OpenAIResponsesCompat{SupportsStrictMode: true}
	overrides := map[string]json.RawMessage{}
	for key, value := range model.Compat {
		if samplingNonNull(value) {
			overrides[key] = value
		}
	}
	encoded, err := json.Marshal(overrides)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(encoded, &compat); err != nil {
		return nil, err
	}
	grammar, err := CreateGrammarToolInputProperties(GetDeclaredTools(context.Messages()), compat.SupportsOpenAIGrammarTools)
	if err != nil {
		return nil, err
	}
	toolOptions := ResponsesToolsOptions{Strict: json.RawMessage(`null`), SupportsStrictMode: &compat.SupportsStrictMode, SupportsOpenAIGrammarTools: compat.SupportsOpenAIGrammarTools}
	includeSystem := false
	messages, err := ConvertResponsesMessages(rawModel, context, []string{"openai", "openai-codex", "opencode"}, ResponsesMessagesOptions{
		IncludeSystemPrompt: &includeSystem, GrammarToolInputProperties: grammar,
		SupportsMidConvoSystemMessages: compat.SupportsMidConvoSystemMessages,
		SupportsAdditionalTools:        compat.SupportsAdditionalTools, SupportsToolSearch: compat.SupportsToolSearch, ToolOptions: toolOptions,
	})
	if err != nil {
		return nil, err
	}
	instructions := ""
	if initial := GetInitialSystemMessage(context.Messages()); initial != nil {
		instructions = GetSystemMessageText(*initial)
	}
	if instructions == "" {
		instructions = "You are a helpful assistant."
	}
	body := map[string]json.RawMessage{}
	set := func(key string, value any) { body[key], _ = json.Marshal(value) }
	set("model", model.ID)
	set("store", false)
	set("stream", true)
	body["instructions"] = jsonjs.QuoteString(instructions)
	body["input"] = messages
	verbosity := options["textVerbosity"]
	if !samplingTruthy(verbosity) {
		verbosity = json.RawMessage(`"low"`)
	}
	set("text", map[string]json.RawMessage{"verbosity": verbosity})
	set("include", []string{"reasoning.encrypted_content"})
	if cacheSessionID != nil {
		set("prompt_cache_key", *cacheSessionID)
	}
	body["tool_choice"] = options["toolChoice"]
	if !samplingNonNull(body["tool_choice"]) {
		body["tool_choice"] = json.RawMessage(`"auto"`)
	}
	set("parallel_tool_calls", true)
	for input, output := range map[string]string{"temperature": "temperature", "serviceTier": "service_tier"} {
		if value, exists := options[input]; exists {
			body[output] = value
		}
	}
	tools := ResolveTranscriptTools(context.Messages(), compat.SupportsAdditionalTools || compat.SupportsToolSearch)
	if len(tools.RequestTools) > 0 {
		body["tools"], err = ConvertResponsesTools(tools.RequestTools, toolOptions)
		if err != nil {
			return nil, err
		}
	}
	if effort, exists := options["reasoningEffort"]; exists {
		if samplingString(effort) == "none" {
			if mapped, exists := model.ThinkingLevelMap["off"]; exists {
				effort = mapped
			}
		} else if mapped := model.ThinkingLevelMap[samplingString(effort)]; samplingNonNull(mapped) {
			effort = mapped
		}
		if samplingNonNull(effort) {
			summary := options["reasoningSummary"]
			if !samplingNonNull(summary) {
				summary = json.RawMessage(`"auto"`)
			}
			set("reasoning", map[string]json.RawMessage{"effort": effort, "summary": summary})
		}
	} else if model.Reasoning && strings.TrimSpace(string(model.ThinkingLevelMap["off"])) != "null" {
		effort := model.ThinkingLevelMap["off"]
		if !samplingNonNull(effort) {
			effort = json.RawMessage(`"none"`)
		}
		set("reasoning", map[string]json.RawMessage{"effort": effort})
	}
	return json.Marshal(body)
}
