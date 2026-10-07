package ai

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode/utf16"
)

// ResponsesToolsOptions preserves the distinction between absent and null strict.
// SupportsStrictMode defaults to true, as in Pi's shared Responses converter.
type ResponsesToolsOptions struct {
	Strict                     json.RawMessage `json:"strict,omitempty"`
	SupportsStrictMode         *bool           `json:"supportsStrictMode,omitempty"`
	SupportsOpenAIGrammarTools bool            `json:"supportsOpenAIGrammarTools,omitempty"`
	ToolSearchResult           bool            `json:"toolSearchResult,omitempty"`
}

type ResponsesMessagesOptions struct {
	IncludeSystemPrompt            *bool                 `json:"includeSystemPrompt,omitempty"`
	GrammarToolInputProperties     map[string]string     `json:"grammarToolInputProperties,omitempty"`
	SupportsMidConvoSystemMessages bool                  `json:"supportsMidConvoSystemMessages,omitempty"`
	SupportsAdditionalTools        bool                  `json:"supportsAdditionalTools,omitempty"`
	SupportsToolSearch             bool                  `json:"supportsToolSearch,omitempty"`
	ToolOptions                    ResponsesToolsOptions `json:"toolOptions,omitempty"`
}

// ConvertResponsesTools converts declarations for the Responses API, including
// strict schemas, grammar tools and deferred tool-search results.
func ConvertResponsesTools(tools []Tool, options ResponsesToolsOptions) (json.RawMessage, error) {
	output := []map[string]any{}
	supportsStrict := options.SupportsStrictMode == nil || *options.SupportsStrictMode
	for _, tool := range tools {
		grammar, err := ResolveGrammarConstrainedSampling(tool, options.SupportsOpenAIGrammarTools)
		if err != nil {
			return nil, err
		}
		item := map[string]any{"name": tool.Name, "description": tool.Description}
		if grammar != nil {
			item["type"] = "custom"
			item["format"] = map[string]any{"type": "grammar", "syntax": grammar.Format, "definition": grammar.Definition}
		} else {
			constrained, err := ResolveJSONSchemaStrictSampling(tool, supportsStrict, nil)
			if err != nil {
				return nil, err
			}
			strict := options.Strict
			if len(strict) == 0 {
				strict = json.RawMessage(`false`)
			}
			if constrained != nil {
				strict, _ = json.Marshal(*constrained)
			}
			enabled := string(strict) == "true"
			parameters, err := GetJSONSchemaToolParameters(tool, &enabled)
			if err != nil {
				return nil, err
			}
			item["type"] = "function"
			item["parameters"] = parameters
			if supportsStrict {
				item["strict"] = strict
			}
		}
		if options.ToolSearchResult {
			item["defer_loading"] = true
		}
		output = append(output, item)
	}
	return json.Marshal(output)
}

// ConvertResponsesMessages ports the shared Responses transcript conversion.
// allowedToolCallProviders is supplied by the concrete provider; it controls
// cross-provider call-ID normalization, not signature replay permissions.
func ConvertResponsesMessages(rawModel json.RawMessage, context TranscriptContext, allowedToolCallProviders []string, options ResponsesMessagesOptions) (json.RawMessage, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	normalized := ResolveTranscript(context, options.SupportsMidConvoSystemMessages)
	transformed := TransformMessages(normalized.Messages(), Model{ID: model.ID, API: model.API, Provider: model.Provider, Input: model.Input}, func(id string, _ *Model, source *Message) string {
		if !slices.Contains(allowedToolCallProviders, model.Provider) || !strings.Contains(id, "|") {
			return normalizeResponsesIDPart(id)
		}
		parts := strings.Split(id, "|")
		call, item := normalizeResponsesIDPart(parts[0]), normalizeResponsesIDPart(parts[1])
		if source.Provider != model.Provider || source.API != model.API {
			item = "fc_" + completionsShortHash(parts[1])
		}
		if !strings.HasPrefix(item, "fc_") {
			item = normalizeResponsesIDPart("fc_" + item)
		}
		return call + "|" + item
	})
	state := ResolveTranscriptTools(normalized.Messages(), options.SupportsAdditionalTools || options.SupportsToolSearch)
	output := []any{}
	role := "system"
	if model.Reasoning && string(model.Compat["supportsDeveloperRole"]) != "false" {
		role = "developer"
	}
	msgIndex := 0
	for sourceIndex, message := range transformed {
		leading := sourceIndex == 0 && message.Role == "system"
		switch message.Role {
		case "system":
			if !leading && state.AnchorsAdditions && len(message.ToolsAdded) > 0 {
				if options.SupportsAdditionalTools {
					tools, err := ConvertResponsesTools(message.ToolsAdded, options.ToolOptions)
					if err != nil {
						return nil, err
					}
					output = append(output, map[string]any{"type": "additional_tools", "role": "developer", "tools": tools})
				} else if options.SupportsToolSearch {
					names := make([]string, len(message.ToolsAdded))
					for i, tool := range message.ToolsAdded {
						names[i] = tool.Name
					}
					call := "pi_tool_load_" + completionsShortHash(fmt.Sprintf("system:%d:%s", msgIndex, strings.Join(names, ",")))
					toolOptions := options.ToolOptions
					toolOptions.ToolSearchResult = true
					tools, err := ConvertResponsesTools(message.ToolsAdded, toolOptions)
					if err != nil {
						return nil, err
					}
					output = append(output, map[string]any{"type": "tool_search_call", "call_id": call, "execution": "client", "status": "completed", "arguments": map[string]any{"query": strings.Join(names, " "), "limit": len(names)}}, map[string]any{"type": "tool_search_output", "call_id": call, "execution": "client", "status": "completed", "tools": tools})
				}
			}
			if !leading || options.IncludeSystemPrompt == nil || *options.IncludeSystemPrompt {
				text := GetSystemMessageText(message)
				if !leading {
					text = RenderSystemMessageUpdate(message)
				}
				if text != "" {
					output = append(output, map[string]any{"role": role, "content": SanitizeSurrogates(text)})
				}
			}
		case "user":
			content := []map[string]any{}
			if message.Content.Text != nil {
				content = append(content, map[string]any{"type": "input_text", "text": SanitizeSurrogates(*message.Content.Text)})
			} else {
				for _, block := range message.Content.Blocks.Values() {
					if block.Type == "text" {
						content = append(content, map[string]any{"type": "input_text", "text": SanitizeSurrogates(block.Text)})
					} else {
						content = append(content, responsesImage(block))
					}
				}
				if len(content) == 0 {
					continue
				}
			}
			output = append(output, map[string]any{"role": "user", "content": content})
		case "assistant":
			before := len(output)
			sameProviderAPI := message.Provider == model.Provider && message.API == model.API
			sameModel := sameProviderAPI && message.Model == model.ID
			textIndex := 0
			for _, block := range message.Content.Blocks.Values() {
				switch block.Type {
				case "thinking":
					if block.ThinkingSignature != nil && *block.ThinkingSignature != "" {
						reasoning, err := stringifyCompletionsJSON(json.RawMessage(*block.ThinkingSignature))
						if err != nil {
							return nil, err
						}
						output = append(output, reasoning)
					}
				case "text":
					id, phase := parseResponsesTextSignature(block.TextSignature)
					if id == "" {
						id = fmt.Sprintf("msg_pi_%d", msgIndex)
						if textIndex > 0 {
							id += fmt.Sprintf("_%d", textIndex)
						}
					} else if len(utf16.Encode([]rune(id))) > 64 {
						id = "msg_" + completionsShortHash(id)
					}
					textIndex++
					item := map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": SanitizeSurrogates(block.Text), "annotations": []any{}}}, "status": "completed", "id": id}
					if phase != "" {
						item["phase"] = phase
					}
					output = append(output, item)
				case "toolCall":
					parts := strings.Split(block.ID, "|")
					item := map[string]any{"call_id": parts[0], "name": block.Name}
					property, custom := options.GrammarToolInputProperties[block.Name]
					prefix := "fc_"
					if custom {
						prefix = "ctc_"
						input, err := GetGrammarToolInput(block.Name, block.Arguments, property)
						if err != nil {
							return nil, err
						}
						item["type"] = "custom_tool_call"
						item["input"] = SanitizeSurrogates(input)
					} else {
						item["type"] = "function_call"
						if len(block.Arguments) > 0 {
							args, err := stringifyCompletionsJSON(block.Arguments)
							if err != nil {
								return nil, err
							}
							item["arguments"] = string(args)
						}
					}
					if !(sameProviderAPI && !sameModel) && len(parts) > 1 && strings.HasPrefix(parts[1], prefix) {
						item["id"] = parts[1]
					}
					if sameModel {
						if block.Namespace != nil {
							item["namespace"] = *block.Namespace
						} else if raw, exists := block.Extra["namespace"]; exists {
							item["namespace"] = raw
						}
					}
					output = append(output, item)
				}
			}
			if len(output) == before {
				continue
			}
		case "toolResult":
			kind := "function_call_output"
			if _, ok := options.GrammarToolInputProperties[message.ToolName]; ok {
				kind = "custom_tool_call_output"
			}
			output = append(output, map[string]any{"type": kind, "call_id": strings.Split(message.ToolCallID, "|")[0], "output": responsesToolResultOutput(model, message.Content.Blocks.Values())})
		}
		if !leading {
			msgIndex++
		}
	}
	return json.Marshal(output)
}

func normalizeResponsesIDPart(value string) string {
	out := make([]byte, 0, 64)
	for _, unit := range utf16.Encode([]rune(value)) {
		if len(out) == 64 {
			break
		}
		if unit >= 'a' && unit <= 'z' || unit >= 'A' && unit <= 'Z' || unit >= '0' && unit <= '9' || unit == '_' || unit == '-' {
			out = append(out, byte(unit))
		} else {
			out = append(out, '_')
		}
	}
	return strings.TrimRight(string(out), "_")
}
func parseResponsesTextSignature(signature *string) (id, phase string) {
	if signature == nil || *signature == "" {
		return "", ""
	}
	if strings.HasPrefix(*signature, "{") {
		object, ok := samplingObject(json.RawMessage(*signature))
		if ok {
			var version float64
			var parsedID string
			if json.Unmarshal(object["v"], &version) == nil && version == 1 && len(object["id"]) > 0 && object["id"][0] == '"' && json.Unmarshal(object["id"], &parsedID) == nil {
				phase = samplingString(object["phase"])
				if phase != "commentary" && phase != "final_answer" {
					phase = ""
				}
				return parsedID, phase
			}
		}
	}
	return *signature, ""
}
func responsesImage(block *ContentBlock) map[string]any {
	return map[string]any{"type": "input_image", "detail": "auto", "image_url": "data:" + block.MIMEType + ";base64," + block.Data}
}
func responsesToolResultOutput(model completionsModel, blocks []*ContentBlock) any {
	texts := []string{}
	images := []*ContentBlock{}
	for _, block := range blocks {
		if block.Type == "text" {
			texts = append(texts, block.Text)
		} else if block.Type == "image" {
			images = append(images, block)
		}
	}
	text := strings.Join(texts, "\n")
	if len(images) == 0 || !slices.Contains(model.Input, "image") {
		if text != "" {
			return SanitizeSurrogates(text)
		}
		if len(images) > 0 {
			return "(see attached image)"
		}
		return "(no tool output)"
	}
	output := []map[string]any{}
	if text != "" {
		output = append(output, map[string]any{"type": "input_text", "text": SanitizeSurrogates(text)})
	}
	for _, block := range images {
		output = append(output, responsesImage(block))
	}
	return output
}
