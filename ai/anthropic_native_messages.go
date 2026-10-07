package ai

import (
	"encoding/json"
	"slices"
	"strings"
	"unicode/utf16"
)

type AnthropicToolsOptions struct {
	OAuth               bool            `json:"oauth"`
	EagerInputStreaming bool            `json:"eagerInputStreaming"`
	StrictTools         bool            `json:"strictTools"`
	CacheControl        json.RawMessage `json:"cacheControl,omitempty"`
}

type AnthropicMessagesOptions struct {
	OAuth                  bool                                  `json:"oauth"`
	CacheControl           json.RawMessage                       `json:"cacheControl,omitempty"`
	AllowEmptySignature    bool                                  `json:"allowEmptySignature"`
	ManagedProvider        *string                               `json:"managedProvider,omitempty"`
	ConvertToolDefinitions func([]Tool) (json.RawMessage, error) `json:"-"`
}

type AnthropicConvertedMessages struct {
	Messages        json.RawMessage `json:"messages"`
	AssistantLevels map[int]string  `json:"assistantLevels"`
}

var anthropicCanonicalTools = []string{"Read", "Write", "Edit", "Bash", "Grep", "Glob", "AskUserQuestion", "EnterPlanMode", "ExitPlanMode", "KillShell", "NotebookEdit", "Skill", "Task", "TaskOutput", "TodoWrite", "WebFetch", "WebSearch"}

func anthropicToolName(name string, oauth bool) string {
	if oauth {
		// JS lowercasing expands dotted I and does not fold long s to ASCII.
		// Only matches against the ASCII canonical catalogue are observable.
		lower := strings.ToLower(strings.ReplaceAll(name, "İ", "i\u0307"))
		for _, canonical := range anthropicCanonicalTools {
			if lower == strings.ToLower(canonical) {
				return canonical
			}
		}
	}
	return name
}
func normalizeAnthropicToolID(id string) string {
	result := make([]byte, 0, 64)
	for _, unit := range utf16.Encode([]rune(id)) {
		if len(result) == 64 {
			break
		}
		if unit >= 'a' && unit <= 'z' || unit >= 'A' && unit <= 'Z' || unit >= '0' && unit <= '9' || unit == '_' || unit == '-' {
			result = append(result, byte(unit))
		} else {
			result = append(result, '_')
		}
	}
	return string(result)
}
func anthropicUnsupportedStrictKeyword(key string, value json.RawMessage) bool {
	switch key {
	case "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "maxItems", "uniqueItems", "minContains", "maxContains", "minProperties", "maxProperties":
		return true
	case "minItems":
		var number float64
		if len(value) == 0 || value[0] == '"' || !samplingNonNull(value) || json.Unmarshal(value, &number) != nil {
			return true
		}
		return number != 0 && number != 1
	case "format":
		return !slices.Contains([]string{"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}, samplingString(value))
	}
	return false
}

// ConvertAnthropicTools retains Pi's legacy schema projection unless a supported
// strict constraint is selected. Only the last declaration gets a cache marker.
func ConvertAnthropicTools(tools []Tool, options AnthropicToolsOptions) (json.RawMessage, error) {
	output := []map[string]any{}
	for index, tool := range tools {
		strict, err := ResolveJSONSchemaStrictSampling(tool, options.StrictTools, anthropicUnsupportedStrictKeyword)
		if err != nil {
			return nil, err
		}
		parameters, err := GetJSONSchemaToolParameters(tool, strict)
		if err != nil {
			return nil, err
		}
		schema, _ := samplingObject(parameters)
		input := map[string]json.RawMessage{}
		if strict != nil && *strict {
			for key, value := range schema {
				input[key] = value
			}
		}
		input["type"] = json.RawMessage(`"object"`)
		input["properties"] = schema["properties"]
		input["required"] = schema["required"]
		if !samplingNonNull(input["properties"]) {
			input["properties"] = json.RawMessage(`{}`)
		}
		if !samplingNonNull(input["required"]) {
			input["required"] = json.RawMessage(`[]`)
		}
		item := map[string]any{"name": anthropicToolName(tool.Name, options.OAuth), "description": tool.Description, "input_schema": input}
		if options.EagerInputStreaming {
			item["eager_input_streaming"] = true
		}
		if strict != nil && *strict {
			item["strict"] = true
		}
		if samplingTruthy(options.CacheControl) && index == len(tools)-1 {
			item["cache_control"] = options.CacheControl
		}
		output = append(output, item)
	}
	return json.Marshal(output)
}
func anthropicImage(block *ContentBlock) map[string]any {
	return map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": block.MIMEType, "data": block.Data}}
}
func anthropicToolResultContent(content []*ContentBlock) any {
	images := false
	for _, block := range content {
		if block.Type == "image" {
			images = true
			break
		}
	}
	if !images {
		texts := make([]string, len(content))
		for i, block := range content {
			texts[i] = block.Text
		}
		return SanitizeSurrogates(strings.Join(texts, "\n"))
	}
	blocks := []map[string]any{}
	hasText := false
	for _, block := range content {
		if block.Type == "text" {
			blocks = append(blocks, map[string]any{"type": "text", "text": SanitizeSurrogates(block.Text)})
			hasText = true
		} else {
			blocks = append(blocks, anthropicImage(block))
		}
	}
	if !hasText {
		blocks = append([]map[string]any{{"type": "text", "text": "(see attached image)"}}, blocks...)
	}
	return blocks
}
func isAnthropicEffort(effort string) bool {
	return slices.Contains([]string{"low", "medium", "high", "xhigh", "max"}, effort)
}

// ConvertAnthropicMessages consumes an already transformed conversation (the
// initial system message is sent separately). Later system updates are delayed
// until the next assistant so tool results remain adjacent to their tool use.
func ConvertAnthropicMessages(messages []Message, options AnthropicMessagesOptions) (AnthropicConvertedMessages, error) {
	params := []map[string]any{}
	pending := []map[string]any{}
	levels := map[int]string{}
	flush := func() { params = append(params, pending...); pending = nil }
	nonempty := func(text string) bool { return strings.TrimFunc(text, jsWhitespace) != "" }
	for i := 0; i < len(messages); i++ {
		message := messages[i]
		switch message.Role {
		case "system":
			blocks := []map[string]any{}
			text := RenderSystemMessageUpdate(message)
			if text != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": SanitizeSurrogates(text)})
			}
			if options.ConvertToolDefinitions != nil {
				names := map[string]bool{}
				for _, tool := range message.ToolsAdded {
					names[tool.Name] = true
				}
				for _, tool := range message.ToolsRemoved {
					if !names[tool.Name] {
						blocks = append(blocks, map[string]any{"type": "tool_removal", "tool": map[string]any{"type": "tool_reference", "name": anthropicToolName(tool.Name, options.OAuth)}})
					}
				}
				converted, err := options.ConvertToolDefinitions(message.ToolsAdded)
				if err != nil {
					return AnthropicConvertedMessages{}, err
				}
				var tools []json.RawMessage
				if err := json.Unmarshal(converted, &tools); err != nil {
					return AnthropicConvertedMessages{}, err
				}
				for _, tool := range tools {
					blocks = append(blocks, map[string]any{"type": "tool_addition", "tool": map[string]any{"type": "tool_definition", "definition": tool}})
				}
			}
			if len(blocks) > 0 {
				pending = append(pending, map[string]any{"role": "system", "content": blocks})
			}
		case "user":
			if message.Content.Text != nil {
				if nonempty(*message.Content.Text) {
					params = append(params, map[string]any{"role": "user", "content": SanitizeSurrogates(*message.Content.Text)})
				}
			} else {
				blocks := []map[string]any{}
				for _, block := range message.Content.Blocks.Values() {
					if block.Type == "text" {
						text := SanitizeSurrogates(block.Text)
						if nonempty(text) {
							blocks = append(blocks, map[string]any{"type": "text", "text": text})
						}
					} else {
						blocks = append(blocks, anthropicImage(block))
					}
				}
				if len(blocks) > 0 {
					params = append(params, map[string]any{"role": "user", "content": blocks})
				}
			}
		case "assistant":
			flush()
			blocks := []map[string]any{}
			for _, block := range message.Content.Blocks.Values() {
				switch block.Type {
				case "text":
					if nonempty(block.Text) {
						blocks = append(blocks, map[string]any{"type": "text", "text": SanitizeSurrogates(block.Text)})
					}
				case "thinking":
					if block.Redacted != nil && *block.Redacted {
						item := map[string]any{"type": "redacted_thinking"}
						if block.ThinkingSignature != nil {
							item["data"] = *block.ThinkingSignature
						}
						blocks = append(blocks, item)
						continue
					}
					signed := block.ThinkingSignature != nil && nonempty(*block.ThinkingSignature)
					if !nonempty(block.Thinking) && !signed {
						continue
					}
					if !signed && !options.AllowEmptySignature {
						blocks = append(blocks, map[string]any{"type": "text", "text": SanitizeSurrogates(block.Thinking)})
					} else {
						signature := ""
						if signed {
							signature = *block.ThinkingSignature
						}
						blocks = append(blocks, map[string]any{"type": "thinking", "thinking": SanitizeSurrogates(block.Thinking), "signature": signature})
					}
				case "toolCall":
					input := block.Arguments
					if !samplingNonNull(input) {
						input = json.RawMessage(`{}`)
					}
					blocks = append(blocks, map[string]any{"type": "tool_use", "id": block.ID, "name": anthropicToolName(block.Name, options.OAuth), "input": input})
				}
			}
			if len(blocks) == 0 {
				continue
			}
			index := len(params)
			params = append(params, map[string]any{"role": "assistant", "content": blocks})
			if options.ManagedProvider != nil && message.API == "anthropic-messages" && message.Provider == *options.ManagedProvider && message.ProviderThinkingLevel != nil && isAnthropicEffort(*message.ProviderThinkingLevel) {
				levels[index] = *message.ProviderThinkingLevel
			}
		case "toolResult":
			results := []map[string]any{}
			for ; i < len(messages) && messages[i].Role == "toolResult"; i++ {
				result := messages[i]
				results = append(results, map[string]any{"type": "tool_result", "tool_use_id": result.ToolCallID, "content": anthropicToolResultContent(result.Content.Blocks.Values()), "is_error": result.IsError})
			}
			i--
			params = append(params, map[string]any{"role": "user", "content": results})
		}
	}
	flush()
	if samplingTruthy(options.CacheControl) && len(params) > 0 {
		last := params[len(params)-1]
		if last["role"] == "user" || last["role"] == "system" {
			switch content := last["content"].(type) {
			case string:
				last["content"] = []map[string]any{{"type": "text", "text": content, "cache_control": options.CacheControl}}
			case []map[string]any:
				if len(content) > 0 {
					block := content[len(content)-1]
					if slices.Contains([]string{"text", "image", "tool_result", "tool_addition", "tool_removal"}, block["type"].(string)) {
						block["cache_control"] = options.CacheControl
					}
				}
			}
		}
	}
	encoded, err := json.Marshal(params)
	return AnthropicConvertedMessages{Messages: encoded, AssistantLevels: levels}, err
}

func insertAnthropicThinkingLevelMessages(converted AnthropicConvertedMessages, effort json.RawMessage) (json.RawMessage, error) {
	var messages []json.RawMessage
	if err := json.Unmarshal(converted.Messages, &messages); err != nil {
		return nil, err
	}
	output := []json.RawMessage{}
	instruction := func(value json.RawMessage) json.RawMessage {
		raw, _ := json.Marshal(map[string]any{"role": "system", "content": []any{}, "output_config": map[string]json.RawMessage{"effort": value}})
		return raw
	}
	for index, message := range messages {
		if historical, ok := converted.AssistantLevels[index]; ok {
			output = append(output, instruction(json.RawMessage(marshalSamplingString(historical))))
		}
		output = append(output, message)
	}
	output = append(output, instruction(effort))
	return json.Marshal(output)
}
