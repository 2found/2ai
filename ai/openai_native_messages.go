package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode/utf16"

	"github.com/2found/2ai/jsonjs"
)

// ConvertOpenAICompletionsMessages ports Pi's exported convertMessages. The
// input remains a native transcript; no legacy agentcore.Message projection is
// used. Grammar properties come from CreateGrammarToolInputProperties.
func ConvertOpenAICompletionsMessages(rawModel json.RawMessage, context TranscriptContext, compat OpenAICompletionsCompat, grammarProperties map[string]string) (json.RawMessage, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	normalized := ResolveTranscript(context, compat.SupportsMidConvoSystemMessages)
	transformed := TransformMessages(normalized.Messages(), Model{ID: model.ID, API: model.API, Provider: model.Provider, Input: model.Input}, func(id string, _ *Model, _ *Message) string { return normalizeCompletionsToolID(id, model.Provider) })
	toolState := ResolveTranscriptTools(normalized.Messages(), compat.SupportsMidConvoSystemMessages && compat.SupportsMidConvoToolAdditions)
	role := "system"
	if model.Reasoning && compat.SupportsDeveloperRole {
		role = "developer"
	}
	params := []map[string]any{}
	bridge := func() {
		params = append(params, map[string]any{"role": "assistant", "content": "I have processed the tool results."})
	}
	lastRole := ""
	for i := 0; i < len(transformed); i++ {
		message := transformed[i]
		if compat.RequiresAssistantAfterToolResult && lastRole == "toolResult" && message.Role == "user" {
			bridge()
		}
		switch message.Role {
		case "system":
			if i > 0 && toolState.AnchorsAdditions && len(message.ToolsAdded) > 0 {
				tools, err := ConvertOpenAICompletionsTools(message.ToolsAdded, compat)
				if err != nil {
					return nil, err
				}
				params = append(params, map[string]any{"role": "system", "tools": tools})
			}
			text := GetSystemMessageText(message)
			if i > 0 {
				text = RenderSystemMessageUpdate(message)
			}
			if text != "" {
				params = append(params, map[string]any{"role": role, "content": SanitizeSurrogates(text)})
			}
		case "user":
			if message.Content.Text != nil {
				params = append(params, map[string]any{"role": "user", "content": SanitizeSurrogates(*message.Content.Text)})
			} else {
				content := []map[string]any{}
				for _, block := range message.Content.Blocks.Values() {
					if block.Type == "text" {
						if block.Text != "" {
							content = append(content, map[string]any{"type": "text", "text": SanitizeSurrogates(block.Text)})
						}
					} else {
						content = append(content, completionsImage(block))
					}
				}
				if len(content) == 0 {
					continue
				}
				params = append(params, map[string]any{"role": "user", "content": content})
			}
		case "assistant":
			assistant := map[string]any{"role": "assistant", "content": nil}
			if compat.RequiresAssistantAfterToolResult {
				assistant["content"] = ""
			}
			parts := []map[string]any{}
			texts := []string{}
			thinking := []*ContentBlock{}
			calls := []*ContentBlock{}
			for _, block := range message.Content.Blocks.Values() {
				switch block.Type {
				case "text":
					if strings.TrimFunc(block.Text, jsWhitespace) != "" {
						parts = append(parts, map[string]any{"type": "text", "text": SanitizeSurrogates(block.Text)})
						texts = append(texts, SanitizeSurrogates(block.Text))
					}
				case "thinking":
					thinking = append(thinking, block)
				case "toolCall":
					calls = append(calls, block)
				}
			}
			details := []json.RawMessage(nil)
			for _, block := range thinking {
				if block.ThinkingSignature != nil {
					if parsed := parseCompletionsReasoningDetails(*block.ThinkingSignature); parsed != nil {
						details = parsed
						break
					}
				}
			}
			if details == nil {
				for _, call := range calls {
					if call.ThoughtSignature != nil {
						raw := json.RawMessage(*call.ThoughtSignature)
						if isCompletionsReasoningDetail(raw) {
							object, _ := samplingObject(raw)
							if samplingString(object["type"]) == "reasoning.encrypted" && samplingString(object["id"]) != "" && samplingString(object["data"]) != "" {
								normalized, _ := stringifyCompletionsJSON(raw)
								details = append(details, normalized)
							}
						}
					}
				}
			}
			nonempty := []*ContentBlock{}
			thinkingText := []string{}
			for _, block := range thinking {
				if strings.TrimFunc(block.Thinking, jsWhitespace) != "" {
					nonempty = append(nonempty, block)
					thinkingText = append(thinkingText, block.Thinking)
				}
			}
			text := strings.Join(texts, "")
			if len(nonempty) > 0 {
				if compat.RequiresThinkingAsText {
					assistant["content"] = append([]map[string]any{{"type": "text", "text": SanitizeSurrogates(strings.Join(thinkingText, "\n\n"))}}, parts...)
				} else {
					if text != "" {
						assistant["content"] = text
					}
					if details == nil && nonempty[0].ThinkingSignature != nil {
						signature := *nonempty[0].ThinkingSignature
						if model.Provider == "opencode-go" && signature == "reasoning" {
							signature = "reasoning_content"
						}
						if signature == "reasoning" || signature == "reasoning_content" || signature == "reasoning_text" {
							assistant[signature] = json.RawMessage(jsonjs.QuoteString(strings.Join(thinkingText, "\n")))
						}
					}
				}
			} else if text != "" {
				assistant["content"] = text
			}
			if len(calls) > 0 {
				output := []map[string]any{}
				for _, call := range calls {
					if property, ok := grammarProperties[call.Name]; ok {
						input, err := GetGrammarToolInput(call.Name, call.Arguments, property)
						if err != nil {
							return nil, err
						}
						output = append(output, map[string]any{"id": call.ID, "type": "custom", "custom": map[string]any{"name": call.Name, "input": SanitizeSurrogates(input)}})
					} else {
						arguments, err := stringifyCompletionsJSON(call.Arguments)
						if err != nil {
							return nil, err
						}
						output = append(output, map[string]any{"id": call.ID, "type": "function", "function": map[string]any{"name": call.Name, "arguments": string(arguments)}})
					}
				}
				assistant["tool_calls"] = output
			}
			if details != nil {
				assistant["reasoning_details"] = details
			}
			if compat.RequiresReasoningContentOnAssistantMessages && model.Reasoning {
				if _, ok := assistant["reasoning_content"]; !ok {
					assistant["reasoning_content"] = ""
				}
			}
			hasContent := false
			switch content := assistant["content"].(type) {
			case string:
				hasContent = len(content) > 0
			case []map[string]any:
				hasContent = len(content) > 0
			}
			if !hasContent && len(calls) == 0 {
				continue
			}
			params = append(params, assistant)
		case "toolResult":
			images := []map[string]any{}
			for ; i < len(transformed) && transformed[i].Role == "toolResult"; i++ {
				tool := transformed[i]
				texts := []string{}
				hasImages := false
				for _, block := range tool.Content.Blocks.Values() {
					if block.Type == "text" {
						texts = append(texts, block.Text)
					}
					if block.Type == "image" {
						hasImages = true
						if slices.Contains(model.Input, "image") {
							images = append(images, completionsImage(block))
						}
					}
				}
				text := strings.Join(texts, "\n")
				if text == "" {
					if hasImages {
						text = "(see attached image)"
					} else {
						text = "(no tool output)"
					}
				}
				result := map[string]any{"role": "tool", "content": SanitizeSurrogates(text), "tool_call_id": tool.ToolCallID}
				if compat.RequiresToolResultName && tool.ToolName != "" {
					result["name"] = tool.ToolName
				}
				params = append(params, result)
			}
			i--
			if len(images) > 0 {
				if compat.RequiresAssistantAfterToolResult {
					bridge()
				}
				content := append([]map[string]any{{"type": "text", "text": "Attached image(s) from tool result:"}}, images...)
				params = append(params, map[string]any{"role": "user", "content": content})
				lastRole = "user"
			} else {
				lastRole = "toolResult"
			}
			continue
		}
		lastRole = message.Role
	}
	return json.Marshal(params)
}

// ConvertOpenAICompletionsTools shares Pi's strict and grammar decisions with
// both top-level declarations and mid-conversation tool additions.
func ConvertOpenAICompletionsTools(tools []Tool, compat OpenAICompletionsCompat) (json.RawMessage, error) {
	output := []map[string]any{}
	for _, tool := range tools {
		grammar, err := ResolveGrammarConstrainedSampling(tool, compat.SupportsOpenAIGrammarTools)
		if err != nil {
			return nil, err
		}
		if grammar != nil {
			output = append(output, map[string]any{"type": "custom", "custom": map[string]any{"name": tool.Name, "description": tool.Description, "format": map[string]any{"type": "grammar", "grammar": map[string]any{"syntax": grammar.Format, "definition": grammar.Definition}}}})
			continue
		}
		strict, err := ResolveJSONSchemaStrictSampling(tool, compat.SupportsStrictMode, nil)
		if err != nil {
			return nil, err
		}
		parameters, err := GetJSONSchemaToolParameters(tool, strict)
		if err != nil {
			return nil, err
		}
		function := map[string]any{"name": tool.Name, "description": tool.Description, "parameters": parameters}
		if compat.SupportsStrictMode {
			function["strict"] = strict != nil && *strict
		}
		output = append(output, map[string]any{"type": "function", "function": function})
	}
	return json.Marshal(output)
}
func completionsImage(block *ContentBlock) map[string]any {
	return map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:" + block.MIMEType + ";base64," + block.Data}}
}
func normalizeCompletionsToolID(id, provider string) string {
	if separator := strings.IndexByte(id, '|'); separator >= 0 {
		sanitize := func(value string) string {
			var out strings.Builder
			for _, unit := range utf16.Encode([]rune(value)) {
				if unit >= 'a' && unit <= 'z' || unit >= 'A' && unit <= 'Z' || unit >= '0' && unit <= '9' || unit == '_' || unit == '-' {
					out.WriteByte(byte(unit))
				} else {
					out.WriteByte('_')
				}
			}
			return out.String()
		}
		call, item := sanitize(id[:separator]), sanitize(id[separator+1:])
		combined := call
		if item != "" {
			combined += "_" + item
		}
		if len(combined) <= 40 {
			return combined
		}
		hash := completionsShortHash(id)
		if len(hash) > 8 {
			hash = hash[:8]
		}
		limit := max(1, 40-len(hash)-1)
		if len(call) > limit {
			call = call[:limit]
		}
		return call + "_" + hash
	}
	if provider == "openai" {
		units := utf16.Encode([]rune(id))
		if len(units) > 40 {
			return string(utf16.Decode(units[:40]))
		}
	}
	return id
}
func completionsShortHash(value string) string {
	h1, h2 := uint32(0xdeadbeef), uint32(0x41c6ce57)
	for _, unit := range utf16.Encode([]rune(value)) {
		h1 = (h1 ^ uint32(unit)) * 2654435761
		h2 = (h2 ^ uint32(unit)) * 1597334677
	}
	h1 = (h1^(h1>>16))*2246822507 ^ (h2^(h2>>13))*3266489909
	h2 = (h2^(h2>>16))*2246822507 ^ (h1^(h1>>13))*3266489909
	return strconv.FormatUint(uint64(h2), 36) + strconv.FormatUint(uint64(h1), 36)
}
func parseCompletionsReasoningDetails(signature string) []json.RawMessage {
	var details []json.RawMessage
	if json.Unmarshal([]byte(signature), &details) != nil || len(details) == 0 {
		return nil
	}
	for i, detail := range details {
		if !isCompletionsReasoningDetail(detail) {
			return nil
		}
		// Pi JSON.parse materializes JS numbers inside the provider payload. The
		// original signature string in transcript storage must remain untouched.
		details[i], _ = stringifyCompletionsJSON(detail)
	}
	return details
}
func isCompletionsReasoningDetail(raw json.RawMessage) bool {
	detail, ok := samplingObject(raw)
	if !ok {
		return false
	}
	isString := func(key string, optional, nullable bool) bool {
		value, exists := detail[key]
		if !exists {
			return optional
		}
		if nullable && string(bytes.TrimSpace(value)) == "null" {
			return true
		}
		var text string
		return len(value) > 0 && bytes.TrimSpace(value)[0] == '"' && json.Unmarshal(value, &text) == nil
	}
	if !isString("id", true, true) || !isString("format", true, false) {
		return false
	}
	if index, exists := detail["index"]; exists {
		index = bytes.TrimSpace(index)
		if len(index) == 0 || !(index[0] == '-' || index[0] >= '0' && index[0] <= '9') {
			return false
		}
		number, err := strconv.ParseFloat(string(index), 64)
		if err != nil && !math.IsInf(number, 0) {
			return false
		}
	}
	switch samplingString(detail["type"]) {
	case "reasoning.summary":
		return isString("summary", false, false)
	case "reasoning.encrypted":
		return isString("data", false, false)
	case "reasoning.text":
		return isString("text", false, false) && isString("signature", true, true)
	}
	return false
}

// Arguments are a JSON string within the request, so their property ordering,
// string escapes and number spelling are observable, unlike request object keys.
func stringifyCompletionsJSON(raw json.RawMessage) (json.RawMessage, error) {
	if !json.Valid(raw) {
		return nil, fmt.Errorf("invalid tool arguments JSON")
	}
	return jsonjs.StringifyJSON(raw)
}
