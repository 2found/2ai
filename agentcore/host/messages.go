package host

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/ai/protocol"
)

// ProjectMessage produces a display/observer view. Never use it as a native checkpoint.
func ProjectMessage(raw json.RawMessage) (protocol.Message, error) {
	var native struct {
		Role, ToolCallID, ToolName string
		Content                    json.RawMessage
		Usage                      *struct {
			Input, Output, CacheRead, CacheWrite int
			Cost                                 *struct{ Total float64 }
		}
	}
	if err := json.Unmarshal(raw, &native); err != nil {
		return protocol.Message{}, err
	}
	m := protocol.Message{Role: protocol.Role(native.Role), ToolCallID: native.ToolCallID, Name: native.ToolName}
	if native.Role == "toolResult" {
		m.Role = protocol.RoleTool
	}
	// Custom messages are retained in native state without imposing a legacy
	// content schema or making their extension payload model-visible text.
	if native.Role == "system" || native.Role == "user" || native.Role == "assistant" || native.Role == "toolResult" {
		var err error
		m.Content, m.ContentParts, m.ToolCalls, err = ProjectContent(native.Content)
		if err != nil {
			return m, err
		}
	}
	if native.Role == "system" {
		var system ai.Message
		if err := json.Unmarshal(raw, &system); err != nil {
			return m, err
		}
		m.Content = ai.GetCurrentSystemPrompt([]ai.Message{system})
	}
	if native.Usage != nil {
		u := native.Usage
		m.Usage = &protocol.Usage{InputTokens: u.Input, OutputTokens: u.Output, CacheReadTokens: u.CacheRead, CacheWriteTokens: u.CacheWrite, CostUnpriced: u.Cost == nil}
		if u.Cost != nil {
			m.Usage.CostUSD = u.Cost.Total
		}
	}
	return m, nil
}

// ProjectContent extracts display text, images and calls from native content.
func ProjectContent(raw json.RawMessage) (string, []protocol.ContentPart, []protocol.ToolCall, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil, nil, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text, nil, nil, nil
	}
	var blocks []struct {
		Type, Text, Data, MimeType, ID, Name string
		Arguments                            json.RawMessage
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", nil, nil, err
	}
	var parts []protocol.ContentPart
	var calls []protocol.ToolCall
	for _, block := range blocks {
		switch block.Type {
		case "text":
			text += block.Text
		case "image":
			parts = append(parts, protocol.ContentPart{Type: protocol.ContentPartImage, MIMEType: block.MimeType, Data: block.Data})
		case "toolCall":
			calls = append(calls, protocol.ToolCall{ID: block.ID, Name: block.Name, Arguments: string(block.Arguments)})
		}
	}
	return text, parts, calls, nil
}

// These are newly authored host instructions, never provider/history messages.
// InputMessages converts newly authored host input only; provider history remains native.
func InputMessages(messages []protocol.Message) ([]json.RawMessage, error) {
	result := make([]json.RawMessage, 0, len(messages))
	for _, message := range messages {
		if message.Role != protocol.RoleSystem && message.Role != protocol.RoleUser {
			return nil, fmt.Errorf("Pi host injection cannot impersonate role %q", message.Role)
		}
		if len(message.ToolCalls) > 0 || message.ToolCallID != "" {
			return nil, errors.New("Pi host injection cannot contain tool calls/results")
		}
		var content any = message.Content
		if len(message.ContentParts) > 0 {
			parts := make([]any, 0, len(message.ContentParts)+1)
			if message.Content != "" {
				parts = append(parts, map[string]any{"type": "text", "text": message.Content})
			}
			for _, part := range message.ContentParts {
				switch part.Type {
				case protocol.ContentPartText:
					parts = append(parts, map[string]any{"type": "text", "text": part.Text})
				case protocol.ContentPartImage:
					if message.Role == protocol.RoleSystem {
						return nil, errors.New("native system messages cannot contain images")
					}
					parts = append(parts, map[string]any{"type": "image", "mimeType": part.MIMEType, "data": part.Data})
				default:
					return nil, fmt.Errorf("unsupported host content %q", part.Type)
				}
			}
			content = parts
		}
		value := map[string]any{"role": message.Role, "content": content, "timestamp": time.Now().UnixMilli()}
		if message.InputID != "" {
			value["agentrayInputId"] = message.InputID
		}
		raw, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		result = append(result, raw)
	}
	return result, nil
}
