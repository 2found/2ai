package ai

import (
	"encoding/json"

	"github.com/2found/2ai/ai/protocol"
)

// Protocol converts native usage to the provider-neutral accounting contract,
// keeping the reasoning and 1-hour cache-write subsets.
func (u Usage) Protocol() protocol.Usage {
	subset := func(value *float64) int {
		if value == nil {
			return 0
		}
		return int(*value)
	}
	return protocol.Usage{InputTokens: int(u.Input), OutputTokens: int(u.Output), CacheReadTokens: int(u.CacheRead), CacheWriteTokens: int(u.CacheWrite),
		CacheWrite1hTokens: subset(u.CacheWrite1h), ReasoningTokens: subset(u.Reasoning), CostUSD: u.Cost.Total}
}

// ProjectMessage produces a display/accounting view of one native message.
// It is lossy by design; never use it as a native checkpoint. Usage without a
// cost object projects as unpriced.
func ProjectMessage(raw json.RawMessage) (protocol.Message, error) {
	var native struct {
		Role, ToolCallID, ToolName string
		Content                    json.RawMessage
		Usage                      json.RawMessage
	}
	if err := json.Unmarshal(raw, &native); err != nil {
		return protocol.Message{}, err
	}
	m := protocol.Message{Role: protocol.Role(native.Role), ToolCallID: native.ToolCallID, Name: native.ToolName}
	switch native.Role {
	case "toolResult":
		m.Role = protocol.RoleTool
		fallthrough
	case "user", "assistant":
		var err error
		if m.Content, m.ContentParts, m.ToolCalls, err = ProjectContent(native.Content); err != nil {
			return m, err
		}
	case "system":
		// Custom roles stay out of model-visible text; system text includes sections.
		var system Message
		if err := json.Unmarshal(raw, &system); err != nil {
			return m, err
		}
		m.Content = GetCurrentSystemPrompt([]Message{system})
	}
	if len(native.Usage) > 0 && string(native.Usage) != "null" {
		var usage Usage
		var cost struct{ Cost json.RawMessage }
		if err := json.Unmarshal(native.Usage, &usage); err != nil {
			return m, err
		}
		_ = json.Unmarshal(native.Usage, &cost)
		projected := usage.Protocol()
		projected.CostUnpriced = len(cost.Cost) == 0 || string(cost.Cost) == "null"
		m.Usage = &projected
	}
	return m, nil
}

// ProjectContent extracts display text, images and tool calls from native
// message content (a string or a block array).
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
