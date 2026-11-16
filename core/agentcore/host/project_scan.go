package host

import (
	"encoding/json"

	"github.com/2found/2ai/core/ai"
	"github.com/2found/2ai/core/ai/protocol"
)

// projectFields are the message members ai.ProjectMessage reads.
var projectFields = [...]string{"role", "toolCallId", "toolName", "content", "usage"}

// projectBlockFields are the content block members ai.ProjectContent reads.
var projectBlockFields = [...]string{"type", "text", "data", "mimeType", "id", "name", "arguments"}

// ProjectMessage is ai.ProjectMessage. A message in the shape 2ai writes is
// read in one pass over its bytes, without decoding it; any other is left to
// ai.ProjectMessage, so the projection and its errors are the same.
func ProjectMessage(raw json.RawMessage) (protocol.Message, error) {
	if m, ok := scanProjection(raw); ok {
		return m, nil
	}
	return ai.ProjectMessage(raw)
}

// scanProjection is ai.ProjectMessage for raw that is valid JSON. ok is false
// for anything the scan does not cover: a system message, an escaped, invalid
// UTF-8 or differently spelled member, a repeated member, or a value of
// another type.
func scanProjection(raw json.RawMessage) (m protocol.Message, ok bool) {
	if len(raw) == 0 || raw[0] != '{' {
		return m, false
	}
	var values [len(projectFields)][]byte
	if !collectMembers(raw, projectFields[:], values[:]) {
		return m, false
	}
	var role string
	if role, ok = plainText(values[0]); !ok {
		return m, false
	}
	if m.ToolCallID, ok = plainText(values[1]); !ok {
		return m, false
	}
	if m.Name, ok = plainText(values[2]); !ok {
		return m, false
	}
	m.Role = protocol.Role(role)
	switch role {
	case "system":
		return m, false
	case "toolResult":
		m.Role = protocol.RoleTool
		fallthrough
	case "user", "assistant":
		if m.Content, m.ContentParts, m.ToolCalls, ok = scanContent(values[3]); !ok {
			return m, false
		}
	}
	if usage := values[4]; usage != nil && !isNull(usage) {
		if m.Usage, ok = scanUsage(usage); !ok {
			return m, false
		}
	}
	return m, true
}

var usageProjectionFields = [...]string{"input", "output", "cacheRead", "cacheWrite", "cacheWrite1h", "reasoning", "totalTokens", "cost"}

var usageCostFields = [...]string{"input", "output", "cacheRead", "cacheWrite", "total"}

// scanUsage is ai.ProjectUsage for a usage object of numbers (or nulls) and
// a cost object of the same, without its lossless decode.
func scanUsage(value []byte) (*protocol.Usage, bool) {
	if value[0] != '{' {
		return nil, false
	}
	var parts [len(usageProjectionFields)][]byte
	if !collectMembers(value, usageProjectionFields[:], parts[:]) {
		return nil, false
	}
	var numbers [len(usageProjectionFields) - 1]float64
	for i := range numbers {
		var ok bool
		if parts[i] != nil {
			if numbers[i], ok = plainFloat(parts[i]); !ok {
				return nil, false
			}
		}
	}
	usage := ai.Usage{Input: numbers[0], Output: numbers[1], CacheRead: numbers[2], CacheWrite: numbers[3], TotalTokens: numbers[6]}
	// A null subset decodes as nil, which projects as 0 like an absent one.
	if parts[4] != nil {
		usage.CacheWrite1h = &numbers[4]
	}
	if parts[5] != nil {
		usage.Reasoning = &numbers[5]
	}
	cost := parts[7]
	if cost != nil {
		if cost[0] != '{' {
			return nil, false
		}
		var costParts [len(usageCostFields)][]byte
		if !collectMembers(cost, usageCostFields[:], costParts[:]) {
			return nil, false
		}
		for _, part := range costParts {
			if part != nil {
				if _, ok := plainFloat(part); !ok {
					return nil, false
				}
			}
		}
		if costParts[4] != nil {
			usage.Cost.Total, _ = plainFloat(costParts[4])
		}
	}
	projected := usage.Protocol()
	projected.CostUnpriced = cost == nil
	return &projected, true
}

// collectMembers stores the value of each member of the validated object raw
// named in names at its index in values. It is false when a member repeats or
// another key might match one of names under encoding/json.
func collectMembers(raw []byte, names []string, values [][]byte) bool {
	ok := true
	eachMember(raw, func(key, value []byte) bool {
		index := fieldIndex(key, names)
		switch {
		case index == -1:
			return true
		case index < 0 || values[index] != nil:
			ok = false
			return false
		}
		values[index] = value
		return true
	})
	return ok
}

// plainText is a string member as encoding/json decodes it into a string,
// when that needs no decoding: absent and null are "".
func plainText(value []byte) (string, bool) {
	if value == nil || isNull(value) {
		return "", true
	}
	body, ok := plainString(value)
	if !ok {
		return "", false
	}
	return internString(body), true
}

// scanContent is ai.ProjectContent for valid JSON content.
func scanContent(value []byte) (text string, parts []protocol.ContentPart, calls []protocol.ToolCall, ok bool) {
	switch {
	case value == nil || isNull(value):
		return "", nil, nil, true
	case value[0] == '"':
		body, ok := plainString(value)
		return string(body), nil, nil, ok
	case value[0] != '[':
		return "", nil, nil, false
	}
	ok = true
	var texts [][]byte
	size := 0
	eachElement(value, func(block []byte) bool {
		if isNull(block) {
			return true // decodes as a block of no type
		}
		if block[0] != '{' {
			ok = false
			return false
		}
		var f [len(projectBlockFields)][]byte
		if !collectMembers(block, projectBlockFields[:], f[:]) {
			ok = false
			return false
		}
		// Every string member decodes, whatever the block's type.
		var s [6][]byte
		for i := range s {
			if v := f[i]; v != nil && !isNull(v) {
				if s[i], ok = plainString(v); !ok {
					return false
				}
			}
		}
		switch string(s[0]) {
		case "text":
			texts = append(texts, s[1])
			size += len(s[1])
		case "image":
			parts = append(parts, protocol.ContentPart{Type: protocol.ContentPartImage, MIMEType: string(s[3]), Data: string(s[2])})
		case "toolCall":
			calls = append(calls, protocol.ToolCall{ID: string(s[4]), Name: string(s[5]), Arguments: string(f[6])})
		}
		return true
	})
	if !ok {
		return "", nil, nil, false
	}
	if len(texts) == 1 {
		return string(texts[0]), parts, calls, true
	}
	joined := make([]byte, 0, size)
	for _, t := range texts {
		joined = append(joined, t...)
	}
	return string(joined), parts, calls, true
}
