package host

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/2found/2ai/core/ai/protocol"
)

// These are newly authored host instructions, never provider/history messages.
// InputMessages converts newly authored host input only; provider history remains native.
func InputMessages(messages []protocol.Message) ([]json.RawMessage, error) {
	result := make([]json.RawMessage, 0, len(messages))
	for _, message := range messages {
		if message.Role != protocol.RoleSystem && message.Role != protocol.RoleUser {
			return nil, fmt.Errorf("host injection cannot impersonate role %q", message.Role)
		}
		if len(message.ToolCalls) > 0 || message.ToolCallID != "" {
			return nil, errors.New("host injection cannot contain tool calls/results")
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
