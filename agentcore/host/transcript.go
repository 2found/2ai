package host

import (
	"encoding/json"
	"errors"
	"fmt"
)

// ErrUnsettledEffect marks a missing physical effect receipt or native tool result.
var ErrUnsettledEffect = errors.New("Pi session has an unsettled tool effect")

// ValidateMessages checks completed native tool batches without projecting or
// changing messages. Provider error/aborted partial calls require no results.
func ValidateMessages(raw json.RawMessage) error {
	var messages []json.RawMessage
	if err := json.Unmarshal(raw, &messages); err != nil {
		return err
	}
	return validateMessages(messages)
}

func validateMessages(messages []json.RawMessage) error {
	// Results must finish their own assistant batch. A reused provider call ID
	// in a later batch cannot settle an earlier missing result.
	open := map[string]int{}
	for _, raw := range messages {
		var m struct {
			Role, ToolCallID, StopReason string
			Content                      json.RawMessage
		}
		if err := json.Unmarshal(raw, &m); err != nil {
			return err
		}
		if m.Role != "toolResult" && len(open) > 0 {
			return fmt.Errorf("%w: incomplete tool batch", ErrUnsettledEffect)
		}
		// Pi exits the turn before tool admission on error/aborted messages.
		// Their partial calls remain in history but have no native results to
		// await. Physical effects are checked independently above; this cannot
		// clear an effect that started before a provider failure. A length stop
		// still produces failure results, so it must finish its result batch.
		if m.Role == "assistant" && m.StopReason != "error" && m.StopReason != "aborted" {
			var blocks []struct{ Type, ID string }
			if err := json.Unmarshal(m.Content, &blocks); err != nil {
				return err
			}
			for _, block := range blocks {
				if block.Type == "toolCall" {
					open[block.ID]++
				}
			}
		}
		if m.Role == "toolResult" {
			if open[m.ToolCallID] > 1 {
				open[m.ToolCallID]--
			} else {
				delete(open, m.ToolCallID)
			}
		}
	}
	if len(open) > 0 {
		return fmt.Errorf("%w: tool results need recovery", ErrUnsettledEffect)
	}
	return nil
}
