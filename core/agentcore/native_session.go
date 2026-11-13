package agentcore

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	nativehost "github.com/2found/2ai/core/agentcore/host"
	"github.com/2found/2ai/core/ai"
)

// RecoverNativeTranscript refuses ambiguous effects even when cancellation caused Pi to
// emit a synthetic error result. A native error event is not proof an external
// write didn't happen. Entries in this format never go through RecoverSession.
// This restores provider state only. The consumer still validates workflow
// metadata (questions, delegation, invocation and goals) before running it.
func RecoverNativeTranscript(entries []SessionEntry) (json.RawMessage, error) {
	state := map[string]json.RawMessage{}
	var messages []json.RawMessage
	var pendingResult json.RawMessage
	unsettled := map[string]string{}
	seenState := false
	var selection *ai.FallbackSelection
	for _, entry := range entries {
		switch entry.Kind {
		case EntryNativeModelSelection:
			if !seenState {
				return nil, errors.New("native ladder selection precedes initial state")
			}
			next, err := ai.ParseFallbackSelection(entry.Content, selection)
			if err != nil {
				return nil, err
			}
			selection = &next
			state["model"] = append(json.RawMessage(nil), next.Model...)
		case EntryNativeContextSummary:
			if _, err := nativehost.ParseSummary(entry.Content); err != nil {
				return nil, err
			}
		case EntryNativeAnswer, EntryNativeDelegation, EntryNativeDelegationBatch, EntryNativeGoal, EntryNativeGoalRevision, EntryNativeInvocation, EntryNativeChildResult:
			// Workflow metadata belongs to the consumer. It does not rewrite
			// any provider message during transcript recovery.
		case EntryNativeEffectStart, EntryNativeEffectDone:
			var effect struct {
				ID string `json:"effectId"`
			}
			if err := json.Unmarshal([]byte(entry.Content), &effect); err != nil || effect.ID == "" {
				return nil, errors.New("corrupt native effect record")
			}
			if entry.Kind == EntryNativeEffectStart {
				unsettled[effect.ID] = entry.CallID
			} else {
				delete(unsettled, effect.ID)
			}
		case EntryNativeState:
			state = map[string]json.RawMessage{}
			pendingResult = nil
			if err := json.Unmarshal([]byte(entry.Content), &state); err != nil || state == nil {
				return nil, errors.New("corrupt native state record")
			}
			if err := json.Unmarshal(state["messages"], &messages); err != nil {
				return nil, err
			}
			if selection != nil && !nativehost.SameJSON(state["model"], selection.Model) {
				return nil, errors.New("native checkpoint disagrees with committed native ladder model")
			}
			seenState = true
		case EntryNativeEvent:
			var event struct {
				Type    string          `json:"type"`
				Message json.RawMessage `json:"message"`
			}
			if err := json.Unmarshal([]byte(entry.Content), &event); err != nil {
				return nil, err
			}
			if event.Type == "message_start" {
				var message struct{ Role string }
				if err := json.Unmarshal(event.Message, &message); err != nil {
					return nil, err
				}
				if message.Role == "toolResult" {
					if pendingResult != nil {
						return nil, errors.New("overlapping native tool-result messages")
					}
					// Pi emits the complete immutable result at message_start,
					// including its native timestamp. Never invent that message
					// from a physical effect receipt or tool_execution_end.
					pendingResult = event.Message
				}
			}
			if event.Type == "message_end" {
				if !json.Valid(event.Message) {
					return nil, errors.New("corrupt native message record")
				}
				if pendingResult != nil {
					if string(pendingResult) != string(event.Message) {
						return nil, errors.New("native tool-result start/end mismatch")
					}
					pendingResult = nil
				}
				messages = append(messages, event.Message)
			}
		default:
			return nil, fmt.Errorf("session kind %q is not a native record", entry.Kind)
		}
	}
	if !seenState {
		return nil, errors.New("native session has no initial state")
	}
	if len(unsettled) > 0 {
		ids := make([]string, 0, len(unsettled))
		for id, callID := range unsettled {
			ids = append(ids, callID+"/"+id)
		}
		slices.Sort(ids)
		return nil, fmt.Errorf("%w: %v", nativehost.ErrUnsettledEffect, ids)
	}
	if pendingResult != nil {
		messages = append(messages, pendingResult)
	}
	messagesRaw, _ := json.Marshal(messages)
	if err := nativehost.ValidateMessages(messagesRaw); err != nil {
		return nil, err
	}
	state["messages"] = messagesRaw
	for name := range state {
		if name != "messages" && name != "model" && name != "thinkingLevel" && name != "tools" {
			delete(state, name)
		}
	}
	raw, err := json.Marshal(state)
	if err != nil {
		return nil, err
	}
	return raw, nil
}
