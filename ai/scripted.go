package ai

import (
	"context"
	"encoding/json"
	"sync"
)

// ScriptedStream emits a sequence of native assistant messages for offline
// demos and tests. Each invocation owns its script; concurrent runs must create
// separate streams. Exhaustion returns an empty completed answer.
func ScriptedStream(messages ...Message) StreamFn {
	var mu sync.Mutex
	next := 0
	return func(ctx context.Context, _ json.RawMessage, _ TranscriptContext, _ map[string]any) (*AssistantMessageEventStream, error) {
		mu.Lock()
		message := Message{Role: "assistant", StopReason: "stop", Content: BlockContent()}
		if next < len(messages) {
			// Do not expose the script to mutable live stream events.
			raw, err := json.Marshal(messages[next])
			if err != nil {
				mu.Unlock()
				return nil, err
			}
			if err := json.Unmarshal(raw, &message); err != nil {
				mu.Unlock()
				return nil, err
			}
			next++
		}
		mu.Unlock()
		out := NewAssistantMessageEventStreamFor(ctx)
		out.Push(AssistantMessageEvent{Type: "start", Partial: &message})
		for i, block := range message.Content.Blocks.Values() {
			if block.Type == "text" {
				out.Push(AssistantMessageEvent{Type: "text_delta", ContentIndex: i, Delta: block.Text, Partial: &message})
			}
		}
		if message.StopReason == "error" || message.StopReason == "aborted" {
			out.Push(AssistantMessageEvent{Type: "error", Reason: message.StopReason, Error: &message})
		} else {
			out.Push(AssistantMessageEvent{Type: "done", Reason: message.StopReason, Message: &message})
		}
		out.End()
		return out, nil
	}
}
