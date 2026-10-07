package ai_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/ai/protocol"
)

// A public consumer can compose fallback with the native engine without any
// application runtime, legacy chat adapter or replacement tool loop.
func TestFallbackProviderRunsNativeEngineToolCycle(t *testing.T) {
	ctx := context.Background()
	primary, secondary, executions := 0, 0, 0
	failed := &ai.Message{Role: "assistant", StopReason: "error", Usage: &ai.Usage{Input: 1}}
	final := &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})}
	provider := ai.FallbackProvider{Retry: protocol.RetryPolicy{MaxAttempts: 1}, Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"primary"}`), Stream: func(ctx context.Context, _ json.RawMessage, _ ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
			primary++
			stream := ai.NewAssistantMessageEventStreamFor(ctx)
			stream.Push(ai.AssistantMessageEvent{Type: "start", Partial: failed})
			stream.Push(ai.AssistantMessageEvent{Type: "error", Error: failed})
			stream.End()
			return stream, nil
		}},
		{Model: json.RawMessage(`{"id":"secondary"}`), Stream: func(ctx context.Context, _ json.RawMessage, transcript ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
			secondary++
			message := final
			if secondary == 1 {
				message = &ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "call-1", Name: "remember", Arguments: json.RawMessage(`{"value":"fact"}`)})}
			} else {
				found := false
				for _, message := range transcript.Messages() {
					if message.Role == "toolResult" && message.ToolCallID == "call-1" {
						found = true
					}
				}
				if !found {
					t.Error("engine lost native tool result")
				}
			}
			stream := ai.NewAssistantMessageEventStreamFor(ctx)
			stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: message.StopReason, Message: message})
			stream.End()
			return stream, nil
		}},
	}}
	tool := &engine.Tool{Tool: ai.Tool{Name: "remember", Description: "remember a fact", Parameters: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}`)}, Execute: func(_ context.Context, id string, args any, _ func(*engine.ToolResult)) (*engine.ToolResult, error) {
		executions++
		if id != "call-1" {
			t.Error("call identity changed", id)
		}
		return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "saved"})}, nil
	}}
	history := engine.NewList[*ai.Message]()
	messages, err := engine.Run(ctx, engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("remember and finish")}), engine.Context{Messages: history, Tools: engine.NewList(tool)}, engine.Config{Model: json.RawMessage(`{"id":"primary"}`), ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }}, func(event engine.Event) error {
		if event.Type == "message_end" && event.Message == failed {
			t.Error("discarded provider failure entered native transcript")
		}
		return nil
	}, provider.Stream)
	if err != nil {
		t.Fatal(err)
	}
	if primary != 2 || secondary != 2 || executions != 1 {
		t.Fatal(primary, secondary, executions)
	}
	if messages.Len() == 0 || messages.Get(messages.Len()-1) != final {
		t.Fatal("native final message identity lost")
	}
}
