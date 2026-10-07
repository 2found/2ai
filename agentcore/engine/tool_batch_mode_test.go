package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiToolBatchMode(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-tool-batch-mode.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Mode, Initial, Replacement, Phase string
			Expected                          json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 144 {
		t.Fatal("unexpected batch mode oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Mode+"/"+tc.Initial+"/"+tc.Replacement+"/"+tc.Phase, func(t *testing.T) {
			assistant := &ai.Message{Role: "assistant", StopReason: tc.Initial, Content: ai.BlockContent(
				ai.ContentBlock{Type: "toolCall", ID: "one", Name: "op", Arguments: json.RawMessage(`{}`)},
				ai.ContentBlock{Type: "toolCall", ID: "two", Name: "op", Arguments: json.RawMessage(`{}`)},
			), API: "test", Provider: "test", Model: "m", Usage: &ai.Usage{}, Timestamp: 123}
			var mu sync.Mutex
			executed, starts := []string{}, []string{}
			results := []map[string]any{}
			mutate := func(phase, id string) {
				if phase == tc.Phase && id == "one" {
					assistant.StopReason = tc.Replacement
				}
			}
			tool := &engine.Tool{Tool: ai.Tool{Name: "op", Description: "op", Parameters: json.RawMessage(`{"type":"object"}`)}}
			tool.Execute = func(_ context.Context, id string, _ any, _ func(*engine.ToolResult)) (*engine.ToolResult, error) {
				mu.Lock()
				mutate("execute", id)
				executed = append(executed, id)
				mu.Unlock()
				return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "ran " + id}), Details: engine.NewObject()}, nil
			}
			config := engine.Config{Model: json.RawMessage(`{"api":"test","provider":"test","id":"m"}`), ToolExecution: tc.Mode, Now: func() int64 { return 123 }}
			config.ConvertToLLM = func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }
			config.Before = func(_ context.Context, call *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
				mu.Lock()
				defer mu.Unlock()
				mutate("before", call.ToolCall.ID)
				return nil, nil
			}
			config.After = func(_ context.Context, call engine.AfterToolCall) (*engine.AfterToolResult, error) {
				mu.Lock()
				defer mu.Unlock()
				mutate("after", call.ToolCall.ID)
				return nil, nil
			}
			config.FinishTurn = func(context.Context, *engine.Turn) (string, error) { return "end", nil }
			messages, err := engine.Run(context.Background(), engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("run"), Timestamp: 123}), engine.Context{Messages: engine.NewList[*ai.Message](), Tools: engine.NewList(tool)}, config, func(event engine.Event) error {
				mu.Lock()
				defer mu.Unlock()
				switch event.Type {
				case "tool_execution_start":
					starts = append(starts, event.ToolCallID)
					mutate("start", event.ToolCallID)
				case "tool_execution_end":
					mutate("end", event.ToolCallID)
				case "message_end":
					if event.Message.Role == "assistant" {
						// Mutate the loop-owned message retained by an event listener.
						assistant = event.Message
					}
					if event.Message.Role == "toolResult" {
						results = append(results, map[string]any{"id": event.Message.ToolCallID, "isError": event.Message.IsError, "text": event.Message.Content.Blocks.Get(0).Text})
						mutate("result", event.Message.ToolCallID)
					}
				}
				return nil
			}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				stream := ai.NewAssistantMessageEventStream()
				stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: tc.Initial, Message: assistant})
				stream.End()
				return stream, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			resultIDs := []string{}
			for _, message := range messages.Values() {
				if message.Role == "toolResult" {
					resultIDs = append(resultIDs, message.ToolCallID)
				}
			}
			// Parallel completion order is unconstrained; execution membership and
			// the ordered tool-result transcript must still match the source.
			slices.Sort(executed)
			actual, err := json.Marshal(map[string]any{"executed": executed, "starts": starts, "results": results, "stopReason": assistant.StopReason, "resultIDs": resultIDs})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
