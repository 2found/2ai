package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiToolReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-tool-references.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Phase string }
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	if fixtures.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixtures.Cases) != 53 {
		t.Fatal("unexpected tool reference oracle coverage")
	}
	for _, fixture := range fixtures.Cases {
		t.Run(fixture.Input.Mode+"/"+fixture.Input.Phase, func(t *testing.T) {
			input := fixture.Input
			events, hooks, executed, updates := []json.RawMessage{}, []json.RawMessage{}, []json.RawMessage{}, []json.RawMessage{}
			capture := func(value any) json.RawMessage {
				raw, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				return raw
			}
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{"value":"original"}`)}), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: "toolUse", Timestamp: 1}
			if strings.HasPrefix(input.Phase, "next_") {
				assistant.Content.Blocks.Append(&ai.ContentBlock{Type: "toolCall", ID: "second", Name: "echo", Arguments: json.RawMessage(`{"value":"second"}`)})
			}
			retained := assistant.Content.Blocks.Get(0)
			firstDone := make(chan struct{})
			mutate := func() {
				retained.ID = "changed"
				retained.Name = "renamed"
				retained.Arguments = json.RawMessage(`{"value":"replacement"}`)
			}
			editContent := func(at string) {
				if input.Phase == at+"_slot" {
					assistant.Content.Blocks.Set(0, &ai.ContentBlock{Type: "toolCall", ID: "slot", Name: "renamed", Arguments: json.RawMessage(`{"value":"slot"}`)})
				}
				if input.Phase == at+"_grow" {
					for range 8 {
						assistant.Content.Blocks.Append(&ai.ContentBlock{Type: "text", Text: "padding"})
					}
				}
			}
			terminate := true
			tools := []*engine.Tool{}
			for _, name := range []string{"echo", "renamed"} {
				tools = append(tools, &engine.Tool{Tool: ai.Tool{Name: name, Description: name, Parameters: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}`)}, Label: name, Execute: func(_ context.Context, id string, args any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
					if id == "second" && input.Mode == "parallel" {
						<-firstDone
					}
					executed = append(executed, capture(map[string]any{"tool": name, "id": id, "args": args}))
					if input.Phase == "execute" {
						mutate()
					}
					update(&engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "partial"}), Details: argumentRef(`{}`)})
					return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "result"}), Details: argumentRef(`{}`), Terminate: &terminate}, nil
				}})
			}
			toolHooks := engine.ToolHooks{
				Before: func(_ context.Context, hook *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
					hooks = append(hooks, capture(map[string]any{"hook": "before", "identity": hook.ToolCall == hook.AssistantMessage.Content.Blocks.Get(0), "call": hook.ToolCall, "args": hook.Args}))
					if hook.ToolCall.ID != "second" {
						retained = hook.ToolCall
					}
					if input.Phase == "next_before" && hook.ToolCall.ID == "second" {
						mutate()
					}
					editContent("before")
					if strings.HasPrefix(input.Phase, "before") {
						mutate()
					}
					if input.Phase == "before_replace" {
						hook.ToolCall = &ai.ContentBlock{Type: "toolCall", ID: "ignored", Name: "ignored", Arguments: json.RawMessage(`{}`)}
					}
					if input.Phase == "before_failure" {
						return nil, errors.New("before failed")
					}
					if input.Phase == "before_block" {
						return &engine.BeforeToolResult{Block: true, Reason: "blocked", Terminate: true}, nil
					}
					return nil, nil
				},
				After: func(_ context.Context, hook engine.AfterToolCall) (*engine.AfterToolResult, error) {
					hooks = append(hooks, capture(map[string]any{"hook": "after", "identity": hook.ToolCall == hook.AssistantMessage.Content.Blocks.Get(0), "call": hook.ToolCall, "args": hook.Args}))
					editContent("after")
					if strings.HasPrefix(input.Phase, "after") {
						mutate()
					}
					if input.Phase == "next_after" && hook.ToolCall.ID == "second" {
						mutate()
					}
					if input.Phase == "after_replace" {
						hook.ToolCall = &ai.ContentBlock{Type: "toolCall", ID: "ignored", Name: "ignored", Arguments: json.RawMessage(`{}`)}
					}
					if input.Phase == "after_failure" {
						return nil, errors.New("after failed")
					}
					return nil, nil
				},
			}
			var outcome engine.ToolOutcome
			var messages *engine.MessageList
			var err error
			if input.Mode == "programmatic" {
				outcome, err = engine.RunToolCall(context.Background(), retained, engine.NewList(tools...), assistant, &engine.Context{Messages: engine.NewList([]*ai.Message{assistant}...), Tools: engine.NewList(tools...)}, toolHooks, func(partial *engine.ToolResult) error {
					updates = append(updates, capture(partial))
					if input.Phase == "update" {
						mutate()
					}
					return nil
				})
			} else {
				stream := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
					s := ai.NewAssistantMessageEventStream()
					s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "toolUse", Message: assistant})
					return s, nil
				}
				config := engine.Config{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), Now: func() int64 { return 1000 }, ToolExecution: input.Mode, ToolHooks: toolHooks, ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }, FinishTurn: func(context.Context, *engine.Turn) (string, error) { return "end", nil }}
				messages, err = engine.Run(context.Background(), engine.NewList([]*ai.Message{{Role: "user", Content: ai.TextContent("go")}}...), engine.Context{Messages: engine.NewList([]*ai.Message{}...), Tools: engine.NewList(tools...)}, config, func(event engine.Event) error {
					events = append(events, capture(event))
					if event.Type == "message_end" && event.Message.Role == "assistant" {
						assistant = event.Message
						retained = assistant.Content.Blocks.Get(0)
					}
					if input.Phase == "start" && event.Type == "tool_execution_start" || input.Phase == "update" && event.Type == "tool_execution_update" || input.Phase == "end" && event.Type == "tool_execution_end" || input.Phase == "message_start" && event.Type == "message_start" && event.Message.Role == "toolResult" {
						mutate()
					}
					if input.Phase == "next_start" && event.Type == "tool_execution_start" && event.ToolCallID == "second" || input.Phase == "next_result" && event.Type == "message_start" && event.Message.Role == "toolResult" && event.Message.ToolCallID == "second" {
						mutate()
					}
					if event.Type == "tool_execution_end" && event.ToolCallID != "second" {
						close(firstDone)
					}
					return nil
				}, stream)
			}
			if err != nil {
				t.Fatal(err)
			}
			result := map[string]any{"events": events, "hooks": hooks, "executed": executed, "updates": updates, "assistant": assistant, "retained": retained}
			if input.Mode == "programmatic" {
				result["outcome"] = outcome
			} else {
				result["messages"] = messages
			}
			actual := capture(result)
			var got, want any
			if err = json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(fixture.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, fixture.Expected)
			}
		})
	}
}
