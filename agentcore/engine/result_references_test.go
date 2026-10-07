package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiResultReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-result-references.json")
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
	if fixtures.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixtures.Cases) != 37 {
		t.Fatal("unexpected result reference coverage")
	}
	for _, fixture := range fixtures.Cases {
		t.Run(fixture.Input.Mode+"/"+fixture.Input.Phase, func(t *testing.T) {
			input := fixture.Input
			events, hooks, updateSnapshots := []json.RawMessage{}, []json.RawMessage{}, []json.RawMessage{}
			liveUpdates := []*engine.ToolResult{}
			identities := []map[string]bool{}
			capture := func(value any) json.RawMessage {
				raw, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				return raw
			}
			makeResult := func(text string) *engine.ToolResult {
				terminate := true
				return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: text}), Details: argumentRef(`{}`), Terminate: &terminate}
			}
			original, partial := makeResult("original"), makeResult("partial")
			if input.Phase == "update_nil" {
				partial = nil
			}
			produced := original
			var endResult, afterResult *engine.ToolResult
			mutate := func(value *engine.ToolResult) {
				value.Content = ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "mutated"})
				value.Details = argumentRef(`{"changed":true}`)
			}
			onUpdate := func(value *engine.ToolResult) {
				identities = append(identities, map[string]bool{"update": value == partial})
				updateSnapshots = append(updateSnapshots, capture(value))
				liveUpdates = append(liveUpdates, value)
				if input.Phase == "update_mutate" {
					mutate(value)
				}
			}
			tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Description: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}, Label: "echo", Execute: func(_ context.Context, _ string, _ any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
				update(partial)
				if input.Phase == "update_later" {
					mutate(partial)
				}
				if input.Phase == "update_returned" {
					produced = partial
				} else if input.Phase == "nil_result" {
					produced = nil
				} else {
					produced = original
				}
				return produced, nil
			}}
			after := func(_ context.Context, value engine.AfterToolCall) (*engine.AfterToolResult, error) {
				afterResult = value.Result
				identities = append(identities, map[string]bool{"after": value.Result == produced})
				hooks = append(hooks, capture(map[string]any{"result": value.Result, "isError": value.IsError}))
				if input.Phase == "after_mutate" || input.Phase == "after_failure" {
					mutate(value.Result)
				}
				if input.Phase == "after_failure" {
					return nil, errors.New("after failed")
				}
				if input.Phase == "after_override" {
					return &engine.AfterToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "override"})}, nil
				}
				if input.Phase == "after_same_override" {
					return value.Result, nil
				}
				if input.Phase == "after_replace" {
					value.Result = makeResult("ignored")
				}
				return nil, nil
			}
			call := &ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)}
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockReferences(call), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: "toolUse", Timestamp: 1}
			var outcome engine.ToolOutcome
			var messages *engine.MessageList
			var err error
			if input.Mode == "programmatic" {
				outcome, err = engine.RunToolCall(context.Background(), call, engine.NewList([]*engine.Tool{tool}...), assistant, &engine.Context{Messages: engine.NewList([]*ai.Message{}...), Tools: engine.NewList([]*engine.Tool{tool}...)}, engine.ToolHooks{After: after}, func(partial *engine.ToolResult) error { onUpdate(partial); return nil })
				identities = append(identities, map[string]bool{"outcome": outcome.Result == produced})
			} else {
				config := engine.Config{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), Now: func() int64 { return 1000 }, ToolExecution: input.Mode, ToolHooks: engine.ToolHooks{After: after}, ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }, FinishTurn: func(context.Context, *engine.Turn) (string, error) { return "end", nil }}
				messages, err = engine.Run(context.Background(), engine.NewList([]*ai.Message{{Role: "user", Content: ai.TextContent("go")}}...), engine.Context{Messages: engine.NewList([]*ai.Message{}...), Tools: engine.NewList([]*engine.Tool{tool}...)}, config, func(event engine.Event) error {
					events = append(events, capture(event))
					if event.Type == "tool_execution_update" {
						onUpdate(event.PartialResult)
					}
					if event.Type == "tool_execution_end" {
						endResult = event.Result
						identities = append(identities, map[string]bool{"end": event.Result == produced})
						if input.Phase == "end_mutate" {
							mutate(event.Result)
						}
						if input.Phase == "end_original" {
							mutate(produced)
						}
					}
					return nil
				}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
					s := ai.NewAssistantMessageEventStream()
					s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "toolUse", Message: assistant})
					return s, nil
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			if produced != nil {
				produced.Details = argumentRef(`{"post":true}`)
			}
			result := map[string]any{"events": events, "hooks": hooks, "updateSnapshots": updateSnapshots, "liveUpdates": liveUpdates, "identities": identities, "original": original, "partial": partial, "produced": produced, "afterResult": afterResult}
			if endResult != nil {
				result["endResult"] = endResult
			}
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
