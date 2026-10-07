package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

func argumentRef(raw string) any {
	value, err := jsonjs.DecodeValue([]byte(raw))
	if err != nil {
		panic(err)
	}
	return value
}

func argumentJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := jsonjs.MarshalValue(value)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = jsonjs.StringifyJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestPiArgumentReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-argument-references.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Shape, Phase string }
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 72 {
		t.Fatal("unexpected argument reference coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Mode+"/"+tc.Input.Shape+"/"+tc.Input.Phase, func(t *testing.T) {
			mode, shape, phase := tc.Input.Mode, tc.Input.Shape, tc.Input.Phase
			snapshots := []json.RawMessage{}
			capture := func(stage string, args any) {
				encoded, err := json.Marshal(map[string]any{"stage": stage, "args": args})
				if err != nil {
					t.Fatal(err)
				}
				snapshots = append(snapshots, encoded)
			}
			initial := `{"value":"original"}`
			if shape == "array" {
				initial = `["original"]`
			} else if shape == "number" {
				initial = `7`
			}
			replacement := func() any {
				if shape == "array" {
					return argumentRef(`["replacement"]`)
				}
				if shape == "number" {
					return argumentRef(`99`)
				}
				return argumentRef(`{"replacement":true}`)
			}
			mutate := func(value any) {
				switch v := value.(type) {
				case *engine.Array:
					v.Append("changed")
				case *engine.Object:
					v.Set("changed", true)
				default:
					t.Fatalf("cannot mutate %T", value)
				}
			}
			var beforeContext *engine.BeforeToolCall
			var selected, beforeLocal, executed, executeLocal, afterLocal any
			hooks := engine.ToolHooks{
				Before: func(_ context.Context, value *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
					beforeContext = value
					selected = value.Args
					capture("before", value.Args)
					if phase == "before_mutate" || phase == "before_mutate_replace" {
						mutate(value.Args)
					}
					if phase == "before_replace" || phase == "before_mutate_replace" || phase == "before_replace_mutate" {
						value.Args = replacement()
					}
					if phase == "before_null" {
						value.Args = nil
					}
					if phase == "before_replace_mutate" {
						mutate(value.Args)
					}
					beforeLocal = value.Args
					return nil, nil
				},
				After: func(_ context.Context, value engine.AfterToolCall) (*engine.AfterToolResult, error) {
					capture("after", value.Args)
					if phase == "after_mutate" {
						mutate(value.Args)
					}
					if phase == "after_replace" {
						value.Args = replacement()
					}
					afterLocal = value.Args
					return nil, nil
				},
			}
			update := func() {
				if phase == "update_mutate" || phase == "late_before_mutate" {
					mutate(beforeContext.Args)
				}
				if phase == "late_before_replace" {
					beforeContext.Args = replacement()
				}
			}
			terminate := true
			tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Description: "echo", Parameters: json.RawMessage(`{"type":"` + shape + `"}`)}, Label: "echo", Execute: func(_ context.Context, _ string, value any, _update func(*engine.ToolResult)) (*engine.ToolResult, error) {
				executed = value
				capture("execute", value)
				if phase == "execute_mutate" {
					mutate(value)
				}
				if phase == "execute_replace" {
					value = replacement()
				}
				executeLocal = value
				_update(&engine.ToolResult{Content: ai.NewBlockList(), Details: argumentRef(`{}`)})
				capture("execute_after_update", value)
				return &engine.ToolResult{Content: ai.NewBlockList(), Details: argumentRef(`{}`), Terminate: &terminate}, nil
			}}
			call := &ai.ContentBlock{Type: "toolCall", Name: "echo", ID: "first", Arguments: json.RawMessage(initial)}
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockReferences(call), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: "toolUse", Timestamp: 1}
			if mode == "programmatic" {
				outcome, err := engine.RunToolCall(context.Background(), call, engine.NewList([]*engine.Tool{tool}...), assistant, &engine.Context{}, hooks, func(*engine.ToolResult) error { update(); return nil })
				if err != nil || outcome.IsError {
					t.Fatalf("tool failed: %+v %v", outcome, err)
				}
			} else {
				stream := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
					s := ai.NewAssistantMessageEventStream()
					s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "toolUse", Message: assistant})
					return s, nil
				}
				_, err := engine.Run(context.Background(), nil, engine.Context{Messages: engine.NewList([]*ai.Message{}...), Tools: engine.NewList([]*engine.Tool{tool}...)}, engine.Config{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }, ToolExecution: mode, ToolHooks: hooks, FinishTurn: func(context.Context, *engine.Turn) (string, error) { return "end", nil }}, func(event engine.Event) error {
					if event.Type == "tool_execution_update" {
						update()
					}
					return nil
				}, stream)
				if err != nil {
					t.Fatal(err)
				}
			}
			actual, err := json.Marshal(map[string]any{"snapshots": snapshots, "selected": selected, "beforeLocal": beforeLocal, "executed": executed, "executeLocal": executeLocal, "afterLocal": afterLocal, "lateBefore": beforeContext.Args, "raw": call.Arguments, "same": selected == executed})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			decoder := json.NewDecoder(strings.NewReader(string(actual)))
			decoder.UseNumber()
			if err = decoder.Decode(&got); err != nil {
				t.Fatal(err)
			}
			decoder = json.NewDecoder(strings.NewReader(string(tc.Expected)))
			decoder.UseNumber()
			if err = decoder.Decode(&want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
