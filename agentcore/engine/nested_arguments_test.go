package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiNestedArguments(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-nested-arguments.json")
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
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 54 {
		t.Fatal("unexpected nested argument coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Mode+"/"+tc.Input.Shape+"/"+tc.Input.Phase, func(t *testing.T) {
			mode, shape, phase := tc.Input.Mode, tc.Input.Shape, tc.Input.Phase
			var selected *engine.Object
			var retained, executed, after any
			snapshots := []json.RawMessage{}
			capture := func(stage string, args any) {
				raw, err := json.Marshal(map[string]any{"stage": stage, "args": args, "retained": retained})
				if err != nil {
					t.Fatal(err)
				}
				snapshots = append(snapshots, raw)
			}
			branch := func(value string) any {
				child := engine.NewObject(engine.Property{Name: "value", Value: value})
				if shape == "array" {
					return engine.NewArray(child)
				}
				return engine.NewObject(engine.Property{Name: "child", Value: child})
			}
			edit := func(value any) {
				if shape == "array" {
					a := value.(*engine.Array)
					a.Get(0).(*engine.Object).Set("value", "changed")
					a.Append(engine.NewObject(engine.Property{Name: "value", Value: "appended"}))
				} else {
					o := value.(*engine.Object)
					o.Get("child").(*engine.Object).Set("value", "changed")
					o.Set("added", true)
				}
			}
			hooks := engine.ToolHooks{
				Before: func(_ context.Context, call *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
					selected = call.Args.(*engine.Object)
					retained = selected.Get("branch")
					if phase == "before_edit" {
						edit(retained)
					}
					if phase == "before_replace" || phase == "replace_then_update" {
						selected.Set("branch", branch("replacement"))
					}
					capture("before", call.Args)
					return nil, nil
				},
				After: func(_ context.Context, call engine.AfterToolCall) (*engine.AfterToolResult, error) {
					args := call.Args.(*engine.Object)
					after = args.Get("branch")
					if phase == "after_edit" {
						edit(retained)
					}
					if phase == "after_replace" {
						args.Set("branch", branch("replacement"))
					}
					capture("after", args)
					return nil, nil
				},
			}
			update := func() {
				if phase == "update_edit" || phase == "replace_then_update" {
					edit(retained)
				}
			}
			terminate := true
			tool := &engine.Tool{
				Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)},
				Execute: func(_ context.Context, _ string, value any, onUpdate func(*engine.ToolResult)) (*engine.ToolResult, error) {
					args := value.(*engine.Object)
					executed = args.Get("branch")
					if phase == "execute_edit" {
						edit(retained)
					}
					if phase == "execute_replace" {
						args.Set("branch", branch("replacement"))
					}
					onUpdate(&engine.ToolResult{Content: ai.NewBlockList(), Details: argumentRef(`{}`)})
					capture("execute", args)
					return &engine.ToolResult{Content: ai.NewBlockList(), Details: argumentRef(`{}`), Terminate: &terminate}, nil
				},
			}
			initial, err := json.Marshal(engine.NewObject(engine.Property{Name: "branch", Value: branch("original")}))
			if err != nil {
				t.Fatal(err)
			}
			call := &ai.ContentBlock{Type: "toolCall", ID: "first", Name: "echo", Arguments: initial}
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
				_, err := engine.Run(context.Background(), nil, engine.Context{Messages: engine.NewList([]*ai.Message{}...), Tools: engine.NewList([]*engine.Tool{tool}...)}, engine.Config{
					Model:         json.RawMessage(`{"id":"test","api":"test","provider":"test"}`),
					ConvertToLLM:  func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil },
					ToolExecution: mode, ToolHooks: hooks,
					FinishTurn: func(context.Context, *engine.Turn) (string, error) { return "end", nil },
				}, func(event engine.Event) error {
					if event.Type == "tool_execution_update" {
						update()
					}
					return nil
				}, stream)
				if err != nil {
					t.Fatal(err)
				}
			}
			if phase == "settled_edit" {
				edit(retained)
			}
			actual, err := json.Marshal(map[string]any{
				"snapshots": snapshots, "selected": selected, "retained": retained,
				"executed": executed, "after": after, "raw": call.Arguments,
				"sameExecute": retained == executed, "sameAfter": retained == after,
				"stillAttached": retained == selected.Get("branch"),
			})
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
