package engine_test

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

func TestPiResultValues(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-result-values.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Mode, Shape, Phase string
				Decoded            bool
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 160 {
		t.Fatal("unexpected result value coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(fmt.Sprintf("%s/%s/%s/%t", tc.Input.Mode, tc.Input.Shape, tc.Input.Phase, tc.Input.Decoded), func(t *testing.T) {
			mode, shape, phase := tc.Input.Mode, tc.Input.Shape, tc.Input.Phase
			leaf := func(text string) any {
				child := engine.NewObject(engine.Property{Name: "text", Value: text})
				if shape == "array" {
					return engine.NewArray(child)
				}
				return engine.NewObject(engine.Property{Name: "child", Value: child})
			}
			shared := leaf("original")
			branch := func(value any) *engine.Object { return engine.NewObject(engine.Property{Name: "branch", Value: value}) }
			replacement := func() *engine.Object { return branch(leaf("replacement")) }
			edit := func(value any) {
				if shape == "array" {
					value := value.(*engine.Array)
					value.Get(0).(*engine.Object).Set("text", "changed")
					value.Append(engine.NewObject(engine.Property{Name: "text", Value: "appended"}))
				} else {
					value := value.(*engine.Object)
					value.Get("child").(*engine.Object).Set("text", "changed")
					value.Set("added", true)
				}
			}
			terminate := true
			original := &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "done"}), Details: branch(shared), StructuredContent: branch(shared), Terminate: &terminate}
			if tc.Input.Decoded {
				raw, err := json.Marshal(original)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(raw, original); err != nil {
					t.Fatal(err)
				}
				shared = original.Details.(*engine.Object).Get("branch")
			}
			partial := &engine.ToolResult{Content: ai.NewBlockList(), Details: original.Details, StructuredContent: original.StructuredContent}
			snapshots := []json.RawMessage{}
			capture := func(stage string, result any) {
				raw, err := json.Marshal(map[string]any{"stage": stage, "result": result})
				if err != nil {
					t.Fatal(err)
				}
				snapshots = append(snapshots, raw)
			}
			var end, afterResult *engine.ToolResult
			messages := []*ai.Message{}
			update := func(value *engine.ToolResult) {
				capture("update", value)
				if phase == "update_edit" {
					edit(value.Details.(*engine.Object).Get("branch"))
				}
				if phase == "update_replace" {
					value.Details = replacement()
				}
			}
			after := func(_ context.Context, value engine.AfterToolCall) (*engine.AfterToolResult, error) {
				afterResult = value.Result
				capture("after", value.Result)
				switch phase {
				case "after_edit":
					edit(value.Result.Details.(*engine.Object).Get("branch"))
				case "after_result_clear":
					value.Result.Details = engine.Undefined
					value.Result.StructuredContent = engine.Null
				case "after_empty_override":
					return &engine.AfterToolResult{}, nil
				case "after_content_override":
					return &engine.AfterToolResult{Content: ai.NewBlockList()}, nil
				case "after_details_override":
					return &engine.AfterToolResult{Details: replacement()}, nil
				case "after_structured_override":
					return &engine.AfterToolResult{StructuredContent: replacement()}, nil
				case "after_null_override":
					return &engine.AfterToolResult{Details: engine.Null, StructuredContent: engine.Null}, nil
				case "after_null_content_override":
					return &engine.AfterToolResult{Details: engine.Null, StructuredContent: engine.Null, Content: ai.NewBlockList()}, nil
				}
				return nil, nil
			}
			tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}, Execute: func(_ context.Context, _ string, _ any, onUpdate func(*engine.ToolResult)) (*engine.ToolResult, error) {
				onUpdate(partial)
				return original, nil
			}}
			call := &ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)}
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockReferences(call), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: "toolUse", Timestamp: 1}
			if mode == "programmatic" {
				outcome, err := engine.RunToolCall(context.Background(), call, engine.NewList([]*engine.Tool{tool}...), assistant, &engine.Context{}, engine.ToolHooks{After: after}, func(value *engine.ToolResult) error { update(value); return nil })
				if err != nil || outcome.IsError {
					t.Fatalf("failed: %+v %v", outcome, err)
				}
				end = outcome.Result
			} else {
				stream := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
					s := ai.NewAssistantMessageEventStream()
					s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "toolUse", Message: assistant})
					return s, nil
				}
				result, err := engine.Run(context.Background(), nil, engine.Context{Messages: engine.NewList([]*ai.Message{}...), Tools: engine.NewList([]*engine.Tool{tool}...)}, engine.Config{
					Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`),
					Now:   func() int64 { return 1000 }, ToolExecution: mode,
					ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil },
					ToolHooks:    engine.ToolHooks{After: after},
					FinishTurn:   func(context.Context, *engine.Turn) (string, error) { return "end", nil },
				}, func(event engine.Event) error {
					switch event.Type {
					case "tool_execution_update":
						update(event.PartialResult)
					case "tool_execution_end":
						end = event.Result
						capture("end", end)
						if phase == "end_edit" {
							edit(end.Details.(*engine.Object).Get("branch"))
						}
					case "message_start":
						if event.Message.Role == "toolResult" {
							capture("message", event.Message)
							if phase == "message_edit" {
								edit(event.Message.Details.(*engine.Object).Get("branch"))
							}
						}
					}
					return nil
				}, stream)
				if err != nil {
					t.Fatal(err)
				}
				for _, message := range result.Values() {
					if message.Role == "toolResult" {
						messages = append(messages, message)
					}
				}
			}
			if phase == "settled_edit" {
				edit(shared)
			}
			var messageIdentity any
			if mode != "programmatic" {
				messageIdentity = messages[0].Details == end.Details
			}
			nested := false
			if details, ok := end.Details.(*engine.Object); ok {
				nested = details.Get("branch") == shared
			}
			actual, err := json.Marshal(map[string]any{
				"snapshots": snapshots, "original": original, "partial": partial, "end": end, "messages": messages, "shared": shared,
				"identity": map[string]any{"after": afterResult == original, "end": end == original, "details": end.Details == original.Details, "structured": end.StructuredContent == original.StructuredContent, "nested": nested, "message": messageIdentity},
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

func TestPiResultOptionalValues(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-result-values.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Values         []struct {
			Input struct {
				Name, Raw, Phase string
				Function         bool
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Values) != 66 {
		t.Fatal("unexpected optional result coverage")
	}
	for _, tc := range fixture.Values {
		t.Run(tc.Input.Name, func(t *testing.T) {
			var original engine.ToolResult
			if err := json.Unmarshal([]byte(tc.Input.Raw), &original); err != nil {
				t.Fatal(err)
			}
			if tc.Input.Function {
				original.Details = func() {}
				original.StructuredContent = func() {}
			}
			tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}, Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
				return &original, nil
			}}
			outcome, err := engine.RunToolCall(context.Background(), &ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)}, engine.NewList([]*engine.Tool{tool}...), &ai.Message{}, &engine.Context{}, engine.ToolHooks{
				After: func(_ context.Context, c engine.AfterToolCall) (*engine.AfterToolResult, error) {
					switch tc.Input.Phase {
					case "empty_override":
						return &engine.AfterToolResult{}, nil
					case "null_override":
						return &engine.AfterToolResult{Details: engine.Null, StructuredContent: engine.Null}, nil
					case "content_override":
						return &engine.AfterToolResult{Content: ai.NewBlockList()}, nil
					case "primitive_override":
						return &engine.AfterToolResult{Details: false, StructuredContent: float64(0)}, nil
					case "clear":
						c.Result.Details = engine.Undefined
						c.Result.StructuredContent = engine.Null
					}
					return nil, nil
				},
			}, nil)
			if err != nil || outcome.IsError {
				t.Fatalf("failed: %+v %v", outcome, err)
			}
			describe := func(value any) any {
				if value == nil || jsonjs.IsUndefined(value) {
					return map[string]any{"kind": "undefined"}
				}
				if jsonjs.IsNull(value) {
					return map[string]any{"kind": "null"}
				}
				if reflect.TypeOf(value).Kind() == reflect.Func {
					return map[string]any{"kind": "function"}
				}
				kind := "object"
				result := map[string]any{"wire": argumentJSON(t, value)}
				switch value := value.(type) {
				case bool:
					kind = "boolean"
				case string:
					kind = "string"
				case float64:
					kind = "number"
					result["bits"] = fmt.Sprintf("%016x", math.Float64bits(value))
				}
				result["kind"] = kind
				return result
			}
			encoded, err := json.Marshal(outcome.Result)
			if err != nil {
				t.Fatal(err)
			}
			wire, err := jsonjs.StringifyJSON(encoded)
			if err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"wire": string(wire), "details": describe(outcome.Result.Details), "structured": describe(outcome.Result.StructuredContent)})
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

type passiveResultText string

func (passiveResultText) MarshalJSON() ([]byte, error) { panic("user serializer must not run") }

func TestToolResultLiveGraphDoesNotSerializeDuringExecution(t *testing.T) {
	details := engine.NewObject(engine.Property{Name: "text", Value: passiveResultText("passive")}, engine.Property{Name: "number", Value: math.Inf(1)})
	details.Set("self", details)
	details.Set("function", func() {})
	original := &engine.ToolResult{Content: ai.NewBlockList(), Details: details, StructuredContent: details}
	tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}, Execute: func(_ context.Context, _ string, _ any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
		update(original)
		return original, nil
	}}
	updates := 0
	outcome, err := engine.RunToolCall(context.Background(), &ai.ContentBlock{Type: "toolCall", Name: "echo", Arguments: json.RawMessage(`{}`)}, engine.NewList([]*engine.Tool{tool}...), nil, &engine.Context{}, engine.ToolHooks{
		After: func(_ context.Context, call engine.AfterToolCall) (*engine.AfterToolResult, error) {
			if call.Result != original || call.Result.Details != details {
				t.Fatal("after hook lost live graph")
			}
			return &engine.AfterToolResult{}, nil
		},
	}, func(value *engine.ToolResult) error {
		updates++
		if value != original || value.Details != details {
			t.Fatal("update lost live graph")
		}
		return nil
	})
	if err != nil || outcome.IsError || updates != 1 || outcome.Result == original || outcome.Result.Details != details || outcome.Result.StructuredContent != details {
		t.Fatalf("execution changed live result graph: %+v %v updates=%d", outcome, err, updates)
	}
	if _, err := json.Marshal(outcome.Result); err == nil {
		t.Fatal("explicit export accepted a cycle")
	}
	details.Delete("self")
	raw, err := json.Marshal(outcome.Result)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"content":[],"details":{"text":"passive","number":null},"structuredContent":{"text":"passive","number":null}}`
	if string(raw) != want {
		t.Fatalf("passive export: %s", raw)
	}
}
