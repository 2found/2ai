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

func TestPiArgumentValues(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-argument-values.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name, Raw string
				Schema    json.RawMessage
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 884 {
		t.Fatal("unexpected argument value coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			inspect := func(value any) any {
				numbers := [][2]string{}
				var visit func(any, []any)
				visit = func(value any, path []any) {
					switch v := value.(type) {
					case float64:
						numbers = append(numbers, [2]string{argumentJSON(t, path), fmt.Sprintf("%016x", math.Float64bits(v))})
					case *engine.Object:
						for _, p := range v.Entries() {
							visit(p.Value, append(append([]any{}, path...), p.Name))
						}
					case *engine.Array:
						for i := 0; i < v.Len(); i++ {
							visit(v.Get(i), append(append([]any{}, path...), i))
						}
					}
				}
				visit(value, []any{})
				return map[string]any{"wire": argumentJSON(t, value), "numbers": numbers}
			}
			var before, executed any
			tool := &engine.Tool{
				Tool: ai.Tool{Name: "echo", Parameters: tc.Input.Schema},
				Execute: func(_ context.Context, _ string, args any, _ func(*engine.ToolResult)) (*engine.ToolResult, error) {
					executed = inspect(args)
					return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "done"}), Details: argumentRef(`{}`)}, nil
				},
			}
			call := &ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(tc.Input.Raw)}
			outcome, err := engine.RunToolCall(context.Background(), call, engine.NewList([]*engine.Tool{tool}...), &ai.Message{}, &engine.Context{}, engine.ToolHooks{
				Before: func(_ context.Context, c *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
					before = inspect(c.Args)
					return nil, nil
				},
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if string(call.Arguments) != tc.Input.Raw {
				t.Fatal("validation mutated raw model arguments")
			}
			source, err := jsonjs.StringifyJSON(call.Arguments)
			if err != nil {
				t.Fatal(err)
			}
			block, err := json.Marshal(outcome.Result.Content.Get(0))
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(block, &fields); err != nil {
				t.Fatal(err)
			}
			wireText, err := jsonjs.StringifyJSON(fields["text"])
			if err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"before": before, "executed": executed, "isError": outcome.IsError, "text": string(jsonjs.QuoteString(outcome.Result.Content.Get(0).Text)), "wireText": string(wireText), "source": string(source)})
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
