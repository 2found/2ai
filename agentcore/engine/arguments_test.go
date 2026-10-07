package engine_test

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiToolArgumentsOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/pi-arguments.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string `json:"upstreamCommit"`
		Cases          []struct {
			Name                       string
			Parameters, Args, Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 1812 {
		t.Fatal("unexpected argument oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			executed := false
			var prepared *string
			tool := &engine.Tool{Tool: ai.Tool{Name: "number", Parameters: tc.Parameters}}
			tool.Execute = func(_ context.Context, _ string, args any, _ func(*engine.ToolResult)) (*engine.ToolResult, error) {
				executed = true
				return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: argumentJSON(t, args)}), Details: argumentRef(`{}`)}, nil
			}
			before := append([]byte(nil), tc.Args...)
			call := ai.ContentBlock{Type: "toolCall", ID: "call", Name: "number", Arguments: tc.Args}
			outcome, err := engine.RunToolCall(context.Background(), &call, engine.NewList([]*engine.Tool{tool}...), &ai.Message{}, &engine.Context{Tools: engine.NewList([]*engine.Tool{tool}...)}, engine.ToolHooks{
				Before: func(_ context.Context, hook *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
					text := argumentJSON(t, hook.Args)
					prepared = &text
					return nil, nil
				},
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(before, []byte(tc.Args)) {
				t.Fatal("argument coercion mutated source history")
			}
			var original bytes.Buffer
			if err := json.Compact(&original, tc.Args); err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"executed": executed, "prepared": prepared, "isError": outcome.IsError, "content": outcome.Result.Content.Get(0).Text, "source": original.String()})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			_ = json.Unmarshal(actual, &got)
			_ = json.Unmarshal(tc.Expected, &want)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
