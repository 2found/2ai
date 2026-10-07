package engine_test

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiArgumentComposition(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-argument-composition.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, TypeboxVersion string
		Cases                          []struct {
			Name             string
			Parameters, Args json.RawMessage
			Expected         struct {
				Error   bool
				Content string
			}
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || fixture.TypeboxVersion != "1.3.27" || len(fixture.Cases) != 696 {
		t.Fatal("unexpected composition oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			executed := false
			tool := &engine.Tool{Tool: ai.Tool{Name: "audit", Parameters: tc.Parameters}}
			tool.Execute = func(_ context.Context, _ string, args any, _ func(*engine.ToolResult)) (*engine.ToolResult, error) {
				executed = true
				return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: argumentJSON(t, args)})}, nil
			}
			original := bytes.Clone(tc.Args)
			tools := engine.NewList(tool)
			outcome, err := engine.RunToolCall(context.Background(), &ai.ContentBlock{Type: "toolCall", ID: "call", Name: "audit", Arguments: tc.Args}, tools, &ai.Message{}, &engine.Context{Tools: tools}, engine.ToolHooks{}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if outcome.IsError != tc.Expected.Error || executed == tc.Expected.Error || outcome.Result.Content.Len() != 1 || outcome.Result.Content.Get(0).Text != tc.Expected.Content {
				t.Fatalf("Go: error=%v executed=%v content=%+v\nPi: %+v", outcome.IsError, executed, outcome.Result.Content, tc.Expected)
			}
			if !bytes.Equal(tc.Args, original) {
				t.Fatal("validation mutated transcript arguments")
			}
		})
	}
}
