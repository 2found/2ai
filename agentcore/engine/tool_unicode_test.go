package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

func TestPiToolUnicode(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-tool-unicode.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          json.RawMessage
		Cases          []struct {
			Name     string
			Text     json.RawMessage
			Input    []ai.Message
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 12 {
		t.Fatal("unexpected tool Unicode oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			value, err := jsonjs.DecodeJSON(tc.Text)
			if err != nil {
				t.Fatal(err)
			}
			text := value.(string)
			toolName, toolID := "tool_"+text, "call_"+text
			tools := tc.Input[0].ToolsAdded
			removal := map[string]any{"messages": tc.Input, "current": ai.GetCurrentTools(tc.Input), "changes": ai.GetToolStateChanges(tools, tools[1:])}
			executed, pending := []any{}, []any{}
			tool := &engine.Tool{Tool: tools[0], Label: text, Execute: func(_ context.Context, id string, args any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
				executed = append(executed, map[string]any{"id": id, "args": args})
				update(&engine.ToolResult{Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "progress"}).Blocks, Details: engine.NewObject()})
				return &engine.ToolResult{Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "ok"}).Blocks, Details: engine.NewObject()}, nil
			}}
			calls := 0
			agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: fixture.Model, Tools: []*engine.Tool{tool}}, AgentConfig: engine.AgentConfig{
				Config: engine.Config{Now: func() int64 { return 1000 }},
				StreamFn: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
					content := ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})
					reason := "stop"
					if calls == 0 {
						content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: toolID, Name: toolName, Arguments: json.RawMessage(`{}`)})
						reason = "toolUse"
					}
					calls++
					message := &ai.Message{Role: "assistant", Content: content, API: "test", Provider: "test", Model: "m", Usage: &ai.Usage{}, StopReason: reason, Timestamp: 1000}
					stream := ai.NewAssistantMessageEventStream()
					stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: reason, Message: message})
					stream.End()
					return stream, nil
				},
			}})
			if err != nil {
				t.Fatal(err)
			}
			events := []json.RawMessage{}
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
				raw, err := json.Marshal(event)
				if err != nil {
					return err
				}
				events = append(events, raw)
				switch event.Type {
				case "tool_execution_start", "tool_execution_update", "tool_execution_end":
					pending = append(pending, map[string]any{"type": event.Type, "ids": agent.State().PendingToolCalls.Values()})
				}
				return nil
			}})
			if err := agent.Prompt(context.Background(), "go"); err != nil {
				t.Fatal(err)
			}
			// Native observations use the same lossless codec as their source strings;
			// only the exported events/state below exercise the production JSON methods.
			observed, err := jsonjs.MarshalValue(map[string]any{"pending": pending, "executed": executed})
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(observed, &fields); err != nil {
				t.Fatal(err)
			}
			actual := map[string]any{"removal": removal, "events": events, "pending": fields["pending"], "executed": fields["executed"], "state": agent.State()}
			encoded, err := json.Marshal(actual)
			if err != nil {
				t.Fatal(err)
			}
			got, err := jsonjs.DecodeJSON(encoded)
			if err != nil {
				t.Fatal(err)
			}
			want, err := jsonjs.DecodeJSON(tc.Expected)
			if err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"removal", "events", "pending", "executed", "state"} {
				g, w := got.(map[string]any)[key], want.(map[string]any)[key]
				if !reflect.DeepEqual(g, w) {
					t.Error(firstJSONDifference(g, w, key))
				}
			}
		})
	}
}
