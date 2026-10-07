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

func TestPiMetadataKeys(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-metadata-keys.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          json.RawMessage
		Cases          []struct {
			Name  string
			Input struct {
				Key             json.RawMessage
				Initial         ai.Message
				Declaration     ai.Tool
				Result, Partial engine.ToolResult
				Replies         []*ai.Message
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 8 {
		t.Fatal("unexpected metadata-key oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			decodedKey, err := jsonjs.DecodeJSON(tc.Input.Key)
			if err != nil {
				t.Fatal(err)
			}
			key := decodedKey.(string)
			events, requests := []json.RawMessage{}, []json.RawMessage{}
			tool := &engine.Tool{Tool: tc.Input.Declaration, Label: "Echo", Execute: func(_ context.Context, _ string, _ any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
				update(&tc.Input.Partial)
				return &tc.Input.Result, nil
			}}
			calls := 0
			agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: fixture.Model, Tools: []*engine.Tool{tool}, Messages: []*ai.Message{&tc.Input.Initial}}, AgentConfig: engine.AgentConfig{
				Config: engine.Config{Now: func() int64 { return 1000 }},
				StreamFn: func(_ context.Context, _ json.RawMessage, transcript ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
					raw, err := json.Marshal(transcript.Messages())
					if err != nil {
						return nil, err
					}
					requests = append(requests, raw)
					message := tc.Input.Replies[calls]
					calls++
					stream := ai.NewAssistantMessageEventStream()
					stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: message.StopReason, Message: message})
					stream.End()
					return stream, nil
				},
			}})
			if err != nil {
				t.Fatal(err)
			}
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
				if event.Type == "message_end" && event.Message.Role == "assistant" {
					event.Message.Extra[key] = json.RawMessage(`{"changed":"message"}`)
					event.Message.Content.Blocks.Get(0).Extra[key] = json.RawMessage(`{"changed":"block"}`)
				}
				raw, err := json.Marshal(event)
				events = append(events, raw)
				return err
			}})
			if err := agent.Prompt(context.Background(), "go"); err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"requests": requests, "events": events, "state": agent.State()})
			if err != nil {
				t.Fatal(err)
			}
			got, err := jsonjs.DecodeJSON(actual)
			if err != nil {
				t.Fatal(err)
			}
			want, err := jsonjs.DecodeJSON(tc.Expected)
			if err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"requests", "events", "state"} {
				g, w := got.(map[string]any)[key], want.(map[string]any)[key]
				if !reflect.DeepEqual(g, w) {
					t.Error(firstJSONDifference(g, w, key))
				}
			}
		})
	}
}
