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
	"github.com/2found/2ai/jsonjs"
)

func TestPiFailureUnicode(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-failure-unicode.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          json.RawMessage
		Cases          []struct {
			Name, Mode     string
			Text, Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 48 {
		t.Fatal("unexpected failure Unicode oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			decoded, err := jsonjs.DecodeJSON(tc.Text)
			if err != nil {
				t.Fatal(err)
			}
			text := decoded.(string)
			agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: fixture.Model}, AgentConfig: engine.AgentConfig{
				Config: engine.Config{Now: func() int64 { return 1000 }},
				StreamFn: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
					switch tc.Mode {
					case "throw_error":
						return nil, errors.New(text)
					case "throw_string":
						panic(text)
					}
					failure := text
					if tc.Mode == "listener" {
						failure = "before"
					}
					message := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: ""}), API: "test", Provider: "test", Model: "m", Usage: &ai.Usage{}, StopReason: "error", ErrorMessage: &failure, Timestamp: 1000}
					stream := ai.NewAssistantMessageEventStream()
					stream.Push(ai.AssistantMessageEvent{Type: "error", Reason: "error", Error: message})
					stream.End()
					return stream, nil
				},
			}})
			if err != nil {
				t.Fatal(err)
			}
			events := []json.RawMessage{}
			observed := []map[string]any{}
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
				if tc.Mode == "listener" && event.Type == "message_end" && event.Message.Role == "assistant" {
					event.Message.ErrorMessage = &text
				}
				encoded, err := json.Marshal(event)
				if err != nil {
					return err
				}
				events = append(events, encoded)
				state, err := json.Marshal(agent.State())
				if err != nil {
					return err
				}
				observed = append(observed, map[string]any{"type": event.Type, "state": json.RawMessage(state)})
				return nil
			}})
			if err := agent.Prompt(context.Background(), "go"); err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"events": events, "observed": observed, "state": agent.State()})
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
			if !reflect.DeepEqual(got, want) {
				t.Fatal(firstJSONDifference(got, want, ""))
			}
		})
	}
}
