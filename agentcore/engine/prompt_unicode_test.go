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

func TestPiPromptUnicode(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-prompt-unicode.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Name            string
			Input, Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 45 {
		t.Fatal("unexpected prompt unicode oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			var message ai.Message
			if err := json.Unmarshal(tc.Input, &message); err != nil {
				t.Fatal(err)
			}
			agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Messages: []*ai.Message{&message}}, AgentConfig: engine.AgentConfig{StreamFn: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				t.Fatal("unexpected request")
				return nil, nil
			}}})
			if err != nil {
				t.Fatal(err)
			}
			read := func() map[string]any {
				state, err := json.Marshal(agent.State())
				if err != nil {
					t.Fatal(err)
				}
				decodedState, err := jsonjs.DecodeJSON(state)
				if err != nil {
					t.Fatal(err)
				}
				wire, err := json.Marshal(message)
				if err != nil {
					t.Fatal(err)
				}
				decodedWire, err := jsonjs.DecodeJSON(wire)
				if err != nil {
					t.Fatal(err)
				}
				return map[string]any{"content": ai.ContentText(message.Content), "system": ai.GetSystemMessageText(message), "update": ai.RenderSystemMessageUpdate(message), "current": ai.GetCurrentSystemPrompt([]ai.Message{message}), "state": decodedState.(map[string]any)["systemPrompt"], "wire": decodedWire}
			}
			before := read()
			*message.Content.Text += "/edited"
			for _, section := range message.Sections {
				if section.Value != nil {
					*section.Value += "/edited"
				}
			}
			actual, err := jsonjs.MarshalValue(map[string]any{"before": before, "after": read()})
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
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
