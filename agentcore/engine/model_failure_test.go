package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiModelFailure(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-model-failure.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Kind, Phase, Failure string
				Model                json.RawMessage
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 144 {
		t.Fatal("unexpected model/failure coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Kind+"/"+tc.Input.Phase+"/"+tc.Input.Failure, func(t *testing.T) {
			input := tc.Input
			normal := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
			events, requests, hooks := []json.RawMessage{}, []any{}, []string{}
			recovering := false
			var agent *engine.Agent
			capture := func(v any) json.RawMessage {
				raw, err := json.Marshal(v)
				if err != nil {
					t.Fatal(err)
				}
				return raw
			}
			initial := normal
			if input.Phase == "initial" {
				initial = input.Model
			}
			agent, err = engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: initial}, AgentConfig: engine.AgentConfig{Config: engine.Config{
				Now: func() int64 { return 1000 },
				TransformContext: func(_ context.Context, messages *engine.MessageList) (*engine.MessageList, error) {
					hooks = append(hooks, "transform")
					if !recovering && input.Phase == "transform" {
						agent.SetModel(input.Model)
					}
					if !recovering && input.Failure == "transform" {
						return nil, errors.New("transform failed")
					}
					return messages, nil
				},
			}, StreamFn: func(_ context.Context, requested json.RawMessage, _ ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				hooks = append(hooks, "stream")
				value := map[string]any{"defined": requested != nil}
				if requested != nil {
					value["model"] = requested
				}
				requests = append(requests, value)
				if !recovering && input.Phase == "stream" {
					agent.SetModel(input.Model)
				}
				if !recovering && input.Failure == "stream" {
					return nil, errors.New("stream failed")
				}
				return loopListResponse(99+len(requests), "stop"), nil
			}}})
			if err != nil {
				t.Fatal(err)
			}
			if input.Phase == "before" {
				agent.SetModel(input.Model)
			}
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { events = append(events, capture(event)); return nil }})
			observe := func() any { return map[string]any{"state": capture(agent.State()), "signal": agent.Signal() != nil} }
			before := observe()
			var failure, recoveryFailure any
			if err := agent.Prompt(context.Background(), loopListUser(1)); err != nil {
				failure = err.Error()
			}
			if err := agent.WaitForIdle(context.Background()); err != nil {
				t.Fatal(err)
			}
			after := observe()
			recovering = true
			agent.SetModel(normal)
			if err := agent.Prompt(context.Background(), loopListUser(2)); err != nil {
				recoveryFailure = err.Error()
			}
			if err := agent.WaitForIdle(context.Background()); err != nil {
				t.Fatal(err)
			}
			assertStateListJSON(t, map[string]any{"initial": before, "error": failure, "after": after, "recoveryError": recoveryFailure, "recovered": observe(), "events": events, "requests": requests, "hooks": hooks}, tc.Expected)
		})
	}
}
