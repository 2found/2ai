package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiTurnOptions(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-turn-options.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Seed, Phase, ModelKind, Thinking string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 336 {
		t.Fatal("unexpected turn-option coverage")
	}
	model := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
	models := map[string]json.RawMessage{"null": json.RawMessage(`null`), "replacement": json.RawMessage(`{"id":"new","api":"new","provider":"new"}`), "false": json.RawMessage(`false`), "zero": json.RawMessage(`0`), "empty_string": json.RawMessage(`""`)}
	levels := map[string]string{"off": "off", "low": "low", "empty": ""}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Mode+"/"+tc.Input.Seed+"/"+tc.Input.Phase+"/"+tc.Input.ModelKind+"/"+tc.Input.Thinking, func(t *testing.T) {
			input := tc.Input
			requests, prepared, events := []any{}, []any{}, []json.RawMessage{}
			count, finished := 0, 0
			applied := false
			update := func() *engine.TurnUpdate {
				if applied {
					return nil
				}
				applied = true
				result := &engine.TurnUpdate{Model: models[input.ModelKind]}
				if input.Thinking != "unset" {
					level := levels[input.Thinking]
					result.ThinkingLevel = &level
				}
				return result
			}
			config := engine.Config{Model: model, Reasoning: levels[input.Seed], Now: func() int64 { return 1000 },
				ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil },
				PrepareRequest: func(_ context.Context, request engine.Request) (*engine.TurnUpdate, error) {
					prepared = append(prepared, map[string]any{"model": request.Model, "thinkingLevel": request.ThinkingLevel})
					if input.Phase == "request" {
						return update(), nil
					}
					return nil, nil
				},
				PrepareNextTurn: func(*engine.Turn) (*engine.TurnUpdate, error) {
					if input.Phase == "next" {
						return update(), nil
					}
					return nil, nil
				},
				FinishTurn: func(context.Context, *engine.Turn) (string, error) {
					finished++
					if finished < 3 {
						return "continue", nil
					}
					return "end", nil
				},
			}
			stream := func(_ context.Context, requested json.RawMessage, _ ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
				count++
				value, defined := options["reasoning"]
				reasoning := map[string]any{"defined": defined}
				if defined {
					reasoning["value"] = value
				}
				requests = append(requests, map[string]any{"model": requested, "spreadModel": options["model"], "reasoning": reasoning})
				return loopListResponse(99+count, "stop"), nil
			}
			emit := func(event engine.Event) error {
				raw, err := json.Marshal(event)
				if err != nil {
					return err
				}
				events = append(events, raw)
				return nil
			}
			var result *engine.MessageList
			var stateThinking any
			if input.Mode == "loop" {
				result, err = engine.Run(context.Background(), engine.NewList(loopListUser(1)), engine.Context{}, config, emit, stream)
			} else {
				var agent *engine.Agent
				agent, err = engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: model, ThinkingLevel: levels[input.Seed]}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: stream, PrepareNextTurn: func(context.Context) (*engine.TurnUpdate, error) { return config.PrepareNextTurn(nil) }}})
				if err != nil {
					t.Fatal(err)
				}
				if input.Seed == "empty" {
					agent.SetThinkingLevel("")
				}
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return emit(event) }})
				err = agent.Prompt(context.Background(), loopListUser(1))
				result, stateThinking = agent.State().Messages, agent.State().ThinkingLevel
			}
			if err != nil {
				t.Fatal(err)
			}
			assertStateListJSON(t, map[string]any{"requests": requests, "prepared": prepared, "events": events, "result": result, "stateThinking": stateThinking}, tc.Expected)
		})
	}
}

func TestRequestAdmissionPreservesSelectionAndEmptyThinking(t *testing.T) {
	calls := 0
	config := engine.Config{Model: json.RawMessage(`{"id":"test"}`), Reasoning: "low",
		AdmitRequest: func(_ context.Context, request engine.Request, options map[string]any) (*engine.RequestAdmission, error) {
			calls++
			if calls == 1 {
				request.ThinkingLevel = ""
				request.Model = json.RawMessage(`null`)
			} else {
				spread, ok := options["model"].(json.RawMessage)
				if string(request.Model) != "null" || !ok || string(spread) != "null" {
					t.Fatalf("admitted model lost: request=%s, option=%v", request.Model, options["model"])
				}
				value, defined := options["reasoning"]
				if request.ThinkingLevel != "" || !defined || value != "" {
					t.Fatalf("empty reasoning lost at admission: request=%q, option=%v, defined=%v", request.ThinkingLevel, value, defined)
				}
				request.ThinkingLevel = "off"
			}
			return &engine.RequestAdmission{Request: request, Stream: loopListResponse(calls, "stop")}, nil
		},
		FinishTurn: func(context.Context, *engine.Turn) (string, error) {
			if calls == 1 {
				return "continue", nil
			}
			return "end", nil
		},
	}
	result, err := engine.Run(context.Background(), nil, engine.Context{}, config, func(engine.Event) error { return nil }, nil)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || result.Len() != 2 || result.Get(0).ThinkingLevel == nil || *result.Get(0).ThinkingLevel != "" || result.Get(1).ThinkingLevel == nil || *result.Get(1).ThinkingLevel != "off" {
		t.Fatalf("unexpected admitted thinking: %+v", result.Values())
	}
}
