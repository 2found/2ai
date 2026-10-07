package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func inputListMessage(message *ai.Message) any {
	if message == nil {
		return nil
	}
	tools := []string{}
	for _, tool := range message.ToolsAdded {
		tools = append(tools, tool.Name)
	}
	return map[string]any{"role": message.Role, "timestamp": message.Timestamp, "tools": tools}
}

func inputListValues(list *engine.MessageList) []any {
	values := []any{}
	for _, message := range list.Values() {
		values = append(values, inputListMessage(message))
	}
	return values
}

func TestPiInputLists(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-input-lists.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Kind, Mode, Declaration, Subject, Phase, Action string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 594 {
		t.Fatal("unexpected input-list oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Kind+"/"+tc.Input.Mode+"/"+tc.Input.Declaration+tc.Input.Subject+"/"+tc.Input.Phase+"/"+tc.Input.Action, func(t *testing.T) {
			input := tc.Input
			selected := engine.NewList(loopListUser(10), loopListUser(11))
			if input.Kind == "prompt" && input.Declaration == "rewrite" {
				selected.Set(0, &ai.Message{Role: "system", Content: ai.TextContent(""), Timestamp: 10})
			}
			retained := selected
			edited := false
			mutate := func(at string) {
				if edited || input.Phase != at {
					return
				}
				edited = true
				switch input.Action {
				case "append":
					retained.Append(loopListUser(77))
				case "replace":
					retained.Set(1, loopListUser(77))
				case "shrink":
					retained.SetLength(0)
				case "truncate":
					retained.SetLength(1)
				case "rebind":
					selected = engine.NewList(loopListUser(77))
				case "nested":
					retained.Get(1).Timestamp = 88
				}
			}
			events, requests, hooks := []any{}, []any{}, []any{}
			turns, requested, finished, steering, followup := 0, 0, 0, 0, 0
			targetTurn := 2
			if input.Subject == "initial" {
				targetTurn = 1
			}
			var result *engine.MessageList
			emit := func(event engine.Event) error {
				if input.Kind == "prompt" {
					mutate(event.Type)
				} else {
					if event.Type == "turn_start" {
						turns++
						if turns == targetTurn {
							mutate("turn_start")
						}
					}
					if turns == targetTurn && event.Message != nil && event.Message.Role == "user" && event.Message.Timestamp >= 10 {
						mutate(event.Type)
					}
				}
				value := map[string]any{"type": event.Type}
				if event.Message != nil {
					value["message"] = inputListMessage(event.Message)
				}
				if event.Messages != nil {
					value["messages"] = inputListValues(event.Messages)
				}
				events = append(events, value)
				if event.Type == "agent_end" {
					result = event.Messages
				}
				return nil
			}
			model := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
			config := engine.Config{Model: model, Now: func() int64 { return 1000 },
				ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil },
				PrepareRequest: func(context.Context, engine.Request) (*engine.TurnUpdate, error) {
					requested++
					if input.Kind == "prompt" || requested == targetTurn {
						mutate("request")
					}
					return nil, nil
				},
				FinishTurn: func(context.Context, *engine.Turn) (string, error) {
					finished++
					if input.Kind == "pending" && finished == 1 {
						return "continue", nil
					}
					return "end", nil
				},
			}
			if input.Kind == "pending" {
				config.PrepareNextTurn = func(*engine.Turn) (*engine.TurnUpdate, error) {
					mutate("next")
					hooks = append(hooks, map[string]any{"type": "next", "selected": inputListValues(selected), "retained": inputListValues(retained)})
					if input.Subject == "prepared" {
						return &engine.TurnUpdate{Messages: selected}, nil
					}
					return nil, nil
				}
				config.GetSteeringMessages = func() (*engine.MessageList, error) {
					steering++
					if steering == 3 {
						mutate("poll")
					}
					output := engine.NewList[*ai.Message]()
					if input.Subject == "initial" && steering == 1 || input.Subject == "steering" && steering == 2 {
						output = selected
					}
					hooks = append(hooks, map[string]any{"type": "steering", "number": steering, "messages": inputListValues(output)})
					return output, nil
				}
				config.GetFollowUpMessages = func() (*engine.MessageList, error) {
					followup++
					output := engine.NewList[*ai.Message]()
					if input.Subject == "followup" && followup == 1 {
						output = selected
					}
					hooks = append(hooks, map[string]any{"type": "followup", "number": followup, "messages": inputListValues(output)})
					return output, nil
				}
			}
			provider := func(_ context.Context, _ json.RawMessage, context ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				requests = append(requests, inputListValues(engine.NewList(engine.MessagePointers(context.Messages())...)))
				return loopListResponse(len(requests), "stop"), nil
			}
			initial := engine.Context{Messages: engine.NewList[*ai.Message](), Tools: engine.NewList[*engine.Tool]()}
			prompts := selected
			if input.Kind == "prompt" && input.Declaration != "none" {
				initial.Tools.Append(&engine.Tool{Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}, Label: "echo"})
			}
			if input.Kind == "pending" {
				initial.Messages.Append(loopListUser(1))
				prompts = engine.NewList(loopListUser(2))
			}
			var agent *engine.Agent
			var err error
			switch input.Mode {
			case "run":
				result, err = engine.Run(context.Background(), prompts, initial, config, emit, provider)
			case "continue":
				result, err = engine.Continue(context.Background(), initial, config, emit, provider)
			case "agent":
				options := engine.AgentOptions{InitialState: engine.InitialState{Model: model, Messages: initial.Messages.Values()}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: provider}}
				if input.Kind == "pending" {
					options.PrepareNextTurnWithContext = func(_ context.Context, turn *engine.Turn) (*engine.TurnUpdate, error) {
						return config.PrepareNextTurn(turn)
					}
				}
				agent, err = engine.NewAgent(options)
				if err != nil {
					t.Fatal(err)
				}
				agent.SetTools(initial.Tools.Values())
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return emit(event) }})
				err = agent.Prompt(context.Background(), prompts)
			}
			if err != nil {
				t.Fatal(err)
			}
			mutate("settled")
			actual := map[string]any{"events": events, "requests": requests, "result": inputListValues(result), "selected": inputListValues(selected), "retained": inputListValues(retained), "same": selected == retained, "edited": edited}
			if input.Kind == "pending" {
				actual["hooks"] = hooks
			}
			if agent != nil {
				actual["state"] = inputListValues(agent.State().Messages)
			}
			assertStateListJSON(t, actual, tc.Expected)
		})
	}
}
