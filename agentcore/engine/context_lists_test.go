package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func contextListMessages(list *engine.MessageList) []any {
	result := []any{}
	for _, m := range list.Values() {
		if m == nil {
			result = append(result, nil)
			continue
		}
		value := map[string]any{"role": m.Role, "timestamp": m.Timestamp}
		if m.Role == "toolResult" {
			value["id"], value["isError"] = m.ToolCallID, m.IsError
		}
		result = append(result, value)
	}
	return result
}

func TestPiDefaultContextFilter(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-context-lists.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Defaults []struct {
			Variant  string
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Defaults) != 6 {
		t.Fatal("unexpected default filter coverage")
	}
	for _, tc := range fixture.Defaults {
		t.Run(tc.Variant, func(t *testing.T) {
			var selected *engine.MessageList
			requests := [][]any{}
			agent, err := engine.NewAgent(engine.AgentOptions{
				InitialState: engine.InitialState{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)},
				AgentConfig: engine.AgentConfig{Config: engine.Config{
					TransformContext: func(_ context.Context, input *engine.MessageList) (*engine.MessageList, error) {
						if tc.Variant == "identity" {
							selected = input
							return selected, nil
						}
						selected = engine.NewList[*ai.Message]()
						selected.SetLength(4)
						if tc.Variant != "grow_empty" {
							selected.Set(1, loopListUser(77))
						}
						if tc.Variant == "mixed" {
							selected.Set(3, &ai.Message{Role: "custom", Content: ai.TextContent("custom"), Timestamp: 99})
						}
						return selected, nil
					},
					GetAPIKey: func(string) (string, error) {
						if tc.Variant == "key_append" {
							selected.Append(loopListUser(88))
						}
						if tc.Variant == "key_nested" {
							selected.Get(1).Timestamp = 88
						}
						return "key", nil
					},
				}, StreamFn: func(_ context.Context, _ json.RawMessage, context ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
					requests = append(requests, contextListMessages(engine.NewList(engine.MessagePointers(context.Messages())...)))
					return loopListResponse(1, "stop"), nil
				}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := agent.Prompt(context.Background(), loopListUser(1)); err != nil {
				t.Fatal(err)
			}
			assertStateListJSON(t, map[string]any{"requests": requests, "selected": contextListMessages(selected), "keys": selected.PropertyKeys(), "state": contextListMessages(agent.State().Messages)}, tc.Expected)
		})
	}
}

func contextListTools(list *engine.ToolList) []any {
	result := []any{}
	for _, tool := range list.Values() {
		result = append(result, map[string]any{"name": tool.Name, "label": tool.Label})
	}
	return result
}

func TestPiContextLists(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-context-lists.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Subject, Phase, Action string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 504 {
		t.Fatal("unexpected context-list oracle coverage")
	}
	for _, tc := range fixture.Cases {
		input := tc.Input
		t.Run(input.Mode+"/"+input.Subject+"/"+input.Phase+"/"+input.Action, func(t *testing.T) {
			observations, requests, executed := []map[string]any{}, [][]any{}, []map[string]string{}
			refs := []any{}
			tool := func(name string) *engine.Tool {
				return &engine.Tool{Tool: ai.Tool{Name: name, Parameters: json.RawMessage(`{"type":"object"}`)}, Label: name, Execute: func(_ context.Context, id string, _ any, _ func(*engine.ToolResult)) (*engine.ToolResult, error) {
					executed = append(executed, map[string]string{"executor": name, "id": id})
					return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "ok"}), Details: engine.NewObject()}, nil
				}}
			}
			initial := engine.Context{Messages: engine.NewList(loopListUser(1)), Tools: engine.NewList(tool("echo"))}
			var current *engine.Context
			var retainedMessages *engine.MessageList
			var retainedTools *engine.ToolList
			var retained any
			var result *engine.MessageList
			var agent *engine.Agent
			edited, requested, responded, finished := false, 0, 0, 0
			ref := func(value any) int {
				id := slices.Index(refs, value)
				if id < 0 {
					id = len(refs)
					refs = append(refs, value)
				}
				return id
			}
			mutate := func(at string) {
				if edited || input.Phase != at {
					return
				}
				edited = true
				if input.Subject == "messages" {
					switch input.Action {
					case "append":
						retainedMessages.Append(loopListUser(77))
					case "replace":
						retainedMessages.Set(0, loopListUser(77))
					case "shrink":
						retainedMessages.SetLength(0)
					case "rebind_same":
						current.Messages = retainedMessages
					case "rebind_new":
						current.Messages = engine.NewList(loopListUser(77))
					case "detach_append":
						current.Messages = retainedMessages.Clone()
						retainedMessages.Append(loopListUser(77))
					case "nested":
						retainedMessages.Get(0).Timestamp = 88
					}
				} else {
					switch input.Action {
					case "append":
						retainedTools.Append(tool("extra"))
					case "replace":
						retainedTools.Set(0, tool("extra"))
					case "shrink":
						retainedTools.SetLength(0)
					case "rebind_same":
						current.Tools = retainedTools
					case "rebind_new":
						current.Tools = engine.NewList(tool("extra"))
					case "detach_append":
						current.Tools = retainedTools.Clone()
						retainedTools.Append(tool("extra"))
					case "nested":
						retainedTools.Get(0).Name = "extra"
					}
				}
			}
			observe := func(stage string, parameter *engine.MessageList) {
				mr, tr, rr := ref(current.Messages), ref(current.Tools), ref(retained)
				retainedValue, same := contextListMessages(retainedMessages), retainedMessages == current.Messages
				if input.Subject == "tools" {
					retainedValue, same = contextListTools(retainedTools), retainedTools == current.Tools
				}
				negative := []any{}
				if value, ok := current.Messages.GetProperty("-1"); ok {
					negative = contextListMessages(engine.NewList(value))
				}
				value := map[string]any{"stage": stage, "messageRef": mr, "toolRef": tr, "messages": contextListMessages(current.Messages), "messageKeys": current.Messages.PropertyKeys(), "negativeLast": negative, "tools": contextListTools(current.Tools), "retainedRef": rr, "retained": retainedValue, "same": same}
				if parameter != nil {
					value["parameterRef"], value["parameter"], value["parameterIsCurrent"] = ref(parameter), contextListMessages(parameter), parameter == current.Messages
				}
				observations = append(observations, value)
			}
			model := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
			config := engine.Config{Model: model, Now: func() int64 { return 1000 }, ToolExecution: "sequential",
				PrepareRequest: func(_ context.Context, request engine.Request) (*engine.TurnUpdate, error) {
					current = request.Context
					if retained == nil {
						if input.Subject == "messages" {
							retainedMessages = current.Messages
							retained = retainedMessages
						} else {
							retainedTools = current.Tools
							retained = retainedTools
						}
					}
					if requested == 0 {
						mutate("request")
					} else {
						mutate("request2")
					}
					requested++
					observe("request:"+strconv.Itoa(requested), nil)
					return nil, nil
				},
				TransformContext: func(_ context.Context, parameter *engine.MessageList) (*engine.MessageList, error) {
					mutate("transform")
					observe("transform", parameter)
					return parameter, nil
				},
				ConvertToLLM: func(parameter *engine.MessageList) (*engine.MessageList, error) {
					mutate("convert")
					observe("convert", parameter)
					return parameter, nil
				},
				GetAPIKey: func(string) (string, error) { mutate("key"); observe("key", nil); return "key", nil },
				ToolHooks: engine.ToolHooks{Before: func(_ context.Context, call *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
					mutate("before")
					observe("before:"+call.ToolCall.ID, nil)
					return nil, nil
				}},
				FinishTurn: func(context.Context, *engine.Turn) (string, error) {
					finished++
					if finished == 1 {
						mutate("finish")
					}
					observe("finish:"+strconv.Itoa(finished), nil)
					if finished == 1 {
						return "continue", nil
					}
					return "end", nil
				},
				PrepareNextTurn: func(*engine.Turn) (*engine.TurnUpdate, error) { mutate("next"); observe("next", nil); return nil, nil },
			}
			stream := func(_ context.Context, _ json.RawMessage, context ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				requests = append(requests, contextListMessages(engine.NewList(engine.MessagePointers(context.Messages())...)))
				responded++
				content := ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})
				if responded == 1 {
					content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "first", Name: "echo", Arguments: json.RawMessage(`{}`)}, ai.ContentBlock{Type: "toolCall", ID: "second", Name: "extra", Arguments: json.RawMessage(`{}`)})
				}
				message := &ai.Message{Role: "assistant", Content: content, API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: "stop", Timestamp: int64(responded + 1)}
				s := ai.NewAssistantMessageEventStream()
				s.Push(ai.AssistantMessageEvent{Type: "start", Partial: message})
				s.Push(ai.AssistantMessageEvent{Type: "text_delta", ContentIndex: 0, Delta: "", Partial: message})
				s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
				return s, nil
			}
			emit := func(event engine.Event) error {
				if event.Message != nil && event.Message.Role == "assistant" {
					at := map[string]string{"message_start": "start", "message_update": "update", "message_end": "end"}[event.Type]
					if at != "" {
						mutate(at)
						observe(at, nil)
					}
				}
				if event.Type == "agent_end" {
					result = event.Messages
					observe("agent_end", nil)
				}
				return nil
			}
			var err error
			switch input.Mode {
			case "run":
				result, err = engine.Run(context.Background(), engine.NewList([]*ai.Message{loopListUser(10)}...), initial, config, emit, stream)
			case "continue":
				result, err = engine.Continue(context.Background(), initial, config, emit, stream)
			case "agent":
				agent, err = engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: model, Messages: initial.Messages.Values(), Tools: initial.Tools.Values()}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: stream,
					PrepareNextTurnWithContext: func(_ context.Context, turn *engine.Turn) (*engine.TurnUpdate, error) {
						return config.PrepareNextTurn(turn)
					},
				}})
				if err != nil {
					t.Fatal(err)
				}
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return emit(event) }})
				err = agent.Prompt(context.Background(), loopListUser(10))
			}
			if err != nil {
				t.Fatal(err)
			}
			mutate("settled")
			observe("settled", nil)
			got := map[string]any{"observations": observations, "requests": requests, "executed": executed, "result": contextListMessages(result), "initial": map[string]any{"messages": contextListMessages(initial.Messages), "tools": contextListTools(initial.Tools), "sameMessages": initial.Messages == current.Messages, "sameTools": initial.Tools == current.Tools}}
			if agent != nil {
				state := agent.State()
				got["state"] = map[string]any{"messages": contextListMessages(state.Messages), "tools": contextListTools(state.Tools), "sameMessages": state.Messages == current.Messages, "sameTools": state.Tools == current.Tools}
			}
			assertStateListJSON(t, got, tc.Expected)
		})
	}
}
