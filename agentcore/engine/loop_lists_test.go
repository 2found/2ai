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

type loopListInput struct{ Mode, Stop, Subject, Phase, Action string }
type loopListFixture struct {
	UpstreamCommit string
	Cases          []struct {
		Input    loopListInput
		Expected json.RawMessage
	}
	Streams []struct {
		Input    loopListInput
		Expected json.RawMessage
	}
}

func loopListFixtures(t *testing.T) loopListFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-loop-lists.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture loopListFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 360 || len(fixture.Streams) != 12 {
		t.Fatal("unexpected loop-list oracle coverage")
	}
	return fixture
}

func loopListSummary(list *engine.MessageList) map[string]any {
	values := []any{}
	for _, message := range list.Values() {
		var value any
		if message != nil {
			value = map[string]any{"role": message.Role, "timestamp": message.Timestamp}
		}
		values = append(values, value)
	}
	return map[string]any{"length": list.Len(), "keys": list.Keys(), "values": values}
}

func loopListUser(timestamp int64) *ai.Message {
	return &ai.Message{Role: "user", Content: ai.TextContent("go"), Timestamp: timestamp}
}

func editLoopList(list *engine.MessageList, action string) {
	switch action {
	case "append":
		list.Append(loopListUser(77))
	case "replace":
		list.Set(0, loopListUser(77))
	case "delete":
		list.Delete(0)
	case "shrink":
		list.SetLength(0)
	case "grow":
		list.SetLength(list.Len() + 2)
	case "nested":
		if message := list.Get(0); message != nil {
			message.Timestamp = 88
		}
	}
}

func loopListResponse(number int, stop string) *ai.AssistantMessageEventStream {
	content := ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})
	if number == 1 && stop == "normal" {
		content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)})
	}
	if stop == "normal" {
		stop = "stop"
	}
	message := &ai.Message{Role: "assistant", Content: content, API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: stop, Timestamp: int64(number + 1)}
	s := ai.NewAssistantMessageEventStream()
	if stop == "error" || stop == "aborted" {
		s.Push(ai.AssistantMessageEvent{Type: "error", Reason: stop, Error: message})
	} else {
		s.Push(ai.AssistantMessageEvent{Type: "done", Reason: stop, Message: message})
	}
	return s
}

func TestPiLoopLists(t *testing.T) {
	for _, tc := range loopListFixtures(t).Cases {
		input := tc.Input
		t.Run(input.Mode+"/"+input.Stop+"/"+input.Subject+"/"+input.Phase+"/"+input.Action, func(t *testing.T) {
			observations, requests := []map[string]any{}, []map[string]any{}
			lists := []*engine.MessageList{}
			var first *engine.Turn
			var terminal, result *engine.MessageList
			var agent *engine.Agent
			turns, responses, requestCount := 0, 0, 0
			describe := func(list *engine.MessageList) map[string]any {
				index := slices.Index(lists, list)
				if index < 0 {
					index = len(lists)
					lists = append(lists, list)
				}
				value := loopListSummary(list)
				value["ref"] = index
				return value
			}
			observe := func(stage string, turn *engine.Turn) {
				value := map[string]any{"stage": stage, "retainedNew": describe(first.NewMessages), "retainedTools": describe(first.ToolResults)}
				if turn != nil {
					value["currentNew"], value["currentTools"] = describe(turn.NewMessages), describe(turn.ToolResults)
				}
				observations = append(observations, value)
			}
			mutate := func(at string) {
				if input.Phase != at {
					return
				}
				list := first.NewMessages
				if input.Subject == "toolResults" {
					list = first.ToolResults
				}
				editLoopList(list, input.Action)
			}
			model := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
			config := engine.Config{Model: model, Now: func() int64 { return 1000 }, ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil },
				FinishTurn: func(_ context.Context, turn *engine.Turn) (string, error) {
					turns++
					if first == nil {
						first = turn
						mutate("finish")
					}
					observe("finish:"+strconv.Itoa(turns), turn)
					if turns == 1 {
						return "continue", nil
					}
					return "end", nil
				},
				PrepareNextTurn: func(turn *engine.Turn) (*engine.TurnUpdate, error) {
					mutate("next")
					observe("next", turn)
					return &engine.TurnUpdate{Messages: engine.NewList([]*ai.Message{loopListUser(10)}...)}, nil
				},
				PrepareRequest: func(context.Context, engine.Request) (*engine.TurnUpdate, error) {
					if requestCount > 0 {
						mutate("request")
						observe("request", nil)
					}
					requestCount++
					return nil, nil
				},
			}
			stream := func(_ context.Context, _ json.RawMessage, current ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				requests = append(requests, loopListSummary(engine.NewList(engine.MessagePointers(current.Messages())...)))
				responses++
				return loopListResponse(responses, input.Stop), nil
			}
			emit := func(event engine.Event) error {
				if event.Type == "turn_end" {
					if turns == 1 {
						mutate("turn_end")
					}
					observe("turn_end:"+strconv.Itoa(turns), nil)
					observations[len(observations)-1]["eventTools"] = describe(event.ToolResults)
				}
				if event.Type == "agent_end" {
					terminal = event.Messages
					mutate("agent_end")
					observe("agent_end", nil)
					observations[len(observations)-1]["terminal"] = describe(terminal)
				}
				return nil
			}
			tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}, Label: "echo", Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
				return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "ok"}), Details: engine.NewObject()}, nil
			}}
			initial := engine.Context{Tools: engine.NewList([]*engine.Tool{tool}...)}
			var err error
			switch input.Mode {
			case "run":
				result, err = engine.Run(context.Background(), engine.NewList([]*ai.Message{loopListUser(1)}...), initial, config, emit, stream)
			case "continue":
				initial.Messages = engine.NewList(loopListUser(1))
				result, err = engine.Continue(context.Background(), initial, config, emit, stream)
			case "agent":
				agent, err = engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: model, Tools: initial.Tools.Values()}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: stream,
					PrepareNextTurnWithContext: func(_ context.Context, turn *engine.Turn) (*engine.TurnUpdate, error) {
						return config.PrepareNextTurn(turn)
					},
				}})
				if err != nil {
					t.Fatal(err)
				}
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return emit(event) }})
				err = agent.Prompt(context.Background(), loopListUser(1))
				result = terminal
			}
			if err != nil {
				t.Fatal(err)
			}
			mutate("settled")
			observe("settled", nil)
			got := map[string]any{"requests": requests, "observations": observations, "result": describe(result), "terminal": describe(terminal)}
			retained := []map[string]any{}
			for _, list := range lists {
				retained = append(retained, loopListSummary(list))
			}
			got["retained"] = retained
			if agent != nil {
				got["state"] = loopListSummary(agent.State().Messages)
			}
			assertStateListJSON(t, got, tc.Expected)
		})
	}
}

func TestPiLoopStreamResultLists(t *testing.T) {
	for _, tc := range loopListFixtures(t).Streams {
		t.Run(tc.Input.Mode+"/"+tc.Input.Action, func(t *testing.T) {
			ctx := context.Background()
			config := engine.Config{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }}
			provider := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				return loopListResponse(1, "stop"), nil
			}
			var stream *engine.AgentEventStream
			var err error
			if tc.Input.Mode == "run" {
				stream = engine.AgentLoop(ctx, engine.NewList([]*ai.Message{loopListUser(1)}...), engine.Context{}, config, provider)
			} else {
				stream, err = engine.AgentLoopContinue(ctx, engine.Context{Messages: engine.NewList([]*ai.Message{loopListUser(1)}...)}, config, provider)
			}
			if err != nil {
				t.Fatal(err)
			}
			result, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if err := stream.Wait(ctx); err != nil {
				t.Fatal(err)
			}
			var terminal *engine.MessageList
			for {
				event, ok, err := stream.Next(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !ok {
					break
				}
				if event.Type == "agent_end" {
					terminal = event.Messages
				}
			}
			editLoopList(result, tc.Input.Action)
			again, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			assertStateListJSON(t, map[string]any{"same": result == terminal, "repeated": result == again, "result": loopListSummary(result), "terminal": loopListSummary(terminal)}, tc.Expected)
		})
	}
}
