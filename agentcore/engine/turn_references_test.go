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

func turnReferenceMessage(message *ai.Message) any {
	if message == nil {
		return nil
	}
	return map[string]any{"role": message.Role, "timestamp": message.Timestamp}
}

func turnReferenceSlice(messages []*ai.Message) []any {
	result := []any{}
	for _, message := range messages {
		result = append(result, turnReferenceMessage(message))
	}
	return result
}

func turnReferenceList(messages *engine.MessageList) any {
	if messages == nil {
		return nil
	}
	return turnReferenceSlice(messages.Values())
}

func TestPiTurnReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-turn-references.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Mode, Stop, Subject, Phase, Action string
				Adopt                              bool
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 471 {
		t.Fatal("unexpected turn-reference oracle coverage")
	}
	for _, tc := range fixture.Cases {
		input := tc.Input
		t.Run(input.Mode+"/"+input.Stop+"/"+input.Subject+"/"+input.Phase+"/"+input.Action+"/adopt="+strconv.FormatBool(input.Adopt), func(t *testing.T) {
			observations, edits := []map[string]any{}, []map[string]any{}
			requests := [][]any{}
			refs := []*engine.Turn{}
			var first *engine.Turn
			var original engine.Turn
			var result *engine.MessageList
			var agent *engine.Agent
			completed, requested, responded := 0, 0, 0
			edited := false
			describe := func(turn *engine.Turn) map[string]any {
				ref := slices.Index(refs, turn)
				if ref < 0 {
					ref = len(refs)
					refs = append(refs, turn)
				}
				var contextMessages any
				if turn.Context != nil {
					contextMessages = turnReferenceSlice(turn.Context.Messages.Values())
				}
				return map[string]any{"ref": ref, "message": turnReferenceMessage(turn.Message), "toolResults": turnReferenceList(turn.ToolResults), "context": contextMessages, "newMessages": turnReferenceList(turn.NewMessages), "original": map[string]bool{
					"message": turn.Message == original.Message, "toolResults": turn.ToolResults == original.ToolResults, "context": turn.Context == original.Context, "newMessages": turn.NewMessages == original.NewMessages,
				}}
			}
			mutate := func(at string) {
				if first == nil || edited || input.Phase != at {
					return
				}
				edited = true
				target := first
				if input.Action == "local" {
					copy := *first
					target = &copy
				}
				switch input.Subject {
				case "message":
					target.Message = &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "replacement"}), Timestamp: 77}
					if input.Action == "clear" {
						target.Message = nil
					}
				case "toolResults":
					target.ToolResults = engine.NewList(&ai.Message{Role: "toolResult", ToolCallID: "replacement", ToolName: "echo", Content: ai.BlockContent(), Timestamp: 77})
					if input.Action == "clear" {
						target.ToolResults = nil
					}
				case "context":
					target.Context = &engine.Context{Messages: engine.NewList([]*ai.Message{loopListUser(77)}...), Tools: engine.NewList([]*engine.Tool{}...)}
					if input.Action == "clear" {
						target.Context = nil
					}
				case "newMessages":
					target.NewMessages = engine.NewList(loopListUser(77))
					if input.Action == "clear" {
						target.NewMessages = nil
					}
				}
				edits = append(edits, map[string]any{"at": at, "same": target == first, "turn": describe(target)})
			}
			observe := func(stage string, turn *engine.Turn) {
				if turn == nil {
					turn = first
				}
				observations = append(observations, map[string]any{"stage": stage, "turn": describe(turn), "retained": describe(first)})
			}
			model := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
			config := engine.Config{Model: model, Now: func() int64 { return 1000 }, ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil },
				FinishTurn: func(_ context.Context, turn *engine.Turn) (string, error) {
					completed++
					if first == nil {
						first = turn
						original = *turn
						mutate("finish")
					}
					observe("finish:"+strconv.Itoa(completed), turn)
					if completed == 1 {
						return "continue", nil
					}
					return "end", nil
				},
				PrepareNextTurn: func(turn *engine.Turn) (*engine.TurnUpdate, error) {
					mutate("next")
					observe("next", turn)
					if input.Adopt {
						return &engine.TurnUpdate{Context: turn.Context}, nil
					}
					return nil, nil
				},
				PrepareRequest: func(context.Context, engine.Request) (*engine.TurnUpdate, error) {
					if requested > 0 {
						mutate("request")
						observe("request", nil)
					}
					requested++
					return nil, nil
				},
				GetSteeringMessages: func() (*engine.MessageList, error) {
					if first != nil {
						mutate("steering")
					}
					return nil, nil
				},
			}
			emit := func(event engine.Event) error {
				if event.Type == "turn_end" {
					if completed == 1 {
						mutate("turn_end")
					}
					observe("turn_end:"+strconv.Itoa(completed), nil)
					observations[len(observations)-1]["event"] = map[string]any{"message": turnReferenceMessage(event.Message), "toolResults": turnReferenceList(event.ToolResults)}
				}
				if event.Type == "agent_end" {
					observe("agent_end", nil)
					observations[len(observations)-1]["event"] = map[string]any{"messages": turnReferenceList(event.Messages)}
					result = event.Messages
				}
				return nil
			}
			stream := func(_ context.Context, _ json.RawMessage, current ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				requests = append(requests, turnReferenceSlice(engine.MessagePointers(current.Messages())))
				responded++
				return loopListResponse(responded, input.Stop), nil
			}
			tools := []*engine.Tool{{Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}, Label: "echo", Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
				return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "ok"}), Details: engine.NewObject()}, nil
			}}}
			var err error
			switch input.Mode {
			case "run":
				result, err = engine.Run(context.Background(), engine.NewList([]*ai.Message{loopListUser(1)}...), engine.Context{Tools: engine.NewList(tools...)}, config, emit, stream)
			case "continue":
				result, err = engine.Continue(context.Background(), engine.Context{Messages: engine.NewList([]*ai.Message{loopListUser(1)}...), Tools: engine.NewList(tools...)}, config, emit, stream)
			case "agent":
				agent, err = engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: model, Tools: tools}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: stream,
					PrepareNextTurnWithContext: func(_ context.Context, turn *engine.Turn) (*engine.TurnUpdate, error) {
						return config.PrepareNextTurn(turn)
					},
				}})
				if err != nil {
					t.Fatal(err)
				}
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return emit(event) }})
				err = agent.Prompt(context.Background(), loopListUser(1))
			}
			if err != nil {
				t.Fatal(err)
			}
			mutate("settled")
			observe("settled", nil)
			retained := []map[string]any{}
			for _, turn := range refs {
				retained = append(retained, describe(turn))
			}
			got := map[string]any{"requests": requests, "observations": observations, "edits": edits, "result": turnReferenceList(result), "retained": retained}
			if agent != nil {
				got["state"] = turnReferenceList(agent.State().Messages)
			}
			assertStateListJSON(t, got, tc.Expected)
		})
	}
}
