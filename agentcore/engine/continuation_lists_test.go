package engine_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func continuationHistory(pattern string) *engine.MessageList {
	list := engine.NewList[*ai.Message]()
	if pattern == "empty" {
		return list
	}
	if pattern == "holes" {
		list.SetLength(2)
		return list
	}
	for index, role := range strings.Split(pattern, "_") {
		if role == "hole" {
			list.SetLength(index + 1)
		} else if role == "null" {
			list.Append(nil)
		} else {
			// Preserve the source object's own fields, including its deliberately
			// minimal assistant shape; the native constructor supplies defaults.
			var message *ai.Message
			if err := json.Unmarshal([]byte(fmt.Sprintf(`{"role":%q,"content":"go","timestamp":%d}`, role, index+1)), &message); err != nil {
				panic(err)
			}
			list.Append(message)
		}
	}
	return list
}

func continuationMessage(m *ai.Message) any {
	if m == nil {
		return nil
	}
	return map[string]any{"role": m.Role, "timestamp": m.Timestamp, "error": m.ErrorMessage}
}

func continuationValues(list *engine.MessageList) []any {
	values := []any{}
	for _, m := range list.Values() {
		values = append(values, continuationMessage(m))
	}
	return values
}

// A continuation admission must reject through its error result, never panic or
// retain the wrapper's mutex. Timeout also catches a stuck second invocation.
func checkedContinuation(t *testing.T, run func() error) error {
	t.Helper()
	type outcome struct {
		err        error
		panicValue any
	}
	done := make(chan outcome, 1)
	go func() {
		var result outcome
		defer func() { result.panicValue = recover(); done <- result }()
		result.err = run()
	}()
	select {
	case result := <-done:
		if result.panicValue != nil {
			t.Fatalf("continuation panicked: %v", result.panicValue)
		}
		return result.err
	case <-time.After(3 * time.Second):
		t.Fatal("continuation did not settle")
		return nil
	}
}

func TestPiContinuationLists(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-continuation-lists.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Pattern, Queue, Edit string }
			Expected json.RawMessage
		}
		Wrappers []struct {
			Pattern string
			Error   string
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 408 || len(fixture.Wrappers) != 6 {
		t.Fatal("unexpected continuation-list oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Mode+"/"+tc.Input.Pattern+"/"+tc.Input.Queue+"/"+tc.Input.Edit, func(t *testing.T) {
			input := tc.Input
			selected := continuationHistory(input.Pattern)
			var agent *engine.Agent
			events, requests := []any{}, []any{}
			repaired := false
			steering, followup := 0, 0
			var ended *engine.MessageList
			emit := func(event engine.Event) error {
				if event.Type == "agent_start" && input.Edit == "start_fill" && !repaired {
					repaired = true
					for i := 0; i < selected.Len(); i++ {
						if selected.Get(i) == nil {
							selected.Set(i, loopListUser(int64(70+i)))
						}
					}
				}
				value := map[string]any{"type": event.Type}
				if event.Message != nil {
					value["message"] = continuationMessage(event.Message)
				}
				if event.Messages != nil {
					value["messages"] = continuationValues(event.Messages)
				}
				events = append(events, value)
				if event.Type == "agent_end" {
					ended = event.Messages
				}
				return nil
			}
			provider := func(_ context.Context, _ json.RawMessage, current ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				requests = append(requests, continuationValues(engine.NewList(engine.MessagePointers(current.Messages())...)))
				return loopListResponse(19, "stop"), nil
			}
			model := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
			config := engine.Config{Model: model, Now: func() int64 { return 1000 },
				FinishTurn:   func(context.Context, *engine.Turn) (string, error) { return "end", nil },
				ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil },
				GetSteeringMessages: func() (*engine.MessageList, error) {
					steering++
					if steering == 1 && (input.Queue == "steering" || input.Queue == "both") {
						return engine.NewList(loopListUser(90)), nil
					}
					return nil, nil
				},
				GetFollowUpMessages: func() (*engine.MessageList, error) {
					followup++
					if followup == 1 && (input.Queue == "followup" || input.Queue == "both") {
						return engine.NewList(loopListUser(91)), nil
					}
					return nil, nil
				},
			}
			if input.Mode == "agent" {
				var err error
				agent, err = engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: model}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: provider}})
				if err != nil {
					t.Fatal(err)
				}
				agent.SetMessageList(selected)
				selected = agent.State().Messages
				if input.Queue == "steering" || input.Queue == "both" {
					agent.Steer(loopListUser(90))
				}
				if input.Queue == "followup" || input.Queue == "both" {
					agent.FollowUp(loopListUser(91))
				}
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return emit(event) }})
			}
			if input.Edit == "constructor" {
				selected.SetProperty("constructor", nil)
			}
			err := checkedContinuation(t, func() error {
				if agent != nil {
					return agent.Continue(context.Background())
				}
				_, err := engine.Continue(context.Background(), engine.Context{Messages: selected}, config, emit, provider)
				return err
			})
			observe := func() map[string]any {
				var result any
				if ended != nil {
					result = continuationValues(ended)
				}
				value := map[string]any{"events": append([]any{}, events...), "requests": append([]any{}, requests...), "history": map[string]any{"length": selected.Len(), "keys": selected.PropertyKeys(), "values": continuationValues(selected)}, "result": result, "repaired": repaired}
				if agent != nil {
					state := agent.State()
					value["streaming"], value["error"], value["signal"] = state.IsStreaming, state.ErrorMessage, agent.Signal() != nil
					value["queued"], value["peek"] = agent.HasQueuedMessages(), continuationValues(engine.NewList(agent.PeekQueuedMessages()...))
				} else {
					value["steering"], value["followup"] = steering, followup
				}
				return value
			}
			first := observe()
			first["failure"] = nil
			if err != nil {
				first["failure"] = err.Error()
			}
			var recovered any
			if agent != nil {
				agent.SetMessages([]*ai.Message{loopListUser(30)})
				selected = agent.State().Messages
				if err := checkedContinuation(t, func() error { return agent.Continue(context.Background()) }); err != nil {
					t.Fatal(err)
				}
				if err := agent.WaitForIdle(context.Background()); err != nil {
					t.Fatal(err)
				}
				recovered = observe()
			}
			assertStateListJSON(t, map[string]any{"first": first, "recovered": recovered}, tc.Expected)
		})
	}
	for _, tc := range fixture.Wrappers {
		t.Run("wrapper/"+tc.Pattern, func(t *testing.T) {
			err := checkedContinuation(t, func() error {
				_, err := engine.AgentLoopContinue(context.Background(), engine.Context{Messages: continuationHistory(tc.Pattern)}, engine.Config{}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
					return nil, fmt.Errorf("unexpected provider call")
				})
				return err
			})
			if err == nil || err.Error() != tc.Error {
				t.Fatalf("error = %v, want %s", err, tc.Error)
			}
		})
	}
}

func TestPiSparseHistoryReads(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-continuation-lists.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Reads []struct {
			Input struct {
				Pattern, Operation string
				Constructor        bool
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Reads) != 102 {
		t.Fatal("unexpected sparse history-read coverage")
	}
	for _, tc := range fixture.Reads {
		t.Run(fmt.Sprintf("%s/%s/constructor=%v", tc.Input.Pattern, tc.Input.Operation, tc.Input.Constructor), func(t *testing.T) {
			agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)}, AgentConfig: engine.AgentConfig{StreamFn: completedAgentStream}})
			if err != nil {
				t.Fatal(err)
			}
			agent.SetMessageList(continuationHistory(tc.Input.Pattern))
			if tc.Input.Constructor {
				agent.State().Messages.SetProperty("constructor", nil)
			}
			agent.Steer(loopListUser(90))
			agent.FollowUp(loopListUser(91))
			var value any
			switch tc.Input.Operation {
			case "prompt":
				func() {
					defer func() {
						if failure := recover(); failure != nil {
							err = fmt.Errorf("%v", failure)
						}
					}()
					value = agent.State().SystemPrompt()
				}()
			case "json":
				var encoded []byte
				encoded, err = agent.State().MarshalJSON()
				if err == nil {
					if err := json.Unmarshal(encoded, &value); err != nil {
						t.Fatal(err)
					}
				}
			case "reset":
				err = checkedContinuation(t, agent.Reset)
			}
			var failure any
			if err != nil {
				failure = err.Error()
			}
			selected := agent.State().Messages
			actual := map[string]any{"value": value, "error": failure, "history": map[string]any{"length": selected.Len(), "keys": selected.PropertyKeys(), "values": continuationValues(selected)}, "queued": agent.HasQueuedMessages(), "peek": continuationValues(engine.NewList(agent.PeekQueuedMessages()...))}
			assertStateListJSON(t, actual, tc.Expected)
		})
	}
}
