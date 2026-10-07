package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiPendingToolSets(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-pending-tools.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Mode, Variant string
				IDs           []string
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 135 {
		t.Fatal("unexpected pending-tool oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(fmt.Sprintf("%s/%v/%s", tc.Input.Mode, tc.Input.IDs, tc.Input.Variant), func(t *testing.T) {
			input := tc.Input
			var agent *engine.Agent
			var mu sync.Mutex
			observations := []map[string]any{}
			sets := []*engine.ToolCallSet{}
			observe := func(stage string) {
				mu.Lock()
				defer mu.Unlock()
				state := agent.State()
				set := state.PendingToolCalls
				ref := slices.Index(sets, set)
				if ref < 0 {
					ref = len(sets)
					sets = append(sets, set)
				}
				has := []bool{}
				for _, id := range append(slices.Clone(input.IDs), "absent") {
					has = append(has, set.Has(id))
				}
				wire, err := json.Marshal(set)
				if err != nil {
					panic(err)
				}
				stateRaw, err := json.Marshal(state)
				if err != nil {
					panic(err)
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(stateRaw, &fields); err != nil {
					panic(err)
				}
				observations = append(observations, map[string]any{"stage": stage, "ref": ref, "size": set.Len(), "values": set.Values(), "has": has, "wire": string(wire), "stateWire": string(fields["pendingToolCalls"]), "repeatedGetter": set == agent.State().PendingToolCalls})
				// Values is a detached observation, not mutable backing storage.
				if values := set.Values(); len(values) > 0 {
					values[0] = "overwritten"
				}
			}
			gates := make([]chan struct{}, len(input.IDs))
			for i := range gates {
				gates[i] = make(chan struct{})
			}
			if len(gates) > 0 {
				close(gates[len(gates)-1])
			}
			result := func() *engine.ToolResult {
				return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "ok"}), Details: engine.NewObject()}
			}
			tools := []*engine.Tool{}
			blocks := []*ai.ContentBlock{}
			for i, id := range input.IDs {
				name := fmt.Sprintf("tool%d", i)
				tools = append(tools, &engine.Tool{Tool: ai.Tool{Name: name, Parameters: json.RawMessage(`{"type":"object"}`)}, Label: name,
					Execute: func(_ context.Context, _ string, _ any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
						if input.Mode == "parallel" {
							<-gates[i]
						}
						observe(fmt.Sprintf("execute:%d", i))
						if input.Variant == "execute_error" {
							return nil, errors.New("execute failed")
						}
						update(result())
						return result(), nil
					},
				})
				args := json.RawMessage(`{}`)
				if input.Variant == "invalid" {
					args = json.RawMessage(`[]`)
				}
				blocks = append(blocks, &ai.ContentBlock{Type: "toolCall", ID: id, Name: name, Arguments: args})
			}
			if input.Variant == "missing" {
				tools = nil
			}
			message := &ai.Message{Role: "assistant", Content: ai.BlockReferences(blocks...), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: "stop", Timestamp: 2}
			if input.Variant == "truncated" {
				message.StopReason = "length"
			}
			var err error
			agent, err = engine.NewAgent(engine.AgentOptions{
				InitialState: engine.InitialState{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), Tools: tools},
				AgentConfig: engine.AgentConfig{
					StreamFn: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
						s := ai.NewAssistantMessageEventStream()
						s.Push(ai.AssistantMessageEvent{Type: "done", Reason: message.StopReason, Message: message})
						return s, nil
					},
					Config: engine.Config{Now: func() int64 { return 1000 }, ToolExecution: input.Mode,
						ToolHooks: engine.ToolHooks{
							Before: func(_ context.Context, call *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
								observe("before:" + call.ToolCall.Name)
								if input.Variant == "mutate_id" {
									call.ToolCall.ID = "changed:" + call.ToolCall.Name
								}
								if input.Variant == "blocked" {
									return &engine.BeforeToolResult{Block: true}, nil
								}
								if input.Variant == "abort_before" {
									agent.Abort()
								}
								return nil, nil
							},
							After: func(_ context.Context, call engine.AfterToolCall) (*engine.AfterToolResult, error) {
								observe("after:" + call.ToolCall.Name)
								if input.Variant == "after_error" {
									return nil, errors.New("after failed")
								}
								return nil, nil
							},
						},
						FinishTurn: func(context.Context, *engine.Turn) (string, error) { observe("finish"); return "end", nil },
					},
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			observe("initial")
			failed := false
			failureEvent := map[string]string{"listener_start_error": "tool_execution_start", "listener_update_error": "tool_execution_update", "listener_end_error": "tool_execution_end", "listener_agent_end_error": "agent_end"}[input.Variant]
			stage := func(event engine.Event) string {
				name := event.ToolName
				if name == "" && event.Message != nil {
					name = event.Message.Role
				}
				return event.Type + ":" + name
			}
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
				observe("first:" + stage(event))
				if !failed && event.Type == failureEvent {
					failed = true
					return errors.New("listener failed")
				}
				return nil
			}})
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
				observe("second:" + stage(event))
				if event.Type == "tool_execution_end" {
					i, err := strconv.Atoi(strings.TrimPrefix(event.ToolName, "tool"))
					if err != nil {
						return err
					}
					if i > 0 {
						close(gates[i-1])
					}
				}
				return nil
			}})
			if err := agent.Prompt(context.Background(), &ai.Message{Role: "user", Content: ai.TextContent("go"), Timestamp: 1}); err != nil {
				t.Fatal(err)
			}
			observe("settled")
			if err := agent.Reset(); err != nil {
				t.Fatal(err)
			}
			observe("reset")
			retained := [][]string{}
			for _, set := range sets {
				retained = append(retained, set.Values())
			}
			assertStateListJSON(t, map[string]any{"observations": observations, "retained": retained}, tc.Expected)
		})
	}
}

func TestPendingToolSetsConcurrentReaders(t *testing.T) {
	tool := &engine.Tool{Tool: ai.Tool{Name: "tool", Parameters: json.RawMessage(`{"type":"object"}`)},
		Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
			return &engine.ToolResult{Content: ai.NewBlockList(), Details: engine.NewObject()}, nil
		},
	}
	blocks := []*ai.ContentBlock{}
	for i := 0; i < 32; i++ {
		blocks = append(blocks, &ai.ContentBlock{Type: "toolCall", ID: strconv.Itoa(i), Name: "tool", Arguments: json.RawMessage(`{}`)})
	}
	agent, err := engine.NewAgent(engine.AgentOptions{
		InitialState: engine.InitialState{Tools: []*engine.Tool{tool}},
		AgentConfig: engine.AgentConfig{
			Config: engine.Config{ToolExecution: "sequential", FinishTurn: func(context.Context, *engine.Turn) (string, error) { return "end", nil }},
			StreamFn: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				s := ai.NewAssistantMessageEventStream()
				s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: &ai.Message{Role: "assistant", Content: ai.BlockReferences(blocks...), StopReason: "stop"}})
				return s, nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	retained := map[*engine.ToolCallSet][]string{}
	agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
		if event.Type == "tool_execution_start" || event.Type == "tool_execution_end" {
			set := agent.State().PendingToolCalls
			retained[set] = set.Values()
		}
		return nil
	}})
	stop := make(chan struct{})
	var readers, ready sync.WaitGroup
	for i := 0; i < 4; i++ {
		readers.Add(1)
		ready.Add(1)
		go func() {
			defer readers.Done()
			ready.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				set := agent.State().PendingToolCalls
				values := set.Values()
				if len(values) != set.Len() {
					t.Error("retained set changed length")
				}
				for _, id := range values {
					if !set.Has(id) {
						t.Error("retained set lost a member")
					}
				}
				if len(values) > 0 {
					values[0] = "detached"
				}
				if raw, err := json.Marshal(set); err != nil || string(raw) != "{}" {
					t.Error("set JSON changed")
				}
			}
		}()
	}
	ready.Wait()
	err = agent.Prompt(context.Background(), "go")
	close(stop)
	readers.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if len(retained) != 64 {
		t.Fatalf("retained %d sets, want 64", len(retained))
	}
	for set, expected := range retained {
		if !slices.Equal(set.Values(), expected) {
			t.Errorf("retained set changed: %v => %v", expected, set.Values())
		}
	}
}
