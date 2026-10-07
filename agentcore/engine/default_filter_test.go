package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiDefaultFilterFailures(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-default-filter.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Pattern, Operation, Key string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 150 {
		t.Fatal("unexpected default-filter coverage")
	}
	message := func(role string, timestamp int64) *ai.Message {
		return &ai.Message{Role: role, Content: ai.TextContent("go"), Timestamp: timestamp}
	}
	shape := func(m *ai.Message) any {
		if m == nil {
			return nil
		}
		return map[string]any{"role": m.Role, "timestamp": m.Timestamp, "error": m.ErrorMessage}
	}
	values := func(list *engine.MessageList) any {
		if list == nil {
			return nil
		}
		result := []any{}
		for _, m := range list.Values() {
			result = append(result, shape(m))
		}
		return result
	}
	list := func(pattern string) *engine.MessageList {
		if pattern == "null_list" {
			return nil
		}
		result := engine.NewList[*ai.Message]()
		switch pattern {
		case "dense":
			for i, role := range []string{"system", "user", "assistant", "toolResult", "custom"} {
				result.Append(message(role, int64(10+i)))
			}
		case "sparse":
			result.SetLength(7)
			result.Set(1, message("system", 11))
			result.Set(3, message("user", 13))
			result.Set(5, message("custom", 15))
		case "null_head":
			result.Append(nil, message("user", 11))
		case "null_tail":
			result.Append(message("user", 10), nil)
		case "hole_head":
			result.Set(1, message("user", 11))
		case "holes":
			result.SetLength(3)
		case "empty", "named_only":
		default:
			result.Append(message("user", 10))
		}
		if strings.HasPrefix(pattern, "constructor_null") {
			result.SetProperty("constructor", nil)
		}
		if pattern == "constructor_object" {
			result.SetProperty("constructor", message("custom", 99))
		}
		if pattern == "filter_null" || pattern == "constructor_null_filter" {
			result.SetProperty("filter", nil)
		}
		if pattern == "filter_object" {
			result.SetProperty("filter", message("custom", 99))
		}
		if pattern == "constructor_null_entry" {
			result.Set(0, nil)
		}
		if pattern == "named_only" {
			result.SetProperty("extra", message("user", 99))
		}
		return result
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Pattern+"/"+tc.Input.Operation+"/"+tc.Input.Key, func(t *testing.T) {
			input := tc.Input
			events, requests, hooks := []any{}, []any{}, []string{}
			recovering := false
			var selected *engine.MessageList
			config := engine.AgentConfig{Config: engine.Config{
				Now: func() int64 { return 1000 },
				TransformContext: func(context.Context, *engine.MessageList) (*engine.MessageList, error) {
					hooks = append(hooks, "transform")
					selected = list(input.Pattern)
					if recovering {
						selected = engine.NewList(message("user", 90))
					}
					return selected, nil
				},
			}, StreamFn: func(_ context.Context, _ json.RawMessage, current ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				hooks = append(hooks, "stream")
				requests = append(requests, values(engine.NewList(engine.MessagePointers(current.Messages())...)))
				return loopListResponse(99+len(requests), "stop"), nil
			}}
			if input.Key != "absent" {
				config.GetAPIKey = func(provider string) (string, error) {
					hooks = append(hooks, "key:"+provider)
					if !recovering {
						switch input.Key {
						case "throw":
							return "", errors.New("key failed")
						case "edit":
							for _, m := range selected.Values() {
								if m != nil {
									m.Timestamp = 88
									break
								}
							}
						case "append":
							if selected != nil {
								selected.Append(message("user", 89))
							}
						}
					}
					return "secret", nil
				}
			}
			agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), Messages: []*ai.Message{message("user", 1)}}, AgentConfig: config})
			if err != nil {
				t.Fatal(err)
			}
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
				value := map[string]any{"type": event.Type}
				if event.Message != nil {
					value["message"] = shape(event.Message)
				}
				events = append(events, value)
				return nil
			}})
			observe := func() any {
				state := agent.State()
				var keys any
				if selected != nil {
					keys = selected.PropertyKeys()
				}
				return map[string]any{"selected": values(selected), "keys": keys, "history": values(state.Messages), "error": state.ErrorMessage, "streaming": state.IsStreaming, "signal": agent.Signal() != nil}
			}
			var failure, recoveryFailure any
			if input.Operation == "prompt" {
				err = agent.Prompt(context.Background(), message("user", 2))
			} else {
				err = agent.Continue(context.Background())
			}
			if err != nil {
				failure = err.Error()
			}
			if err := agent.WaitForIdle(context.Background()); err != nil {
				t.Fatal(err)
			}
			after := observe()
			recovering = true
			if err := agent.Prompt(context.Background(), message("user", 3)); err != nil {
				recoveryFailure = err.Error()
			}
			if err := agent.WaitForIdle(context.Background()); err != nil {
				t.Fatal(err)
			}
			assertStateListJSON(t, map[string]any{"failure": failure, "after": after, "recoveryFailure": recoveryFailure, "recovered": observe(), "events": events, "requests": requests, "hooks": hooks}, tc.Expected)
		})
	}
}
