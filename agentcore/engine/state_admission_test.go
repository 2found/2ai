package engine_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiStateAdmission(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-state-admission.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Constructors   []struct {
			Input struct {
				Head, Tools, Stream string
				Policy              bool
			}
			Expected json.RawMessage
		}
		Queues []struct {
			Input    struct{ Steering, Followup, SteeringMode, FollowUpMode, Operation string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Constructors) != 96 || len(fixture.Queues) != 216 {
		t.Fatal("unexpected state admission coverage")
	}
	model := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
	message := func(role string, timestamp int64) *ai.Message {
		return &ai.Message{Role: role, Content: ai.TextContent("go"), Timestamp: timestamp}
	}
	original, _ := engine.GetDefaultStreamFn()
	defer engine.SetDefaultStreamFn(original)
	for _, tc := range fixture.Constructors {
		t.Run(fmt.Sprintf("construct/%s/%s/%t/%s", tc.Input.Head, tc.Input.Tools, tc.Input.Policy, tc.Input.Stream), func(t *testing.T) {
			input := tc.Input
			tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Description: "tool", Parameters: json.RawMessage(`{"type":"object"}`)}}
			initial := engine.InitialState{Model: model}
			if input.Policy {
				initial.SystemPrompt = "policy"
			}
			if input.Head == "null" {
				initial.Messages = []*ai.Message{nil}
			} else if input.Head != "empty" {
				initial.Messages = []*ai.Message{message(input.Head, 1)}
			}
			switch input.Tools {
			case "one":
				initial.Tools = []*engine.Tool{tool}
			case "null":
				initial.Tools = []*engine.Tool{nil}
			case "one_null":
				initial.Tools = []*engine.Tool{tool, nil}
			}
			fn := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				t.Fatal("unexpected stream")
				return nil, nil
			}
			engine.SetDefaultStreamFn(nil)
			options := engine.AgentOptions{InitialState: initial}
			if input.Stream == "default" {
				engine.SetDefaultStreamFn(fn)
			}
			if input.Stream == "explicit" {
				options.StreamFn = fn
			}
			var problem, state any
			func() {
				defer func() {
					if failure := recover(); failure != nil {
						problem = fmt.Sprint("panic: ", failure)
					}
				}()
				agent, err := engine.NewAgent(options)
				if err != nil {
					problem = err.Error()
					return
				}
				snapshot := agent.State()
				names := []string{}
				for _, tool := range snapshot.Tools.Values() {
					names = append(names, tool.Name)
				}
				state = map[string]any{"messages": snapshot.Messages.Values(), "tools": names, "model": snapshot.Model}
			}()
			assertStateListJSON(t, map[string]any{"error": problem, "state": state}, tc.Expected)
		})
	}
	engine.SetDefaultStreamFn(nil)
	shape := func(m *ai.Message) any {
		if m == nil {
			return nil
		}
		return map[string]any{"role": m.Role, "timestamp": m.Timestamp, "error": m.ErrorMessage}
	}
	values := func(messages []*ai.Message) []any {
		result := []any{}
		for _, m := range messages {
			result = append(result, shape(m))
		}
		return result
	}
	list := func(pattern string, start int64) []*ai.Message {
		result := []*ai.Message{}
		if pattern == "empty" {
			return result
		}
		for index, role := range strings.Split(pattern, "_") {
			var m *ai.Message
			if role != "null" {
				m = message(role, start+int64(index))
			}
			result = append(result, m)
		}
		return result
	}
	for _, tc := range fixture.Queues {
		t.Run("queue/"+tc.Input.Steering+"/"+tc.Input.Followup+"/"+tc.Input.SteeringMode+"/"+tc.Input.FollowUpMode+"/"+tc.Input.Operation, func(t *testing.T) {
			input := tc.Input
			events, requests := []any{}, []any{}
			agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: model, Messages: []*ai.Message{message("assistant", 1)}}, AgentConfig: engine.AgentConfig{
				SteeringMode: input.SteeringMode, FollowUpMode: input.FollowUpMode, Config: engine.Config{Now: func() int64 { return 1000 }, ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }},
				StreamFn: func(_ context.Context, _ json.RawMessage, current ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
					requests = append(requests, values(engine.MessagePointers(current.Messages())))
					return loopListResponse(19+len(requests), "stop"), nil
				},
			}})
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
			for _, m := range list(input.Steering, 10) {
				agent.Steer(m)
			}
			for _, m := range list(input.Followup, 30) {
				agent.FollowUp(m)
			}
			observe := func() any {
				state := agent.State()
				return map[string]any{"peek": values(agent.PeekQueuedMessages()), "queued": agent.HasQueuedMessages(), "history": values(state.Messages.Values()), "error": state.ErrorMessage, "streaming": state.IsStreaming, "signal": agent.Signal() != nil, "pending": state.PendingToolCalls.Values()}
			}
			before := observe()
			var failure any
			var runErr error
			switch input.Operation {
			case "prompt":
				runErr = agent.Prompt(context.Background(), message("user", 2))
			case "continue":
				runErr = agent.Continue(context.Background())
			}
			if runErr != nil {
				failure = runErr.Error()
			}
			if err := agent.WaitForIdle(context.Background()); err != nil {
				t.Fatal(err)
			}
			assertStateListJSON(t, map[string]any{"before": before, "failure": failure, "after": observe(), "events": events, "requests": requests}, tc.Expected)
		})
	}
}
