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

func TestPiSharedFailureUsage(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-failure-usage.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Scope, Phase, Action string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 90 {
		t.Fatal("unexpected failure-usage coverage")
	}
	create := func(t *testing.T) *engine.Agent {
		t.Helper()
		agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)}, AgentConfig: engine.AgentConfig{Config: engine.Config{
			Now: func() int64 { return 1000 },
			TransformContext: func(context.Context, *engine.MessageList) (*engine.MessageList, error) {
				return nil, errors.New("fail")
			},
		}, StreamFn: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
			t.Fatal("unexpected stream")
			return nil, nil
		}}})
		if err != nil {
			t.Fatal(err)
		}
		return agent
	}
	run := func(t *testing.T, agent *engine.Agent) {
		t.Helper()
		if err := agent.Prompt(context.Background(), loopListUser(1)); err != nil {
			t.Fatal(err)
		}
		if err := agent.WaitForIdle(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	var shared *ai.Usage
	probe := create(t)
	probe.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
		if event.Message != nil && event.Message.Role == "assistant" {
			shared = event.Message.Usage
		}
		return nil
	}})
	run(t, probe)
	if shared == nil {
		t.Fatal("missing failure usage")
	}
	baseline := *shared
	defer func() { *shared = baseline }()
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Scope+"/"+tc.Input.Phase+"/"+tc.Input.Action, func(t *testing.T) {
			*shared = baseline
			defer func() { *shared = baseline }()
			input := tc.Input
			events := []any{}
			type namedAgent struct {
				name  string
				agent *engine.Agent
			}
			agents := []namedAgent{}
			var first *ai.Message
			var retained *ai.Usage
			edited, nested := false, false
			refs := map[*ai.Usage]int{shared: 0}
			usage := func(value *ai.Usage) any {
				id, exists := refs[value]
				if !exists {
					id = len(refs)
					refs[value] = id
				}
				raw, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				return map[string]any{"ref": id, "value": json.RawMessage(raw)}
			}
			mutate := func() {
				if edited {
					return
				}
				edited = true
				switch input.Action {
				case "input":
					retained.Input = 7
				case "cost":
					retained.Cost.Total = 8
				case "replace", "detach_edit":
					replacement := baseline
					replacement.Input = 20
					first.Usage = &replacement
					if input.Action == "detach_edit" {
						retained.Output = 9
					}
				case "optional":
					cache, reasoning := float64(4), float64(5)
					retained.CacheWrite1h, retained.Reasoning = &cache, &reasoning
				}
			}
			var second *engine.Agent
			makeAgent := func(name string) *engine.Agent {
				agent := create(t)
				agents = append(agents, namedAgent{name, agent})
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
					var message *ai.Message
					if event.Message != nil && event.Message.Role == "assistant" {
						message = event.Message
					} else if event.Type == "agent_end" {
						message = event.Messages.Get(0)
					}
					if message == nil {
						return nil
					}
					if first == nil {
						first, retained = message, message.Usage
					}
					if name == "a" && !edited && event.Type == input.Phase {
						mutate()
						if input.Scope == "nested" {
							nested = true
							run(t, second)
						}
					}
					events = append(events, map[string]any{"name": name, "type": event.Type, "first": message == first, "usage": usage(message.Usage)})
					return nil
				}})
				return agent
			}
			firstAgent := makeAgent("a")
			second = firstAgent
			if input.Scope != "same" {
				second = makeAgent("b")
			}
			run(t, firstAgent)
			if input.Phase == "settled" {
				mutate()
			}
			if !nested {
				run(t, second)
			}
			run(t, makeAgent("c"))
			firstValue, retainedValue, sharedValue := usage(first.Usage), usage(retained), usage(shared)
			histories := []any{}
			for _, named := range agents {
				usages := []any{}
				for _, message := range named.agent.State().Messages.Values() {
					if message.Role == "assistant" {
						usages = append(usages, usage(message.Usage))
					}
				}
				histories = append(histories, map[string]any{"name": named.name, "usages": usages})
			}
			assertStateListJSON(t, map[string]any{"events": events, "first": firstValue, "retained": retainedValue, "shared": sharedValue, "histories": histories}, tc.Expected)
		})
	}
}
