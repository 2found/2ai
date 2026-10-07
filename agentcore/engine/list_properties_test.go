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

func TestPiListConstructorRecovery(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-list-properties.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Agents []struct {
			Subject  string
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Agents) != 2 {
		t.Fatal("unexpected admission recovery coverage")
	}
	for _, tc := range fixture.Agents {
		t.Run(tc.Subject, func(t *testing.T) {
			events := []string{}
			requests := 0
			agent, err := engine.NewAgent(engine.AgentOptions{
				InitialState: engine.InitialState{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)},
				AgentConfig: engine.AgentConfig{Config: engine.Config{Now: func() int64 { return 1000 }}, StreamFn: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
					requests++
					return loopListResponse(1, "stop"), nil
				}},
			})
			if err != nil {
				t.Fatal(err)
			}
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { events = append(events, event.Type); return nil }})
			state := agent.State()
			if tc.Subject == "messages" {
				state.Messages.SetProperty("constructor", nil)
			} else {
				state.Tools.SetProperty("constructor", nil)
			}
			observe := func() map[string]any {
				current := agent.State()
				messages := []any{}
				for _, m := range current.Messages.Values() {
					messages = append(messages, map[string]any{"role": m.Role, "timestamp": m.Timestamp, "error": m.ErrorMessage})
				}
				return map[string]any{"events": append([]string{}, events...), "requests": requests, "streaming": current.IsStreaming, "error": current.ErrorMessage, "messages": messages}
			}
			if err := agent.Prompt(context.Background(), loopListUser(1)); err != nil {
				t.Fatal(err)
			}
			if err := agent.WaitForIdle(context.Background()); err != nil {
				t.Fatal(err)
			}
			failed := observe()
			if tc.Subject == "messages" {
				state.Messages.DeleteProperty("constructor")
			} else {
				state.Tools.DeleteProperty("constructor")
			}
			if err := agent.Prompt(context.Background(), loopListUser(1)); err != nil {
				t.Fatal(err)
			}
			if err := agent.WaitForIdle(context.Background()); err != nil {
				t.Fatal(err)
			}
			assertStateListJSON(t, map[string]any{"failed": failed, "recovered": observe()}, tc.Expected)
		})
	}
}

func TestPiListProperties(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-list-properties.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Names          []string
		Cases          []struct {
			Phase    string
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 13 {
		t.Fatal("unexpected array-property oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Phase, func(t *testing.T) {
			list := engine.NewList[any]("initial")
			for _, name := range fixture.Names {
				list.SetProperty(name, name)
			}
			switch tc.Phase {
			case "overwrite":
				list.SetProperty("a", "edited")
			case "index_overwrite":
				list.SetProperty("1", "edited")
			case "delete_reinsert":
				list.DeleteProperty("-1")
				list.SetProperty("-1", "edited")
			case "delete_index", "decode_delete":
				list.DeleteProperty("1")
			case "shrink":
				list.SetLength(1)
			case "grow", "decode_grow":
				list.SetLength(5)
			case "clone_default":
				list.DeleteProperty("constructor")
			case "clone_object":
				list.SetProperty("constructor", map[string]any{})
			case "clone_undefined":
				list.SetProperty("constructor", engine.Undefined)
			}
			if strings.HasPrefix(tc.Phase, "clone") {
				var failure any
				func() { defer func() { failure = recover() }(); list = list.Clone() }()
				if failure != nil {
					assertStateListJSON(t, map[string]any{"error": fmt.Sprint(failure)}, tc.Expected)
					return
				}
			}
			if strings.HasPrefix(tc.Phase, "decode_") {
				raw, err := json.Marshal(list)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(raw, list); err != nil {
					t.Fatal(err)
				}
			}
			lookups := map[string]any{}
			for _, name := range append(append([]string{"0", "3"}, fixture.Names...), "absent") {
				value, found := list.GetProperty(name)
				lookups[name] = map[string]any{"found": found, "value": value}
			}
			wire, err := json.Marshal(list)
			if err != nil {
				t.Fatal(err)
			}
			assertStateListJSON(t, map[string]any{"length": list.Len(), "keys": list.PropertyKeys(), "indexKeys": list.Keys(), "wire": string(wire), "lookups": lookups}, tc.Expected)
		})
	}
}
