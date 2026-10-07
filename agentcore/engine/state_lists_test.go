package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"sync"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

type stateListInput struct {
	Subject, Phase, Action string
	Partial                bool
}
type stateListFixture struct {
	UpstreamCommit string
	Cases          []struct {
		Input    stateListInput
		Expected json.RawMessage
	}
	Runs []struct {
		Input    stateListInput
		Expected json.RawMessage
	}
}

func stateListFixtures(t *testing.T) stateListFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-state-lists.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture stateListFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 28 || len(fixture.Runs) != 90 {
		t.Fatal("unexpected state-list coverage")
	}
	return fixture
}

func listUser(text string) *ai.Message {
	return &ai.Message{Role: "user", Content: ai.TextContent(text), Timestamp: 1}
}
func listTool(name string) *engine.Tool {
	return &engine.Tool{Tool: ai.Tool{Name: name, Description: name, Parameters: json.RawMessage(`{"type":"object"}`)}, Label: name}
}

func listAgent(t *testing.T, stream engine.StreamFn) (*engine.Agent, []*ai.Message, []*engine.Tool) {
	t.Helper()
	messages := []*ai.Message{{Role: "system", Content: ai.TextContent("policy"), Timestamp: 0}, listUser("original")}
	tools := []*engine.Tool{listTool("original")}
	if stream == nil {
		stream = listStream
	}
	agent, err := engine.NewAgent(engine.AgentOptions{
		InitialState: engine.InitialState{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), Messages: messages, Tools: tools},
		AgentConfig: engine.AgentConfig{StreamFn: stream, Config: engine.Config{
			Now:          func() int64 { return 1000 },
			ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil },
			FinishTurn:   func(context.Context, *engine.Turn) (string, error) { return "end", nil },
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return agent, messages, tools
}

func listStream(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
	return listStreamEvents(false), nil
}

func listStreamEvents(partial bool) *ai.AssistantMessageEventStream {
	s := ai.NewAssistantMessageEventStream()
	message := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"}), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: "stop", Timestamp: 2}
	if partial {
		s.Push(ai.AssistantMessageEvent{Type: "start", Partial: message})
		s.Push(ai.AssistantMessageEvent{Type: "text_delta", ContentIndex: 0, Delta: "done", Partial: message})
	}
	s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
	return s
}

func assertStateListJSON(t *testing.T, value any, expected json.RawMessage) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var got, want any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(expected, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Go: %s\nPi: %s", raw, expected)
	}
}

func checkStateList[T comparable](t *testing.T, agent *engine.Agent, initial []T, index int, current func() *engine.List[T], assign func(*engine.List[T]), replacement func() T, mutate func(T), phase string, expected json.RawMessage) {
	t.Helper()
	retained := current()
	entry := initial[index]
	probe := replacement()
	initial[index] = probe
	initialSame := retained.Get(index) == probe
	initial[index] = entry
	switch phase {
	case "caller_replace":
		initial[index] = replacement()
	case "entry_replace":
		retained.Set(index, replacement())
	case "append":
		retained.Append(replacement())
	case "nested_edit":
		mutate(retained.Get(index))
	case "delete":
		retained.Delete(index)
	case "shrink":
		retained.SetLength(0)
	case "grow":
		retained.SetLength(retained.Len() + 3)
	case "assign_empty":
		assign(engine.NewList[T]())
	case "assign_new", "retained_after_assign":
		assign(engine.NewList(replacement()))
	case "assign_same":
		assign(retained)
	case "assign_sparse":
		retained.Delete(index)
		assign(retained)
	case "reset":
		if err := agent.Reset(); err != nil {
			t.Fatal(err)
		}
	}
	if phase == "retained_after_assign" {
		retained.Append(replacement())
	}
	latest := current()
	assertStateListJSON(t, map[string]any{
		"initialSame": initialSame, "sameList": latest == retained,
		"sameEntry": retained.Get(index) == entry, "retained": retained,
		"current": latest, "initial": initial, "retainedKeys": retained.Keys(), "currentKeys": latest.Keys(),
	}, expected)
}

func TestPiStateLists(t *testing.T) {
	for _, tc := range stateListFixtures(t).Cases {
		t.Run(tc.Input.Subject+"/"+tc.Input.Phase, func(t *testing.T) {
			agent, messages, tools := listAgent(t, nil)
			if tc.Input.Subject == "messages" {
				checkStateList(t, agent, messages, 1, func() *engine.MessageList { return agent.State().Messages }, agent.SetMessageList,
					func() *ai.Message { return listUser("replacement") }, func(m *ai.Message) { m.Timestamp = 77 }, tc.Input.Phase, tc.Expected)
			} else {
				checkStateList(t, agent, tools, 0, func() *engine.ToolList { return agent.State().Tools }, agent.SetToolList,
					func() *engine.Tool { return listTool("replacement") }, func(tool *engine.Tool) { tool.Name = "changed" }, tc.Input.Phase, tc.Expected)
			}
		})
	}
}

func TestPiStateListsDuringRun(t *testing.T) {
	for _, tc := range stateListFixtures(t).Runs {
		t.Run(tc.Input.Subject+"/"+tc.Input.Phase+"/"+tc.Input.Action, func(t *testing.T) {
			requests := []json.RawMessage{}
			agent, _, _ := listAgent(t, func(ctx context.Context, model json.RawMessage, current ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
				raw, err := json.Marshal(current)
				if err != nil {
					return nil, err
				}
				requests = append(requests, raw)
				return listStreamEvents(tc.Input.Partial), nil
			})
			retainedMessages, retainedTools := agent.State().Messages, agent.State().Tools
			edit := func() {
				if tc.Input.Subject == "messages" {
					switch tc.Input.Action {
					case "append":
						retainedMessages.Append(listUser("edited"))
					case "replace":
						retainedMessages.Set(1, listUser("edited"))
					case "nested":
						retainedMessages.Get(1).Timestamp = 77
					}
				} else {
					switch tc.Input.Action {
					case "append":
						retainedTools.Append(listTool("edited"))
					case "replace":
						retainedTools.Set(0, listTool("edited"))
					case "nested":
						retainedTools.Get(0).Name = "edited"
					}
				}
			}
			if tc.Input.Phase == "before" {
				edit()
			}
			observations := []map[string]any{}
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
				at := ""
				if event.Type == "agent_start" {
					at = "agent_start"
				}
				if event.Type == "message_start" || event.Type == "message_end" {
					at = "assistant"
					if event.Message.Role == "user" {
						at = "user"
					}
					if event.Type == "message_start" {
						at += "_start"
					} else {
						at += "_end"
					}
				}
				if event.Type == "message_update" {
					at = "assistant_update"
				}
				if tc.Input.Phase == at {
					edit()
				}
				if event.Type == "message_start" || event.Type == "message_update" {
					streaming := agent.State().StreamingMessage
					same := streaming == event.Message
					streaming.Timestamp = 92
					if event.Message.Role == "user" {
						streaming.Timestamp = 91
					}
					if event.Type == "message_update" {
						streaming.Timestamp = 93
					}
					observations = append(observations, map[string]any{"stage": at, "same": same, "eventTimestamp": event.Message.Timestamp})
				}
				if event.Type == "message_end" {
					messages := agent.State().Messages
					observations = append(observations, map[string]any{"stage": at, "same": messages.Get(messages.Len()-1) == event.Message})
				}
				return nil
			}})
			if err := agent.Prompt(context.Background(), listUser("prompt")); err != nil {
				t.Fatal(err)
			}
			if tc.Input.Phase == "after" {
				edit()
			}
			var retained, current any = retainedMessages, agent.State().Messages
			sameList := retainedMessages == agent.State().Messages
			if tc.Input.Subject == "tools" {
				retained, current = retainedTools, agent.State().Tools
				sameList = retainedTools == agent.State().Tools
			}
			assertStateListJSON(t, map[string]any{"requests": requests, "observations": observations, "retained": retained, "current": current, "sameList": sameList, "streamingCleared": agent.State().StreamingMessage == nil}, tc.Expected)
		})
	}
}

func TestStateListConcurrentContainerOperations(t *testing.T) {
	list := engine.NewList[*ai.Message]()
	var workers sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		workers.Add(2)
		go func() {
			defer workers.Done()
			for i := 0; i < 80; i++ {
				list.Append(listUser("immutable"))
			}
		}()
		go func() {
			defer workers.Done()
			for i := 0; i < 80; i++ {
				_ = list.Get(list.Len() - 1)
				copied := list.Clone()
				values := copied.Values()
				if len(values) != copied.Len() || len(copied.Keys()) != len(values) {
					t.Error("inconsistent detached collection")
				}
				if _, err := json.Marshal(copied); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	workers.Wait()
	if list.Len() != 320 || len(list.Keys()) != 320 {
		t.Fatalf("lost appends: length=%d keys=%d", list.Len(), len(list.Keys()))
	}
}

func TestStateListSparseCloneAndLazyPrompt(t *testing.T) {
	agent, _, _ := listAgent(t, nil)
	list := agent.State().Messages
	list.Delete(1)
	list.SetLength(5)
	// Reading the live collections must not eagerly invoke systemPrompt's
	// replay getter on incomplete/sparse history.
	if agent.State().Messages != list || list.Has(1) || list.Len() != 5 {
		t.Fatal("lost live sparse list")
	}
	copied := list.Clone()
	if copied == list || copied.Get(0) != list.Get(0) || copied.Has(1) || copied.Len() != 5 {
		t.Fatal("clone changed sparse shape or item identity")
	}
	copied.Set(1, listUser("replacement"))
	if list.Has(1) {
		t.Fatal("clone retained outer storage")
	}
	const maxIndex = 4294967294
	copied.Set(maxIndex, listUser("sparse tail"))
	if copied.Len() != maxIndex+1 || !copied.Has(maxIndex) || len(copied.Keys()) != 3 {
		t.Fatal("distant index lost sparse representation")
	}
	copied.SetLength(1)
	if copied.Has(maxIndex) || copied.Len() != 1 {
		t.Fatal("shrink did not remove distant entry")
	}
	// Native slice input and JSON input both contain real null entries,
	// distinguishable from holes through Has.
	explicitNull := engine.NewList[*ai.Message](nil)
	if !explicitNull.Has(0) || explicitNull.Get(0) != nil {
		t.Fatal("null entry became a hole")
	}
}
