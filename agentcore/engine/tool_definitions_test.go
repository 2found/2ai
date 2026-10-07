package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiToolDefinitionReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-tool-definitions.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Phase string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	if fixtures.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixtures.Cases) != 67 {
		t.Fatal("unexpected tool definition oracle coverage")
	}
	for _, tc := range fixtures.Cases {
		t.Run(tc.Input.Mode+"/"+tc.Input.Phase, func(t *testing.T) {
			mode, phase := tc.Input.Mode, tc.Input.Phase
			var mu sync.Mutex
			executed := []map[string]string{}
			results := []any{}
			makeTool := func(label string) *engine.Tool {
				terminate := true
				return &engine.Tool{Tool: ai.Tool{Name: "echo", Description: label, Parameters: json.RawMessage(`{"type":"object"}`)}, Label: label,
					Execute: func(_ context.Context, id string, _ any, _ func(*engine.ToolResult)) (*engine.ToolResult, error) {
						mu.Lock()
						executed = append(executed, map[string]string{"id": id, "executor": label})
						mu.Unlock()
						return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: label}), Details: argumentRef(`{}`), Terminate: &terminate}, nil
					},
				}
			}
			activeTools := engine.NewList(makeTool("original"))
			retained := activeTools.Get(0)
			mutate := func(at string) {
				if phase == at+"_replace" {
					activeTools.Set(0, makeTool("replacement"))
				}
				if phase == at+"_grow_mutate" {
					for i := 0; i < 8; i++ {
						padding := makeTool("padding")
						padding.Name = "padding" + strconv.Itoa(i)
						activeTools.Append(padding)
					}
				}
				if phase == at+"_mutate" || phase == at+"_grow_mutate" {
					retained.Label, retained.Description, retained.Execute = "mutated", "mutated", makeTool("mutated").Execute
				}
			}
			prepared := false
			retained.PrepareArguments = func(args json.RawMessage) (json.RawMessage, error) {
				if !prepared {
					prepared = true
					mutate("prepare")
				}
				return args, nil
			}
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "first", Name: "echo", Arguments: json.RawMessage(`{}`)}), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: "toolUse", Timestamp: 1}
			if mode != "programmatic" {
				assistant.Content.Blocks.Append(&ai.ContentBlock{Type: "toolCall", ID: "second", Name: "echo", Arguments: json.RawMessage(`{}`)})
			}
			model := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
			config := engine.Config{Model: model, ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }, Now: func() int64 { return 1000 },
				ToolHooks: engine.ToolHooks{Before: func(_ context.Context, hook *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
					if hook.ToolCall.ID == "first" {
						mutate("before")
					} else {
						mutate("next_before")
					}
					return nil, nil
				}},
				PrepareRequest: func(_ context.Context, request engine.Request) (*engine.TurnUpdate, error) {
					activeTools = request.Context.Tools
					return nil, nil
				},
				FinishTurn: func(context.Context, *engine.Turn) (string, error) { return "end", nil },
			}
			stream := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				s := ai.NewAssistantMessageEventStream()
				s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "toolUse", Message: assistant})
				return s, nil
			}
			emit := func(event engine.Event) error {
				if event.Type == "message_end" && event.Message.Role == "toolResult" {
					results = append(results, event.Message)
				}
				return nil
			}
			if mode == "programmatic" {
				outcome, err := engine.RunToolCall(context.Background(), assistant.Content.Blocks.Get(0), activeTools, assistant, &engine.Context{Messages: engine.NewList([]*ai.Message{assistant}...), Tools: activeTools}, config.ToolHooks, nil)
				if err != nil {
					t.Fatal(err)
				}
				results = append(results, map[string]any{"toolCallId": outcome.ToolCall.ID, "content": outcome.Result.Content, "isError": outcome.IsError})
			} else if strings.HasPrefix(mode, "agent_") {
				config.ToolExecution = mode[strings.LastIndex(mode, "_")+1:]
				initialTools := activeTools
				if strings.Contains(mode, "_set_") {
					initialTools = nil
				}
				agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Tools: initialTools.Values(), Model: model}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: stream}})
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(mode, "_set_") {
					agent.SetToolList(activeTools)
				}
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return emit(event) }})
				if err := agent.Prompt(context.Background(), &ai.Message{Role: "user", Content: ai.TextContent("go")}); err != nil {
					t.Fatal(err)
				}
			} else {
				config.ToolExecution = mode
				if _, err := engine.Run(context.Background(), engine.NewList([]*ai.Message{{Role: "user", Content: ai.TextContent("go")}}...), engine.Context{Messages: engine.NewList([]*ai.Message{}...), Tools: activeTools}, config, emit, stream); err != nil {
					t.Fatal(err)
				}
			}
			sort.Slice(executed, func(i, j int) bool { return executed[i]["id"] < executed[j]["id"] })
			actual, err := json.Marshal(map[string]any{"executed": executed, "results": results, "retained": retained.Label, "slot": activeTools.Get(0).Label, "same": retained == activeTools.Get(0)})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
