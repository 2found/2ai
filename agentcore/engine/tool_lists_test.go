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

func TestPiToolLists(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-tool-lists.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Pattern, Stage, Mode, Reason string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 392 {
		t.Fatal("unexpected tool-list oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Pattern+"/"+tc.Input.Stage+"/"+tc.Input.Mode+"/"+tc.Input.Reason, func(t *testing.T) {
			input := tc.Input
			events, executed := []any{}, []string{}
			requests, changed := 0, false
			tool := func(name, mode string) *engine.Tool {
				return &engine.Tool{Tool: ai.Tool{Name: name, Description: "tool", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}, ExecutionMode: mode,
					Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
						executed = append(executed, name)
						return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: name}), Details: engine.NewObject()}, nil
					},
				}
			}
			tools := engine.NewList(tool("first", ""), tool("missing", ""))
			apply := func() {
				changed = true
				tools.SetLength(0)
				if input.Pattern == "constructor" || input.Pattern == "constructor_null" {
					var value *engine.Tool
					if input.Pattern == "constructor" {
						value = tool("first", "")
					}
					tools.Append(value)
					tools.SetProperty("constructor", nil)
					return
				}
				if input.Pattern == "empty" {
					return
				}
				for _, token := range strings.Split(input.Pattern, "_") {
					switch token {
					case "hole":
						tools.SetLength(tools.Len() + 1)
					case "null":
						tools.Append(nil)
					default:
						mode := ""
						if token == "seq" {
							mode = "sequential"
						}
						tools.Append(tool("first", mode))
					}
				}
			}
			calls := []*ai.ContentBlock{}
			for index, name := range []string{"first", "missing"} {
				calls = append(calls, &ai.ContentBlock{Type: "toolCall", ID: fmt.Sprint("c", index), Name: name, Arguments: json.RawMessage(`{}`)})
			}
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockReferences(calls...), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: input.Reason, Timestamp: 20}
			current := engine.Context{Messages: engine.NewList[*ai.Message](), Tools: tools}
			emit := func(event engine.Event) error {
				value := map[string]any{"type": event.Type}
				if event.ToolName != "" {
					value["tool"] = event.ToolName
				}
				if event.Message != nil {
					value["role"] = event.Message.Role
				}
				if event.Result != nil {
					value["isError"], value["text"] = event.IsError, event.Result.Content.Get(0).Text
				}
				events = append(events, value)
				if !changed && event.Type == input.Stage && (input.Stage != "message_end" || event.Message.Role == "assistant") {
					apply()
				}
				return nil
			}
			var problem, outcome any
			func() {
				defer func() {
					if value := recover(); value != nil {
						problem = fmt.Sprint("panic: ", value)
					}
				}()
				if strings.HasPrefix(input.Stage, "direct") {
					apply()
					index := 0
					if strings.HasSuffix(input.Stage, "_missing") {
						index = 1
					}
					selectedTools := tools
					if strings.Contains(input.Stage, "default") {
						selectedTools = nil
					}
					result, err := engine.RunToolCall(context.Background(), calls[index], selectedTools, assistant, &current, engine.ToolHooks{}, nil)
					if err != nil {
						problem = err.Error()
						return
					}
					outcome = map[string]any{"isError": result.IsError, "text": result.Result.Content.Get(0).Text}
				} else {
					if input.Stage == "initial" {
						apply()
					}
					config := engine.Config{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), ToolExecution: input.Mode, Now: func() int64 { return 1000 },
						ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil },
						FinishTurn:   func(context.Context, *engine.Turn) (string, error) { return "end", nil },
					}
					_, err := engine.Run(context.Background(), engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("go"), Timestamp: 1}), current, config, emit,
						func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
							requests++
							stream := ai.NewAssistantMessageEventStream()
							stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: input.Reason, Message: assistant})
							return stream, nil
						})
					if err != nil {
						problem = err.Error()
					}
				}
			}()
			assertStateListJSON(t, map[string]any{"error": problem, "outcome": outcome, "events": events, "executed": executed, "requests": requests}, tc.Expected)
		})
	}
}

func TestExplicitEmptyToolListOverridesContext(t *testing.T) {
	tool := &engine.Tool{Tool: ai.Tool{Name: "first", Parameters: json.RawMessage(`{"type":"object"}`)}, Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
		t.Fatal("explicit empty list fell back to context")
		return nil, nil
	}}
	current := &engine.Context{Tools: engine.NewList(tool)}
	result, err := engine.RunToolCall(context.Background(), &ai.ContentBlock{Name: "first", Arguments: json.RawMessage(`{}`)}, engine.NewList[*engine.Tool](), nil, current, engine.ToolHooks{}, nil)
	if err != nil || !result.IsError || result.Result.Content.Get(0).Text != "Tool first not found" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestSparseToolDeclarationDoesNotDensifyHoles(t *testing.T) {
	tools := engine.NewList[*engine.Tool]()
	tools.Set(1<<30, &engine.Tool{Tool: ai.Tool{Name: "last", Parameters: json.RawMessage(`{"type":"object"}`)}})
	_, err := engine.Run(context.Background(), engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("go")}), engine.Context{Tools: tools}, engine.Config{}, func(engine.Event) error {
		t.Fatal("invalid declaration emitted an event")
		return nil
	}, nil)
	if err == nil || err.Error() != "Type error" {
		t.Fatalf("sparse declaration: %v", err)
	}
}
