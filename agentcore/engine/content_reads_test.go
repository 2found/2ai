package engine_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func contentReadShape(t *testing.T, message *ai.Message) any {
	t.Helper()
	raw, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	var reason any
	if message.StopReason != "" {
		reason = message.StopReason
	}
	content := map[string]any{}
	switch {
	case fields["content"] == nil:
		content["kind"] = "missing"
	case string(fields["content"]) == "null":
		content["kind"] = "null"
	case message.Content.Text != nil:
		content["kind"], content["text"] = "text", *message.Content.Text
	default:
		keys, values := []string{}, []any{}
		for index, block := range message.Content.Blocks.Values() {
			var value any
			if block != nil {
				keys = append(keys, strconv.Itoa(index))
				raw, err := json.Marshal(block)
				if err != nil {
					t.Fatal(err)
				}
				if string(raw) != "null" {
					value = block.Type
				}
			}
			values = append(values, value)
		}
		content["kind"], content["length"], content["keys"], content["values"] = "array", message.Content.Blocks.Len(), keys, values
	}
	return map[string]any{"role": message.Role, "reason": reason, "error": message.ErrorMessage, "content": content}
}

func TestPiContentReads(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-content-reads.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Pattern, Reason, Stage string }
			Expected json.RawMessage
		}
		ProxyCases []struct {
			Input struct {
				Index        int
				Kind, Reason string
			}
			Body     string
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 270 || len(fixture.ProxyCases) != 30 {
		t.Fatal("unexpected content-read oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Mode+"/"+tc.Input.Pattern+"/"+tc.Input.Reason+"/"+tc.Input.Stage, func(t *testing.T) {
			input := tc.Input
			events, executed := []any{}, []string{}
			requests, finished, changed := 0, 0, false
			call := &ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)}
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockReferences(call), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: input.Reason, Timestamp: 20}
			null := new(ai.ContentBlock)
			if err := json.Unmarshal([]byte("null"), null); err != nil {
				t.Fatal(err)
			}
			apply := func(message *ai.Message) {
				changed = true
				switch input.Pattern {
				case "empty":
					message.Content = ai.BlockReferences()
				case "hole":
					message.Content = ai.BlockReferences(nil)
				case "null_slot":
					message.Content = ai.BlockReferences(null)
				case "hole_call":
					message.Content = ai.BlockReferences(nil, call)
				case "call_null":
					message.Content = ai.BlockReferences(call, null)
				case "null_call":
					message.Content = ai.BlockReferences(null, call)
				case "text":
					message.Content = ai.TextContent("plain")
				case "null":
					message.Content = ai.MessageContent{}
				case "missing":
					raw, err := json.Marshal(message)
					if err != nil {
						t.Fatal(err)
					}
					var fields map[string]json.RawMessage
					if err := json.Unmarshal(raw, &fields); err != nil {
						t.Fatal(err)
					}
					delete(fields, "content")
					raw, err = json.Marshal(fields)
					if err != nil {
						t.Fatal(err)
					}
					if err := json.Unmarshal(raw, message); err != nil {
						t.Fatal(err)
					}
				}
			}
			shapeList := func(messages *engine.MessageList) []any {
				values := []any{}
				for _, message := range messages.Values() {
					values = append(values, contentReadShape(t, message))
				}
				return values
			}
			emit := func(event engine.Event) error {
				if !changed && event.Type == input.Stage && event.Message != nil && event.Message.Role == "assistant" {
					apply(event.Message)
				}
				value := map[string]any{"type": event.Type}
				if event.Message != nil {
					value["message"] = contentReadShape(t, event.Message)
				}
				if event.ToolName != "" {
					value["tool"] = event.ToolName
				}
				if event.Messages != nil {
					value["messages"] = shapeList(event.Messages)
				}
				events = append(events, value)
				return nil
			}
			tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Description: "tool", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}, Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
				executed = append(executed, "echo")
				return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "ok"}), Details: engine.NewObject()}, nil
			}}
			stream := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				requests++
				s := ai.NewAssistantMessageEventStream()
				s.Push(ai.AssistantMessageEvent{Type: "done", Reason: input.Reason, Message: assistant})
				return s, nil
			}
			config := engine.Config{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), ToolExecution: "sequential", Now: func() int64 { return 1000 },
				ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil },
				FinishTurn:   func(context.Context, *engine.Turn) (string, error) { finished++; return "end", nil },
			}
			if input.Stage == "initial" {
				apply(assistant)
			}
			var problem, state any
			func() {
				defer func() {
					if value := recover(); value != nil {
						problem = fmt.Sprint("panic: ", value)
					}
				}()
				prompt := &ai.Message{Role: "user", Content: ai.TextContent("go"), Timestamp: 1}
				if input.Mode == "loop" {
					_, err := engine.Run(context.Background(), engine.NewList(prompt), engine.Context{Messages: engine.NewList[*ai.Message](), Tools: engine.NewList(tool)}, config, emit, stream)
					if err != nil {
						problem = err.Error()
					}
				} else {
					agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: config.Model, Tools: []*engine.Tool{tool}}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: stream}})
					if err != nil {
						t.Fatal(err)
					}
					agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return emit(event) }})
					if err := agent.Prompt(context.Background(), prompt); err != nil {
						problem = err.Error()
						return
					}
					if err := agent.WaitForIdle(context.Background()); err != nil {
						t.Fatal(err)
					}
					snapshot := agent.State()
					state = map[string]any{"messages": shapeList(snapshot.Messages), "error": snapshot.ErrorMessage, "streaming": snapshot.IsStreaming, "pending": snapshot.PendingToolCalls.Values(), "signal": agent.Signal() != nil}
				}
			}()
			assertStateListJSON(t, map[string]any{"error": problem, "state": state, "events": events, "executed": executed, "requests": requests, "finished": finished}, tc.Expected)
		})
	}
	for _, tc := range fixture.ProxyCases {
		t.Run(fmt.Sprintf("proxy/%d/%s/%s", tc.Input.Index, tc.Input.Kind, tc.Input.Reason), func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, tc.Body)
			}))
			defer server.Close()
			events, executed := []string{}, 0
			tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Description: "tool", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}, Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
				executed++
				return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "ok"}), Details: engine.NewObject()}, nil
			}}
			config := engine.Config{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), Now: func() int64 { return 1000 },
				ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil },
				FinishTurn:   func(context.Context, *engine.Turn) (string, error) { return "end", nil },
			}
			messages, err := engine.Run(context.Background(), engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("go"), Timestamp: 1}), engine.Context{Tools: engine.NewList(tool)}, config, func(event engine.Event) error {
				events = append(events, event.Type)
				return nil
			}, func(ctx context.Context, model json.RawMessage, transcript ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				return engine.StreamProxy(ctx, model, transcript, engine.ProxyStreamOptions{ProxyURL: server.URL, AuthToken: "test", Client: server.Client(), Now: config.Now}), nil
			})
			var problem, result any
			if err != nil {
				problem = err.Error()
			} else {
				values := []any{}
				for _, message := range messages.Values() {
					values = append(values, contentReadShape(t, message))
				}
				result = values
			}
			assertStateListJSON(t, map[string]any{"error": problem, "result": result, "events": events, "executed": executed, "requests": requests.Load()}, tc.Expected)
		})
	}
}
