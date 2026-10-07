package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

// Exercise the exported Go providers over HTTP, so a passing parser test alone
// cannot mask a provider that still forwards malformed argument strings.
func TestPiProviderToolArguments(t *testing.T) {
	cases := []struct{ name, input, expected string }{
		{"empty", "", `{}`},
		{"unfinished", `{"city":"Hue`, `{"city":"Hue"}`},
		{"invalid escape", `{"path":"C:\work\file"}`, `{"path":"C:\\work\file"}`},
		{"raw control", "{\"text\":\"one\ntwo\"}", `{"text":"one\ntwo"}`},
		{"unparseable", `not-json`, `{}`},
		{"nested partial", `{"data":[1,2,{"text":"partial`, `{"data":[1,2,{"text":"partial"}]}`},
	}
	providers := []struct {
		api string
		new func(string) protocol.LLMProvider
	}{
		{"openai", func(base string) protocol.LLMProvider { return NewOpenAIProvider("test", base, DefaultCompat()) }},
		{"anthropic", func(base string) protocol.LLMProvider { return NewAnthropicProvider("test", base) }},
		{"responses", func(base string) protocol.LLMProvider { return NewOpenAIResponsesProvider("test", base) }},
		{"codex", func(base string) protocol.LLMProvider { p := NewCodexProvider(); p.BaseURL = base; return p }},
	}
	for _, provider := range providers {
		for _, test := range cases {
			for _, method := range []string{"chat", "stream"} {
				// Anthropic's nonstreaming response already contains a JSON
				// object for input; only its SSE API carries argument strings.
				if provider.api == "anthropic" && method == "chat" {
					continue
				}
				t.Run(provider.api+"/"+test.name+"/"+method, func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						w.Header().Set("Content-Type", "text/event-stream")
						write := func(event any) {
							data, err := json.Marshal(event)
							if err != nil {
								t.Error(err)
								return
							}
							fmt.Fprintf(w, "data: %s\n\n", data)
						}
						switch provider.api {
						case "openai":
							write(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": 0, "id": "call", "function": map[string]any{"name": "lookup", "arguments": test.input}}}}, "finish_reason": "tool_calls"}}})
							fmt.Fprint(w, "data: [DONE]\n\n")
						case "anthropic":
							write(map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "tool_use", "id": "call", "name": "lookup", "input": map[string]any{}}})
							write(map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "input_json_delta", "partial_json": test.input}})
							write(map[string]any{"type": "content_block_stop", "index": 0})
							write(map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use"}})
							write(map[string]any{"type": "message_stop"})
						default:
							write(map[string]any{"type": "response.output_item.done", "item": map[string]any{"type": "function_call", "call_id": "call", "name": "lookup", "arguments": test.input}})
							write(map[string]any{"type": "response.completed", "response": map[string]any{"id": "response", "status": "completed"}})
						}
					}))
					defer server.Close()
					client := provider.new(server.URL)
					ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
					defer cancel()
					req := protocol.ChatRequest{Model: "test-model", MaxTokens: 128, Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "look up"}}}
					var calls []protocol.ToolCall
					if method == "chat" {
						response, err := client.Chat(ctx, req)
						if err != nil {
							t.Fatal(err)
						}
						calls = response.Message.ToolCalls
					} else {
						stream, err := client.Stream(ctx, req)
						if err != nil {
							t.Fatal(err)
						}
						for delta := range stream {
							if delta.Err != nil {
								t.Fatal(delta.Err)
							}
							if delta.ToolCall != nil {
								calls = append(calls, *delta.ToolCall)
							}
						}
					}
					if len(calls) != 1 || calls[0].ID != "call" || calls[0].Name != "lookup" {
						t.Fatalf("unexpected tool calls: %+v", calls)
					}
					assertPiJSON(t, json.RawMessage(test.expected), json.RawMessage(calls[0].Arguments))
				})
			}
		}
	}
}

func TestPiProviderNonstreamToolArguments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call","function":{"name":"lookup","arguments":"{\"city\":\"Hue"}}]},"finish_reason":"tool_calls"}]}`)
	}))
	defer server.Close()
	response, err := NewOpenAIProvider("test", server.URL, DefaultCompat()).Chat(context.Background(), protocol.ChatRequest{Model: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Message.ToolCalls) != 1 {
		t.Fatalf("missing tool call: %+v", response)
	}
	assertPiJSON(t, json.RawMessage(`{"city":"Hue"}`), json.RawMessage(response.Message.ToolCalls[0].Arguments))
}
