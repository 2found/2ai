package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiStreamAliases(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-stream-aliases.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Shape string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 8 {
		t.Fatal("unexpected stream alias coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Mode+"/"+tc.Input.Shape, func(t *testing.T) {
			call := &ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)}
			blocks := []*ai.ContentBlock{call}
			if tc.Input.Shape == "repeated" {
				blocks = append(blocks, call)
			}
			if tc.Input.Shape == "distinct" {
				copy := *call
				blocks = append(blocks, &copy)
			}
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockReferences(blocks...), Timestamp: 1, API: "test", Provider: "test", Model: "test", StopReason: "error", Usage: &ai.Usage{}}
			stream := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				s := ai.NewAssistantMessageEventStream()
				toolCall := call
				if tc.Input.Shape == "detached" {
					copy := *call
					toolCall = &copy
				}
				s.Push(ai.AssistantMessageEvent{Type: "start", Partial: assistant})
				s.Push(ai.AssistantMessageEvent{Type: "toolcall_end", ContentIndex: 0, ToolCall: toolCall, Partial: assistant})
				s.Push(ai.AssistantMessageEvent{Type: "error", Reason: "error", Error: assistant})
				s.End()
				return s, nil
			}
			events := []json.RawMessage{}
			aliases := []map[string]any{}
			var messages *engine.MessageList
			sink := func(event engine.Event) error {
				raw, err := json.Marshal(event)
				if err != nil {
					return err
				}
				events = append(events, raw)
				if event.Message != nil && event.Message.Role == "assistant" {
					blocks := event.Message.Content.Blocks.Values()
					indices := []int{}
					for _, block := range blocks {
						for i, original := range blocks {
							if original == block {
								indices = append(indices, i)
								break
							}
						}
					}
					identity := map[string]any{"type": event.Type, "blocks": indices}
					if update := event.AssistantMessageEvent; update != nil {
						identity["samePartial"] = event.Message == update.Partial
						matches := []bool{}
						for _, block := range blocks {
							matches = append(matches, block == update.ToolCall)
						}
						identity["toolCallMatches"] = matches
					}
					aliases = append(aliases, identity)
				}
				if event.Type == "agent_end" {
					messages = event.Messages
				}
				return nil
			}
			config := engine.Config{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), Now: func() int64 { return 1000 }, ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }}
			prompt := &ai.Message{Role: "user", Content: ai.TextContent("go"), Timestamp: 0}
			if tc.Input.Mode == "loop" {
				messages, err = engine.Run(context.Background(), engine.NewList([]*ai.Message{prompt}...), engine.Context{Messages: engine.NewList([]*ai.Message{}...), Tools: engine.NewList([]*engine.Tool{}...)}, config, sink, stream)
			} else {
				agent, createErr := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: config.Model}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: stream}})
				if createErr != nil {
					t.Fatal(createErr)
				}
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return sink(event) }})
				err = agent.Prompt(context.Background(), prompt)
			}
			if err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"events": events, "aliases": aliases, "messages": messages})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err = json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
