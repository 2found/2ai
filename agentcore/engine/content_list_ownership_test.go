package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strconv"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiContentListOwnership(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-content-list-ownership.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Phase, Operation string
			Expected         json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 32 {
		t.Fatal("unexpected content-list oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Phase+"/"+tc.Operation, func(t *testing.T) {
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "a"}, ai.ContentBlock{Type: "text", Text: "b"}), API: "test", Provider: "test", Model: "m", Usage: &ai.Usage{Input: 1}, StopReason: "stop", Timestamp: 1}
			original := assistant.Content
			retained := []*ai.Message{}
			mutate := func(phase string, message *ai.Message) {
				if phase != tc.Phase {
					return
				}
				mutateContentList(tc.Operation, &message.Content)
			}

			config := engine.Config{Model: json.RawMessage(`{"api":"test","provider":"test","id":"m"}`), ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }, FinishTurn: func(_ context.Context, turn *engine.Turn) (string, error) {
				mutate("finish", turn.Message)
				return "end", nil
			}}
			messages, err := engine.Run(context.Background(), engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("run")}), engine.Context{}, config, func(event engine.Event) error {
				if event.Message != nil && event.Message.Role == "assistant" {
					retained = append(retained, event.Message)
					mutate(event.Type, event.Message)
				}
				return nil
			}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				s := ai.NewAssistantMessageEventStream()
				s.Push(ai.AssistantMessageEvent{Type: "start", Partial: assistant})
				s.Push(ai.AssistantMessageEvent{Type: "text_delta", ContentIndex: 0, Delta: "a", Partial: assistant})
				s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: assistant})
				s.End()
				return s, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			observed := []map[string]any{}
			for _, message := range retained {
				observed = append(observed, contentListShape(message.Content))
			}
			actual, err := json.Marshal(map[string]any{"provider": contentListShape(assistant.Content), "result": contentListShape(messages.Get(messages.Len() - 1).Content), "original": contentListShape(original), "retained": observed})
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

func TestPiToolContentListOwnership(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-tool-content-list-ownership.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Phase, Operation string
			Expected         json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 16 {
		t.Fatal("unexpected tool content-list oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Phase+"/"+tc.Operation, func(t *testing.T) {
			original := ai.BlockContent(ai.ContentBlock{Type: "text", Text: "a"}, ai.ContentBlock{Type: "text", Text: "b"})
			produced := &engine.ToolResult{Content: original.Blocks, Details: engine.NewObject()}
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)}), StopReason: "toolUse"}
			retained := []*ai.Message{}
			tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Description: "test", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}, Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
				return produced, nil
			}}
			config := engine.Config{Model: json.RawMessage(`{"api":"test","provider":"test","id":"m"}`), ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }, FinishTurn: func(context.Context, *engine.Turn) (string, error) { return "end", nil }}
			messages, err := engine.Run(context.Background(), engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("run")}), engine.Context{Tools: engine.NewList(tool)}, config, func(event engine.Event) error {
				if event.Message != nil && event.Message.Role == "toolResult" {
					retained = append(retained, event.Message)
					if event.Type == tc.Phase {
						mutateContentList(tc.Operation, &event.Message.Content)
					}
				}
				return nil
			}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				stream := ai.NewAssistantMessageEventStream()
				stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: "toolUse", Message: assistant})
				stream.End()
				return stream, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			observed := []map[string]any{}
			for _, message := range retained {
				observed = append(observed, contentListShape(message.Content))
			}
			actual := map[string]any{"produced": contentListShape(ai.MessageContent{Blocks: produced.Content}), "result": contentListShape(messages.Get(messages.Len() - 1).Content), "original": contentListShape(original), "retained": observed}
			assertStateListJSON(t, actual, tc.Expected)
		})
	}
}

func mutateContentList(operation string, content *ai.MessageContent) {
	block := &ai.ContentBlock{Type: "text", Text: "changed"}
	switch operation {
	case "append":
		content.Blocks.Append(block, &ai.ContentBlock{Type: "text", Text: "tail"})
	case "pop":
		content.Blocks.Pop()
	case "truncate":
		content.Blocks.SetLength(0)
	case "extend":
		content.Blocks.SetLength(5)
	case "delete":
		content.Blocks.Delete(0)
	case "replace_entry":
		content.Blocks.Set(0, block)
	case "replace_list":
		*content = ai.BlockReferences(block)
	case "clear_append":
		content.Blocks.SetLength(0)
		content.Blocks.Append(block)
	}
}
func contentListShape(content ai.MessageContent) map[string]any {
	keys, values := []string{}, []any{}
	for i, block := range content.Blocks.Values() {
		if block == nil {
			values = append(values, nil)
		} else {
			keys = append(keys, strconv.Itoa(i))
			values = append(values, block.Text)
		}
	}
	return map[string]any{"length": content.Blocks.Len(), "keys": keys, "values": values}
}
