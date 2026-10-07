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

func TestPiMessageReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-message-references.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Role, Phase, Through, StopReason string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	if fixtures.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixtures.Cases) != 94 {
		t.Fatal("unexpected reference fixture coverage")
	}
	for _, fixture := range fixtures.Cases {
		t.Run(fixture.Input.Mode+"/"+fixture.Input.Role+"/"+fixture.Input.StopReason+"/"+fixture.Input.Phase+"/"+fixture.Input.Through, func(t *testing.T) {
			input := fixture.Input
			events, requests := []json.RawMessage{}, []json.RawMessage{}
			identities := []map[string]bool{}
			capture := func(value any) json.RawMessage {
				raw, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				return raw
			}
			var retained *ai.Message
			mutated := false
			find := func(messages []*ai.Message) *ai.Message {
				for i := range messages {
					if messages[i].Role == input.Role {
						return messages[i]
					}
				}
				return nil
			}
			mutate := func(phase string, turn *engine.Turn) {
				if phase != input.Phase || mutated || retained == nil {
					return
				}
				target := retained
				switch input.Through {
				case "context":
					target = find(turn.Context.Messages.Values())
				case "newMessages":
					target = find(turn.NewMessages.Values())
				case "message":
					target = turn.Message
				case "toolResults":
					target = turn.ToolResults.Get(0)
				}
				target.Timestamp = 77
				mutated = true
			}
			tools := []*engine.Tool{}
			if input.Role == "toolResult" {
				tools = append(tools, &engine.Tool{Tool: ai.Tool{Name: "echo", Description: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}, Label: "echo", Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
					return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "result"}), Details: argumentRef(`{}`)}, nil
				}})
			}
			turns, requestCount, responseCount := 0, 0, 0
			config := engine.Config{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), Now: func() int64 { return 1000 }, ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }}
			config.TransformContext = func(_ context.Context, messages *engine.MessageList) (*engine.MessageList, error) {
				mutate("transform", nil)
				return messages, nil
			}
			config.ConvertToLLM = func(messages *engine.MessageList) (*engine.MessageList, error) {
				mutate("convert", nil)
				return messages, nil
			}
			config.GetAPIKey = func(string) (string, error) { mutate("key", nil); return "test", nil }
			config.GetSteeringMessages = func() (*engine.MessageList, error) { mutate("steering", nil); return nil, nil }
			config.FinishTurn = func(_ context.Context, turn *engine.Turn) (string, error) {
				turns++
				if turns == 1 {
					identity := map[string]bool{"context": find(turn.Context.Messages.Values()) == retained, "newMessages": find(turn.NewMessages.Values()) == retained}
					if input.Role == "assistant" {
						identity["message"] = turn.Message == retained
					}
					if input.Role == "toolResult" {
						identity["toolResults"] = turn.ToolResults.Get(0) == retained
					}
					identities = append(identities, identity)
					mutate("finish", turn)
					return "continue", nil
				}
				return "end", nil
			}
			config.PrepareNextTurn = func(*engine.Turn) (*engine.TurnUpdate, error) { mutate("next", nil); return nil, nil }
			config.PrepareRequest = func(context.Context, engine.Request) (*engine.TurnUpdate, error) {
				if requestCount > 0 {
					mutate("request", nil)
				}
				requestCount++
				return nil, nil
			}
			stream := func(_ context.Context, _ json.RawMessage, current ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				requests = append(requests, capture(current))
				message := ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"}), API: "test", Provider: "test", Model: "test", Usage: &ai.Usage{}, StopReason: input.StopReason, Timestamp: 100}
				if input.Role == "toolResult" && responseCount == 0 {
					message.Content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)})
					message.StopReason = "toolUse"
				}
				responseCount++
				result := ai.NewAssistantMessageEventStream()
				if message.StopReason == "error" || message.StopReason == "aborted" {
					result.Push(ai.AssistantMessageEvent{Type: "error", Reason: message.StopReason, Error: &message})
				} else {
					result.Push(ai.AssistantMessageEvent{Type: "done", Reason: message.StopReason, Message: &message})
				}
				return result, nil
			}
			var messages *engine.MessageList
			sink := func(event engine.Event) error {
				events = append(events, capture(event))
				if event.Type == "message_end" && event.Message.Role == input.Role && retained == nil {
					retained = event.Message
				}
				mutate(event.Type, nil)
				if event.Type == "agent_end" {
					messages = event.Messages
				}
				return nil
			}
			prompt := &ai.Message{Role: "user", Content: ai.TextContent("hello"), Timestamp: 1}
			var stateMessages *engine.MessageList
			var err error
			if input.Mode == "loop" {
				messages, err = engine.Run(context.Background(), engine.NewList([]*ai.Message{prompt}...), engine.Context{Messages: engine.NewList([]*ai.Message{}...), Tools: engine.NewList(tools...)}, config, sink, stream)
			} else {
				agent, createErr := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: config.Model, Tools: tools}, AgentConfig: engine.AgentConfig{Config: config, StreamFn: stream, PrepareNextTurn: func(context.Context) (*engine.TurnUpdate, error) {
					mutate("next", nil)
					mutate("steering", nil)
					return nil, nil
				}}})
				if createErr != nil {
					t.Fatal(createErr)
				}
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return sink(event) }})
				err = agent.Prompt(context.Background(), prompt)
				stateMessages = agent.State().Messages
			}
			if err != nil {
				t.Fatal(err)
			}
			result := map[string]any{"events": events, "requests": requests, "identities": identities, "messages": messages, "retained": retained, "prompt": prompt}
			if input.Mode == "agent" {
				result["stateMessages"] = stateMessages
			}
			actual := capture(result)
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(fixture.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, fixture.Expected)
			}
		})
	}
}
