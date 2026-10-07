package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func systemContentMessage(t *testing.T, pattern, role string, stamp any) *ai.Message {
	t.Helper()
	fields := map[string]any{"role": role, "content": []any{}, "timestamp": stamp, "sections": map[string]any{"first": "section", "removed": nil}}
	if stamp == "missing" {
		delete(fields, "timestamp")
	}
	text := map[string]any{"type": "text", "text": "body"}
	switch pattern {
	case "null_slot":
		fields["content"] = []any{nil}
	case "text_null":
		fields["content"] = []any{text, nil}
	case "null_text":
		fields["content"] = []any{nil, text}
	case "text":
		fields["content"] = "body"
	case "null":
		fields["content"] = nil
	case "missing":
		delete(fields, "content")
	}
	raw, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	var message ai.Message
	if err := json.Unmarshal(raw, &message); err != nil {
		t.Fatal(err)
	}
	if pattern == "hole" {
		message.Content = ai.BlockReferences(nil)
	}
	if pattern == "hole_text" {
		message.Content = ai.BlockReferences(nil, &ai.ContentBlock{Type: "text", Text: "body"})
	}
	return &message
}

func systemReadFailure(value any) string {
	if err, ok := value.(error); ok {
		var wrapped *json.MarshalerError
		for errors.As(err, &wrapped) {
			err = wrapped.Err
		}
		return err.Error()
	}
	return fmt.Sprint(value)
}

func TestPiSystemContent(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-system-content.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Kind, Pattern, Role, Operation, History string
				Stamp                                   any
				Tools                                   bool
			}
			Expected json.RawMessage
		}
		Helpers []struct {
			Pattern, Operation string
			Expected           json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 201 || len(fixture.Helpers) != 36 {
		t.Fatal("unexpected system-content oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(fmt.Sprintf("%s/%s/%s/%s/%v/%t/%s", tc.Input.Kind, tc.Input.Pattern, tc.Input.Role, tc.Input.History, tc.Input.Stamp, tc.Input.Tools, tc.Input.Operation), func(t *testing.T) {
			input := tc.Input
			var message *ai.Message
			if input.Kind == "content" {
				message = systemContentMessage(t, input.Pattern, input.Role, 1)
			} else {
				message = systemContentMessage(t, "text", "system", input.Stamp)
			}
			if input.Tools {
				message.ToolsAdded = []ai.Tool{{Name: "echo", Description: "tool", Parameters: json.RawMessage(`{"type":"object"}`)}}
			}
			messages := []*ai.Message{message}
			if strings.HasPrefix(input.History, "prefix") {
				var stamp any = 2
				if input.History == "prefix_missing" {
					stamp = "missing"
				}
				if input.History == "prefix_null" {
					stamp = nil
				}
				messages = append([]*ai.Message{systemContentMessage(t, "text", "system", stamp)}, messages...)
			}
			if input.History == "suffix_value" {
				messages = append(messages, systemContentMessage(t, "text", "system", 3))
			}
			agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)}, AgentConfig: engine.AgentConfig{StreamFn: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				t.Fatal("unexpected provider call")
				return nil, nil
			}}})
			if err != nil {
				t.Fatal(err)
			}
			agent.SetMessages(messages)
			agent.Steer(&ai.Message{Role: "user", Content: ai.TextContent("queued"), Timestamp: 5})
			agent.FollowUp(&ai.Message{Role: "user", Content: ai.TextContent("later"), Timestamp: 6})
			var value, problem any
			func() {
				defer func() {
					if failure := recover(); failure != nil {
						if input.Operation != "prompt" {
							t.Errorf("%s panicked instead of returning error: %v", input.Operation, failure)
						}
						problem = systemReadFailure(failure)
					}
				}()
				switch input.Operation {
				case "prompt":
					value = agent.State().SystemPrompt()
				case "json":
					raw, err := json.Marshal(agent.State())
					if err != nil {
						problem = systemReadFailure(err)
						return
					}
					var decoded map[string]any
					if err := json.Unmarshal(raw, &decoded); err != nil {
						t.Fatal(err)
					}
					value = decoded["systemPrompt"]
				case "reset":
					if err := agent.Reset(); err != nil {
						problem = systemReadFailure(err)
					}
				}
			}()
			// Snapshot before repairing the history: a failed reset must retain queues.
			expected := map[string]any{"value": value, "error": problem, "history": agent.State().Messages.Values(), "queued": agent.HasQueuedMessages(), "peek": agent.PeekQueuedMessages()}
			agent.SetMessages([]*ai.Message{{Role: "system", Content: ai.TextContent("repaired"), Timestamp: 9}})
			if err := agent.Reset(); err != nil {
				t.Fatal(err)
			}
			expected["recovered"] = map[string]any{"prompt": agent.State().SystemPrompt(), "queued": agent.HasQueuedMessages()}
			assertStateListJSON(t, expected, tc.Expected)
		})
	}
	for _, tc := range fixture.Helpers {
		t.Run("helper/"+tc.Pattern+"/"+tc.Operation, func(t *testing.T) {
			message := systemContentMessage(t, tc.Pattern, "system", 1)
			var value, problem any
			func() {
				defer func() {
					if failure := recover(); failure != nil {
						problem = systemReadFailure(failure)
					}
				}()
				switch tc.Operation {
				case "content":
					value = ai.ContentText(message.Content, "|")
				case "system":
					value = ai.GetSystemMessageText(*message)
				case "update":
					value = ai.RenderSystemMessageUpdate(*message)
				case "current":
					value = ai.GetCurrentSystemMessage([]ai.Message{*message})
				}
			}()
			assertStateListJSON(t, map[string]any{"value": value, "error": problem}, tc.Expected)
		})
	}
}
