package engine_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestRequestAdmissionOwnsPreparationAndStream(t *testing.T) {
	initial := json.RawMessage(`{"id":"initial"}`)
	selected := json.RawMessage(`{"id":"selected"}`)
	config := engine.Config{
		Options: map[string]any{"temperature": 0.4},
		PrepareRequest: func(context.Context, engine.Request) (*engine.TurnUpdate, error) {
			t.Error("prepared twice")
			return nil, nil
		},
		TransformContext: func(context.Context, *engine.MessageList) (*engine.MessageList, error) {
			t.Error("transformed twice")
			return nil, nil
		},
		ConvertToLLM: func(*engine.MessageList) (*engine.MessageList, error) { t.Error("converted twice"); return nil, nil },
		GetAPIKey:    func(string) (string, error) { t.Error("acquired key twice"); return "", nil },
		AdmitRequest: func(ctx context.Context, request engine.Request, options map[string]any) (*engine.RequestAdmission, error) {
			if string(request.Model) != string(initial) || request.ThinkingLevel != "off" || options["temperature"] != 0.4 || options["toolExecution"] != "parallel" {
				t.Fatal("admission missing loop request/options")
			}
			request.Model, request.ThinkingLevel = selected, "high"
			stream := ai.NewAssistantMessageEventStreamFor(ctx)
			message := &ai.Message{Role: "assistant", Model: "selected", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "complete"}), StopReason: "stop"}
			stream.Push(ai.AssistantMessageEvent{Type: "done", Message: message})
			stream.End()
			return &engine.RequestAdmission{Request: request, Stream: stream}, nil
		},
	}
	agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: initial}, AgentConfig: engine.AgentConfig{Config: config}})
	if err != nil {
		t.Fatal(err)
	}
	if err = agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatal(err)
	}
	state := agent.State()
	last := state.Messages.Get(state.Messages.Len() - 1)
	if last.Model != "selected" || last.ThinkingLevel == nil || *last.ThinkingLevel != "high" || last.StopReason != "stop" {
		t.Fatal("admitted reasoning not applied to final response", last)
	}
}

func TestRequestAdmissionRejectsIncompleteRequest(t *testing.T) {
	for _, mode := range []string{"nil", "context", "stream", "model"} {
		t.Run(mode, func(t *testing.T) {
			config := engine.Config{Model: json.RawMessage(`{"id":"initial"}`), AdmitRequest: func(_ context.Context, request engine.Request, _ map[string]any) (*engine.RequestAdmission, error) {
				result := &engine.RequestAdmission{Request: request, Stream: ai.NewAssistantMessageEventStream()}
				switch mode {
				case "nil":
					return nil, nil
				case "context":
					result.Request.Context = nil
				case "stream":
					result.Stream = nil
				case "model":
					result.Request.Model = nil
				}
				return result, nil
			}}
			_, err := engine.Run(context.Background(), nil, engine.Context{}, config, func(event engine.Event) error {
				if event.Type == "agent_end" {
					t.Error("invalid admission completed a run")
				}
				return nil
			}, nil)
			if err == nil || !strings.Contains(err.Error(), "invalid prepared request admission") {
				t.Fatal("invalid admission accepted", err)
			}
		})
	}
}
