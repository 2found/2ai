package subagent_test

import (
	"context"
	"encoding/json"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

// echoTool is a trivial always-succeeding tool. These tests are about who runs,
// with what scope, and what comes back — not about what a tool does.
type echoTool struct{ name string }

func (t *echoTool) Name() string { return t.name }

func (t *echoTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:        t.name,
		Description: "echo the given text",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"text": map[string]any{"type": "string"}},
			"required":   []string{"text"},
		},
	}
}

func (t *echoTool) Run(_ context.Context, args string) (string, error) {
	var in struct {
		Text string `json:"text"`
	}
	_ = json.Unmarshal([]byte(args), &in)
	return in.Text, nil
}

// AssistantToolCall builds a scripted assistant turn that issues one tool call.
func AssistantToolCall(id, name, args string) ai.Message {
	return nativeCall(id, name, args)
}
