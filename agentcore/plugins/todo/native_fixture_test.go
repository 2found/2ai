package todo_test

import (
	"encoding/json"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/todo"
	"github.com/2found/2ai/ai"
)

func nativePlanCall(id, name, args string) ai.Message {
	return ai.Message{Role: "assistant", Model: "test", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: id, Name: name, Arguments: json.RawMessage(args)})}
}

func nativePlanAnswer(text string) ai.Message {
	return ai.Message{Role: "assistant", Model: "test", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: text})}
}

func nativePlanProvider(stream ai.StreamFn) *ai.FallbackProvider {
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test","contextWindow":128000}`), Stream: stream}}}
}

func newPlanAgent(t *testing.T, store *todo.Store, stream ai.StreamFn) *agentcore.Agent {
	t.Helper()
	retry := agentcore.RetryPolicy{MaxAttempts: 1}
	a, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{NativeProvider: nativePlanProvider(stream), Model: "test", Retry: &retry, Policy: agentcore.NewAllowList(todo.ToolName)}), todo.With(store))
	if err != nil {
		t.Fatal(err)
	}
	return a
}
