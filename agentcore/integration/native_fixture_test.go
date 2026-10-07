package integration

// Native provider fixtures for the integration tests: the scripted seam every
// Prompt-driven test runs against, plus a request recorder standing in for
// FauxProvider.Recorded.

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/host"
	"github.com/2found/2ai/ai"
)

// scriptedNativeMessages translates scripted test data to native messages.
func scriptedNativeMessages(responses ...agentcore.ChatResponse) []ai.Message {
	messages := make([]ai.Message, 0, len(responses))
	for _, response := range responses {
		blocks := []ai.ContentBlock{}
		if response.Message.Content != "" {
			blocks = append(blocks, ai.ContentBlock{Type: "text", Text: response.Message.Content})
		}
		for _, call := range response.Message.ToolCalls {
			arguments := json.RawMessage(call.Arguments)
			if !json.Valid(arguments) {
				arguments = ai.ParseStreamingJSON(call.Arguments)
			}
			blocks = append(blocks, ai.ContentBlock{Type: "toolCall", ID: call.ID, Name: call.Name, Arguments: arguments})
		}
		stop := response.StopReason
		switch {
		case stop == "":
			stop = "stop"
		case stop == "tool_calls":
			stop = "toolUse"
		}
		if len(response.Message.ToolCalls) > 0 && stop == "stop" {
			stop = "toolUse"
		}
		u := response.Usage
		messages = append(messages, ai.Message{Role: "assistant", Content: ai.BlockContent(blocks...), StopReason: stop, Usage: &ai.Usage{Input: float64(u.InputTokens), Output: float64(u.OutputTokens), CacheRead: float64(u.CacheReadTokens), CacheWrite: float64(u.CacheWriteTokens), Cost: ai.UsageCost{Total: u.CostUSD}}})
	}
	return messages
}

// scriptedNativeProvider translates scripted test data to native messages;
// every turn, tool and ceiling decision executes in the real native engine.
func scriptedNativeProvider(responses ...agentcore.ChatResponse) *ai.FallbackProvider {
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test","contextWindow":128000}`), Stream: ai.ScriptedStream(scriptedNativeMessages(responses...)...)}}}
}

// recordedRequest is the projected view of one outgoing provider request —
// the native counterpart of FauxProvider.Recorded entries.
type recordedRequest struct {
	Model    string
	Messages []agentcore.Message
	Tools    []string
	Options  map[string]any
}

// nativeRecorder captures every request a stream sees, in order, so tests can
// assert what the engine sent the model the way faux.Recorded did.
type nativeRecorder struct {
	mu       sync.Mutex
	requests []recordedRequest
}

// wrap records the request view, then delegates to inner.
func (r *nativeRecorder) wrap(inner ai.StreamFn) ai.StreamFn {
	return func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		recorded := recordedRequest{Options: options}
		for _, msg := range view.Messages() {
			raw, err := json.Marshal(msg)
			if err != nil {
				return nil, err
			}
			projected, err := host.ProjectMessage(raw)
			if err != nil {
				return nil, err
			}
			recorded.Messages = append(recorded.Messages, projected)
		}
		for _, tool := range ai.GetCurrentTools(view.Messages()) {
			recorded.Tools = append(recorded.Tools, tool.Name)
		}
		r.mu.Lock()
		r.requests = append(r.requests, recorded)
		r.mu.Unlock()
		return inner(ctx, model, view, options)
	}
}

// all returns a snapshot of every recorded request.
func (r *nativeRecorder) all() []recordedRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]recordedRequest, len(r.requests))
	copy(out, r.requests)
	return out
}

// recordedNativeProvider builds a scripted provider that also records every
// request view — the drop-in for tests that inspected faux.Recorded.
func recordedNativeProvider(rec *nativeRecorder, responses ...agentcore.ChatResponse) *ai.FallbackProvider {
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test","contextWindow":128000}`), Stream: rec.wrap(ai.ScriptedStream(scriptedNativeMessages(responses...)...))}}}
}
