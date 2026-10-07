package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"

	"github.com/2found/2ai/agentcore/host"
	"github.com/2found/2ai/ai"
)

// scriptedNativeMessages translates scripted test data to native messages.
func scriptedNativeMessages(responses ...ChatResponse) []ai.Message {
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
func scriptedNativeProvider(responses ...ChatResponse) *ai.FallbackProvider {
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test","contextWindow":128000}`), Stream: ai.ScriptedStream(scriptedNativeMessages(responses...)...)}}}
}

// nativeCandidate wraps any StreamFn as a single-candidate FallbackProvider.
func nativeCandidate(model string, stream ai.StreamFn) *ai.FallbackProvider {
	raw, _ := json.Marshal(map[string]any{"id": model, "contextWindow": 128000})
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: raw, Stream: stream}}}
}

// nativeModelID extracts the candidate's model id for assertions.
func nativeModelID(raw json.RawMessage) string {
	var m struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal(raw, &m)
	return m.ID
}

// recordedRequest is the projected view of one outgoing provider request —
// the native counterpart of FauxProvider.Recorded entries.
type recordedRequest struct {
	Model    string
	Messages []Message
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
		recorded := recordedRequest{Model: nativeModelID(model), Options: options}
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

// count reports how many requests the provider received.
func (r *nativeRecorder) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.requests)
}

// recordedNativeProvider builds a scripted provider that also records every
// request view — the drop-in for tests that inspected faux.Recorded.
func recordedNativeProvider(rec *nativeRecorder, responses ...ChatResponse) *ai.FallbackProvider {
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test","contextWindow":128000}`), Stream: rec.wrap(ai.ScriptedStream(scriptedNativeMessages(responses...)...))}}}
}

// failingNativeStream returns a StreamFn whose every call fails with err —
// the native counterpart of an erroring LLMProvider.
func failingNativeStream(cause error) ai.StreamFn {
	return func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		return nil, cause
	}
}

// scriptedNativeStream exposes the same script as a bare StreamFn for tests
// composing their own candidates (fallback ladders, per-request logic).
func scriptedNativeStream(responses ...ChatResponse) ai.StreamFn {
	return ai.ScriptedStream(scriptedNativeMessages(responses...)...)
}

// nativeEmit builds a stream that replays one assistant message with visible
// deltas, so tests can put a terminal error on either side of committed output.
func nativeEmit(message *ai.Message) ai.StreamFn {
	return func(ctx context.Context, _ json.RawMessage, _ ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
		out := ai.NewAssistantMessageEventStreamFor(ctx)
		out.Push(ai.AssistantMessageEvent{Type: "start", Partial: message})
		for i, block := range message.Content.Blocks.Values() {
			if block.Type == "text" {
				out.Push(ai.AssistantMessageEvent{Type: "text_delta", ContentIndex: i, Delta: block.Text, Partial: message})
			}
		}
		if message.StopReason == "error" || message.StopReason == "aborted" {
			out.Push(ai.AssistantMessageEvent{Type: "error", Reason: message.StopReason, Error: message})
		} else {
			out.Push(ai.AssistantMessageEvent{Type: "done", Reason: message.StopReason, Message: message})
		}
		out.End()
		return out, nil
	}
}

// nativeAttempts serves a different script per attempt, counting calls — the
// retry/candidate counterpart of a per-call scripted provider.
func nativeAttempts(calls *int32, scripts ...ai.StreamFn) ai.StreamFn {
	return func(ctx context.Context, m json.RawMessage, v ai.TranscriptContext, o map[string]any) (*ai.AssistantMessageEventStream, error) {
		i := int(atomic.AddInt32(calls, 1)) - 1
		if i >= len(scripts) {
			return nil, errors.New("unexpected stream attempt")
		}
		return scripts[i](ctx, m, v, o)
	}
}
