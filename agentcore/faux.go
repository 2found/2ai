package agentcore

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// FauxProvider is a scripted LLM seam for tests: it replays a fixed list of
// responses in order, with no network, no keys, and no tokens (pi faux-provider
// pattern). It lets the loop, hooks, and permission gate be tested
// deterministically.
type FauxProvider struct {
	Responses []ChatResponse
	calls     int
	Recorded  []ChatRequest
}

// NewFauxProvider builds a provider that returns the given responses in order.
func NewFauxProvider(responses ...ChatResponse) *FauxProvider {
	return &FauxProvider{Responses: responses}
}

func (f *FauxProvider) Name() string        { return "faux" }
func (f *FauxProvider) SupportsTools() bool { return true }

// Chat returns the next scripted response. After the script is exhausted it
// returns a plain assistant message with stop reason "stop" so loops terminate.
func (f *FauxProvider) Chat(_ context.Context, req ChatRequest) (ChatResponse, error) {
	f.Recorded = append(f.Recorded, req)
	if f.calls >= len(f.Responses) {
		return ChatResponse{
			Message:    Message{Role: RoleAssistant, Content: "(end)"},
			StopReason: "stop",
		}, nil
	}
	resp := f.Responses[f.calls]
	f.calls++
	return resp, nil
}

// Stream adapts Chat into a delta channel: it emits the content word-by-word (so
// tests exercise token concatenation), then one delta per tool call, then a
// terminal Done — mirroring how the real provider streams a turn.
func (f *FauxProvider) Stream(ctx context.Context, req ChatRequest) (<-chan ChatDelta, error) {
	ch := make(chan ChatDelta, 8)
	go func() {
		defer close(ch)
		resp, _ := f.Chat(ctx, req)
		for i, word := range strings.Fields(resp.Message.Content) {
			frag := word
			if i > 0 {
				frag = " " + word
			}
			ch <- ChatDelta{ContentDelta: frag}
		}
		for i := range resp.Message.ReasoningBlocks {
			block := resp.Message.ReasoningBlocks[i]
			ch <- ChatDelta{ReasoningBlock: &block}
		}
		for i := range resp.Message.ToolCalls {
			tc := resp.Message.ToolCalls[i]
			ch <- ChatDelta{ToolCall: &tc}
		}
		ch <- ChatDelta{Done: true, StopReason: resp.StopReason, Usage: resp.Usage}
	}()
	return ch, nil
}

// AssistantText is a helper to script a plain text response.
func AssistantText(content string) ChatResponse {
	return ChatResponse{Message: Message{Role: RoleAssistant, Content: content}, StopReason: "stop"}
}

// AssistantToolCall is a helper to script a tool-calling response.
func AssistantToolCall(id, name, args string) ChatResponse {
	return ChatResponse{
		Message:    Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: id, Name: name, Arguments: args}}},
		StopReason: "tool_calls",
	}
}

// ReplayProvider is a deterministic LLMProvider over a recorded []TurnRecord.
//
// omp has no record/replay of this shape — the recon looked — so this is
// agentray's. TurnRecord already persists everything a provider needs to be
// replayed (the request Messages, the Response, the ToolCalls the model
// asked for, advertised Tools, StopReason, Error, tokens/cost). Serving those
// as ChatResponses, in order, lets a test drive the real loop against a
// transcript instead of a scripted FauxProvider that ignores the request.
//
// The assertion is the whole point. FauxProvider will happily answer a loop
// that rebuilt history differently; ReplayProvider fails the call, which is
// what turns a replay into a regression test. Comparison is the recorded
// transcript, not the whole ChatRequest:
//
//   - Messages: Role, Content, ContentParts, Name, ToolCallID, Directive, Error,
//     and each ToolCall's ID/Name/Arguments. That is the history the loop rebuilt.
//   - Advertised tool names, in order (TurnRecord.Tools), against req.Tools.
//
// Ignored, because they are not the history and are legitimately
// nondeterministic or unrecorded:
//
//   - Message.CacheAnchor — request-scoped, json:"-", dropped on any
//     persistence round-trip of the trace.
//   - Message.Usage — provider-reported accounting, not part of rebuild.
//   - Model, Temperature, MaxTokens, CacheKey, CacheRetention,
//     ReasoningEffort, OutputSchema — not on TurnRecord.
//
// Empty recording: Chat returns an error ("replay: empty recording"). Extra
// call past the transcript: Chat returns an error naming the index. Neither
// invents a response — FauxProvider's "(end)" would hide a loop that failed
// to stop. A recorded Error is returned as Chat's error (with the recorded
// response still filled in); the loop's retry policy still applies, so a
// test of a failed turn should set Retry.MaxAttempts=1 or record the retries.
type ReplayProvider struct {
	Records []TurnRecord
	calls   int
}

// NewReplayProvider plays the given records in order.
func NewReplayProvider(records ...TurnRecord) *ReplayProvider {
	return &ReplayProvider{Records: records}
}

func (r *ReplayProvider) Name() string        { return "replay" }
func (r *ReplayProvider) SupportsTools() bool { return true }

// Chat serves the next recorded response after asserting the request matches
// that turn's recording. See ReplayProvider's doc for the comparison rule.
func (r *ReplayProvider) Chat(_ context.Context, req ChatRequest) (ChatResponse, error) {
	if len(r.Records) == 0 {
		return ChatResponse{}, errors.New("replay: empty recording")
	}
	if r.calls >= len(r.Records) {
		return ChatResponse{}, fmt.Errorf("replay: extra call %d, recording has %d turns", r.calls+1, len(r.Records))
	}
	rec := r.Records[r.calls]
	if len(rec.NativeTrace) > 0 && string(rec.NativeTrace) != "null" {
		return ChatResponse{}, errors.New("replay: native traces require the native Pi runtime")
	}
	if drift := replayDrift(req, rec); drift != "" {
		return ChatResponse{}, fmt.Errorf("replay: turn %d request drifted from recording: %s", r.calls+1, drift)
	}
	r.calls++
	resp := ChatResponse{
		Message: Message{
			Role:            RoleAssistant,
			Content:         rec.Response,
			ReasoningBlocks: rec.ReasoningBlocks,
			ToolCalls:       rec.ToolCalls,
		},
		StopReason: rec.StopReason,
		Usage:      Usage{InputTokens: rec.TokensIn, OutputTokens: rec.TokensOut, CostUSD: rec.CostUSD},
	}
	if rec.Error != "" {
		return resp, errors.New(rec.Error)
	}
	return resp, nil
}

// Stream adapts Chat into a delta channel, matching FauxProvider's shape so
// the loop's streamTurn path is exercised. Unlike FauxProvider it must not
// swallow Chat's error: a mismatch or recorded failure has to reach streamTurn
// as ChatDelta.Err, or a drifting loop would look like a silent empty turn.
func (r *ReplayProvider) Stream(ctx context.Context, req ChatRequest) (<-chan ChatDelta, error) {
	ch := make(chan ChatDelta, 8)
	go func() {
		defer close(ch)
		resp, err := r.Chat(ctx, req)
		if err != nil {
			ch <- ChatDelta{Done: true, Err: err, StopReason: resp.StopReason, Usage: resp.Usage}
			return
		}
		for i, word := range strings.Fields(resp.Message.Content) {
			frag := word
			if i > 0 {
				frag = " " + word
			}
			ch <- ChatDelta{ContentDelta: frag}
		}
		for i := range resp.Message.ReasoningBlocks {
			block := resp.Message.ReasoningBlocks[i]
			ch <- ChatDelta{ReasoningBlock: &block}
		}
		for i := range resp.Message.ToolCalls {
			tc := resp.Message.ToolCalls[i]
			ch <- ChatDelta{ToolCall: &tc}
		}
		ch <- ChatDelta{Done: true, StopReason: resp.StopReason, Usage: resp.Usage}
	}()
	return ch, nil
}

func replayDrift(req ChatRequest, rec TurnRecord) string {
	if len(req.Messages) != len(rec.Messages) {
		return fmt.Sprintf("message count got %d want %d", len(req.Messages), len(rec.Messages))
	}
	for i := range req.Messages {
		if !replayMessageEqual(req.Messages[i], rec.Messages[i]) {
			got, want := req.Messages[i], rec.Messages[i]
			return fmt.Sprintf("message[%d] role %s/%s content %q/%q", i, got.Role, want.Role, clipReplay(got.Content), clipReplay(want.Content))
		}
	}
	gotTools := advertisedToolNames(req.Tools)
	if !sameToolNames(gotTools, rec.Tools) {
		return fmt.Sprintf("tools got %v want %v", gotTools, rec.Tools)
	}
	return ""
}

func sameToolNames(got, want []string) bool {
	if len(got) == 0 && len(want) == 0 {
		return true
	}
	return slices.Equal(got, want)
}

func replayMessageEqual(got, want Message) bool {
	if got.Role != want.Role || got.Content != want.Content || got.Name != want.Name ||
		got.ToolCallID != want.ToolCallID || got.Directive != want.Directive || got.Error != want.Error {
		return false
	}
	if len(got.ToolCalls) != len(want.ToolCalls) {
		return false
	}
	for i := range got.ToolCalls {
		a, b := got.ToolCalls[i], want.ToolCalls[i]
		if a.ID != b.ID || a.Name != b.Name || a.Arguments != b.Arguments {
			return false
		}
	}
	if len(got.ContentParts) != len(want.ContentParts) {
		return false
	}
	for i := range got.ContentParts {
		if got.ContentParts[i] != want.ContentParts[i] {
			return false
		}
	}
	if !slices.Equal(got.ReasoningBlocks, want.ReasoningBlocks) {
		return false
	}
	return true
}

func advertisedToolNames(tools []ToolSchema) []string {
	if len(tools) == 0 {
		return nil
	}
	out := make([]string, len(tools))
	for i, t := range tools {
		out[i] = t.Name
	}
	return out
}

func clipReplay(s string) string {
	const n = 80
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
