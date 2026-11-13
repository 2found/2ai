package agentcore_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/core/ai"
)

// A run that fell back probes the primary again once the ladder is due, and
// returns to it when the probe is served; a failed probe stays on the rung that
// works. The checkpoint records the model that served the last request.
func TestNativeRunReturnsToThePrimaryAfterFallback(t *testing.T) {
	for _, tc := range []struct {
		name         string
		primaryDown  int    // primary requests that fail
		wantSequence string // which rung each provider call reached
		wantModel    string
	}{
		// Turn 1 falls back; turn 2 stays; turn 3 probes the primary, which
		// serves; turn 4 starts on the primary.
		{"recovers", 1, "P F F P P", "primary"},
		// The probe on turn 3 fails, so the fallback serves it and the ladder
		// restarts its count: turn 4 stays on the fallback without a probe.
		{"failed probe", 2, "P F F P F F", "fallback"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls []string
			turns, primaryCalls, effects := 0, 0, 0
			reply := func(ctx context.Context) (*ai.AssistantMessageEventStream, error) {
				turns++
				message := &ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "r" + string(rune('0'+turns)), Name: "read", Arguments: json.RawMessage(`{}`)})}
				if turns == 4 {
					message = &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})}
				}
				return nativeReply(ctx, message)
			}
			provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
				{Model: json.RawMessage(`{"id":"primary"}`), Stream: func(ctx context.Context, _ json.RawMessage, _ ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
					calls = append(calls, "P")
					if primaryCalls++; primaryCalls <= tc.primaryDown {
						return nil, &agentcore.ProviderError{Status: 503, Message: "busy"}
					}
					return reply(ctx)
				}},
				{Model: json.RawMessage(`{"id":"fallback"}`), Stream: func(ctx context.Context, _ json.RawMessage, _ ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
					calls = append(calls, "F")
					return reply(ctx)
				}},
			}}
			retry := agentcore.RetryPolicy{MaxAttempts: 1}
			a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "primary", Retry: &retry, Tools: agentcore.NewToolSet(nativeReadTool{&effects}), Policy: agentcore.NewAllowList("read")})
			if err != nil {
				t.Fatal(err)
			}
			clock := time.Unix(0, 0)
			ladder := &ai.FallbackLadder{RecoverAfterSuccesses: 1, RecoverAfter: time.Hour, Now: func() time.Time { return clock }}
			result, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "go"}}, Fallback: ladder})
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Join(calls, " "); result.Final != "done" || got != tc.wantSequence || effects != 3 {
				t.Fatalf("calls %q want %q (final %q, effects %d)", got, tc.wantSequence, result.Final, effects)
			}
			var checkpoint struct{ Model struct{ ID string } }
			if err := json.Unmarshal(result.NativeState, &checkpoint); err != nil || checkpoint.Model.ID != tc.wantModel {
				t.Fatalf("checkpoint model %q want %q (%v)", checkpoint.Model.ID, tc.wantModel, err)
			}
			if want := map[string]int{"primary": 0, "fallback": 1}[tc.wantModel]; ladder.Rung() != want {
				t.Fatalf("ladder rung %d want %d", ladder.Rung(), want)
			}
		})
	}
}
