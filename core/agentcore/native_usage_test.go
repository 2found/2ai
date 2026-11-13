package agentcore_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/core/ai"
)

// RunResult.Usage sums every settled attempt, including a failed fallback
// attempt, and keeps the reasoning and 1-hour cache-write subsets.
func TestNativeRunUsageCarriesReasoningAndLongCacheWrites(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	effects := 0
	usage := func(reasoning, long float64) *ai.Usage {
		return &ai.Usage{Input: 1, Output: 10, CacheWrite: 4, Reasoning: f(reasoning), CacheWrite1h: f(long), Cost: ai.UsageCost{Total: 0.5}}
	}
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"primary"}`), Stream: func(ctx context.Context, _ json.RawMessage, _ ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
			return nativeReply(ctx, &ai.Message{Role: "assistant", StopReason: "error", Content: ai.BlockContent(), Usage: usage(1, 1)})
		}},
		{Model: json.RawMessage(`{"id":"fallback"}`), Stream: ai.ScriptedStream(
			ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "r", Name: "read", Arguments: json.RawMessage(`{}`)}), Usage: usage(3, 2)},
			ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"}), Usage: usage(5, 4)},
		)},
	}}
	retry := agentcore.RetryPolicy{MaxAttempts: 1}
	a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "primary", Retry: &retry, Tools: agentcore.NewToolSet(nativeReadTool{&effects}), Policy: agentcore.NewAllowList("read")})
	if err != nil {
		t.Fatal(err)
	}
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "go"}}})
	if err != nil {
		t.Fatal(err)
	}
	u := result.Usage
	if result.Final != "done" || u.OutputTokens != 30 || u.ReasoningTokens != 9 || u.CacheWriteTokens != 12 || u.CacheWrite1hTokens != 7 || u.CostUSD != 1.5 {
		t.Fatalf("usage lost a subset or an attempt: %+v", u)
	}
}

// Sub-agent spend folded into the parent keeps the subsets too.
func TestChildUsageKeepsSubsets(t *testing.T) {
	a, err := agentcore.New(agentcore.Config{NativeProvider: &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"m"}`), Stream: ai.ScriptedStream()}}}, Model: "m", Tools: agentcore.NewToolSet(), Policy: agentcore.NewAllowList()})
	if err != nil {
		t.Fatal(err)
	}
	a.AddChildUsage(agentcore.Usage{OutputTokens: 8, ReasoningTokens: 6, CacheWriteTokens: 3, CacheWrite1hTokens: 2, CostUnpriced: true})
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "go"}}})
	if err != nil {
		t.Fatal(err)
	}
	if u := result.Usage; u.ReasoningTokens != 6 || u.CacheWrite1hTokens != 2 || !u.CostUnpriced {
		t.Fatalf("child usage subsets lost: %+v", u)
	}
}
