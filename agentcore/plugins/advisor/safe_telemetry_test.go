package advisor_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/advisor"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry"
)

func TestNativeAdvisorSafeFailedFallbackAccountsOnce(t *testing.T) {
	o, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := telemetry.WithSafeObserver(context.Background(), o, telemetry.SafeExecution{ExecutionID: "advisor-owner"})
	if err != nil {
		t.Fatal(err)
	}
	usage := func(n float64) *ai.Usage {
		return &ai.Usage{Input: n, Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}
	}
	reviewer := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"failed"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "error", Usage: usage(2)})},
		{Model: json.RawMessage(`{"id":"reviewer"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: `{"notes":[]}`}), Usage: usage(11)})},
	}}
	a, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{Model: "test", NativeProvider: provider(ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "primary answer"}), Usage: usage(3)}))}), advisor.Native(reviewer))
	if err != nil {
		t.Fatal(err)
	}
	result, err := a.RunNative(ctx, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "work"}}})
	if err != nil || result.Final != "primary answer" || result.Usage.InputTokens != 16 {
		t.Fatalf("advisor accounting result=%+v err=%v", result, err)
	}
	main, reviews, tokens := 0, 0, 0.0
	for _, r := range o.Drain(256) {
		if r.Kind == "attempt_settled" {
			if r.ExecutionID != "advisor-owner" || r.SpanID == nil {
				t.Fatal("advisor lost owner/span")
			}
			switch r.Category {
			case telemetry.CategoryMain:
				main++
			case telemetry.CategoryAdvisor:
				reviews++
			}
			tokens += r.Accounting.Tokens.Input
		}
	}
	if main != 1 || reviews != 2 || tokens != 16 {
		t.Fatalf("advisor safe totals main=%d reviews=%d tokens=%g", main, reviews, tokens)
	}
}
