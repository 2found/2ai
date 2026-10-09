package memory

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry"
)

func TestNativeMemorySafeDeferredOriginsCoalescingAndUsage(t *testing.T) {
	for _, coalesced := range []bool{false, true} {
		t.Run(map[bool]string{false: "single-origin", true: "coalesced"}[coalesced], func(t *testing.T) {
			o, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{})
			if err != nil {
				t.Fatal(err)
			}
			ctx, err := telemetry.WithSafeObserver(context.Background(), o, telemetry.SafeExecution{ExecutionID: "memory-owner"})
			if err != nil {
				t.Fatal(err)
			}
			w := NewConsolidationWorker(1)
			store := &consolidationStore{}
			u := func(n float64) *ai.Usage {
				return &ai.Usage{Input: n, Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}
			}
			secondary := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
				{Model: json.RawMessage(`{"id":"failed"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "error", Usage: u(5)})},
				{Model: json.RawMessage(`{"id":"memory"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: `{"changes":[]}`}), Usage: u(13)})},
			}}
			build := func() *agentcore.Agent {
				a, err := agentcore.New(agentcore.Config{Model: "test", Definition: agentcore.AgentDefinition{ScopeID: "scope"}, NativeProvider: &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "primary answer"}), Usage: u(2)})}}}, Extensions: []agentcore.ExtensionFactory{Plugin{Store: store, NativeProvider: secondary, Worker: w}}})
				if err != nil {
					t.Fatal(err)
				}
				return a
			}
			result, err := build().RunNative(ctx, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "first work"}}})
			if err != nil || result.Usage.InputTokens != 2 {
				t.Fatalf("deferred memory changed primary: %+v %v", result, err)
			}
			before := o.Drain(256)
			for _, r := range before {
				if r.Kind == "attempt_settled" && r.Category != telemetry.CategoryMain {
					t.Fatal("background memory ran on primary path")
				}
			}
			if coalesced {
				other, _ := telemetry.WithSafeObserver(context.Background(), o, telemetry.SafeExecution{ExecutionID: "other-owner"})
				if _, err := build().RunNative(other, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "different work"}}}); err != nil {
					t.Fatal(err)
				}
				_ = o.Drain(256)
			}
			// Start only after the primary result and its safe root have settled.
			workCtx, cancel := context.WithCancel(context.Background())
			done := make(chan struct{})
			committed := make(chan struct{})
			w.mu.Lock()
			queued := w.queued["scope"]
			queued.consolidation = &commitSignalStore{consolidationStore: store, committed: committed}
			w.mu.Unlock()
			go func() { w.Run(workCtx); close(done) }()
			<-committed
			cancel()
			<-done
			attempts, tokens := 0, 0.0
			for _, r := range o.Drain(256) {
				if r.Kind != "attempt_settled" {
					continue
				}
				attempts++
				tokens += r.Accounting.Tokens.Input
				if r.Category != telemetry.CategoryMemory || r.ExecutionID == "memory-owner" || r.SpanID == nil {
					t.Fatalf("memory identity/category: %+v", r)
				}
				if coalesced && r.OriginExecutionID != nil || !coalesced && (r.OriginExecutionID == nil || *r.OriginExecutionID != "memory-owner") {
					t.Fatalf("memory guessed origin: %+v", r)
				}
			}
			if attempts != 2 || tokens != 18 || result.Usage.InputTokens != 2 || store.commits != 1 || len(store.pending) != 0 {
				t.Fatalf("memory usage duplicated/lost attempts=%d tokens=%g result=%+v", attempts, tokens, result.Usage)
			}
		})
	}
}
