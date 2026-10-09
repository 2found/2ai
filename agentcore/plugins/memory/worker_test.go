package memory

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

func TestDeferredConsolidationCapacityCoalescingAndShutdown(t *testing.T) {
	w := NewConsolidationWorker(1)
	var calls atomic.Int32
	var reports atomic.Int32
	makeCuration := func(scope string) (*curation, *consolidationStore) {
		s := &consolidationStore{}
		p := Plugin{Store: s, Worker: w, OnConsolidationError: func(context.Context, error) { reports.Add(1) }, Consolidator: func(ctx context.Context, in Consolidation) ([]Change, error) {
			calls.Add(1)
			return []Change{{Entry: &agentcore.MemoryEntry{Content: "Keep verified lessons."}}}, nil
		}}
		e, err := p.BeginRun(context.Background(), agentcore.RunInfo{ScopeID: scope})
		if err != nil {
			t.Fatal(err)
		}
		return e.(*curation), s
	}
	a, sa := makeCuration("a")
	b, sb := makeCuration("b")
	result := agentcore.RunResult{Turns: 1, StopReason: "stop", Final: "done", Messages: []agentcore.Message{{Role: agentcore.RoleAssistant, Content: "evidence"}}}
	for _, c := range []*curation{a, a, b} {
		if err := c.FinalizeRun(context.Background(), result, nil); err != nil {
			t.Fatal(err)
		}
	}
	if calls.Load() != 0 || len(w.queued) != 1 || reports.Load() != 1 || len(sa.pending) != 1 || len(sb.pending) != 1 {
		t.Fatal("unbounded, duplicate or synchronous work")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	// Cancel only after commit; synchronization makes the fake store safe to inspect.
	committed := make(chan struct{})
	a.onConsolidationError = func(context.Context, error) { t.Error("unexpected failure") }
	// Use a wrapper to signal the commit rather than polling store internals.
	wrapped := &commitSignalStore{consolidationStore: sa, committed: committed}
	a.consolidation = wrapped
	go func() { w.Run(ctx); close(done) }()
	select {
	case <-committed:
	case <-time.After(time.Second):
		cancel()
		<-done
		t.Fatal("worker did not commit")
	}
	cancel()
	<-done
	if calls.Load() != 1 || sa.commits != 1 || len(sa.pending) != 0 || len(sb.pending) != 1 {
		t.Fatal("consolidation lost evidence")
	}
	if w.submit(b) {
		t.Fatal("closed worker admitted new work")
	}
	// Retained evidence can be consumed by the next host worker after shutdown.
	b.worker = nil
	if err := b.FinalizeRun(context.Background(), result, nil); err != nil {
		t.Fatal(err)
	}
	if sb.commits != 1 || len(sb.pending) != 0 {
		t.Fatal("pending evidence could not recover")
	}
}

type commitSignalStore struct {
	*consolidationStore
	committed chan struct{}
}

func (s *commitSignalStore) CommitConsolidation(ctx context.Context, scope string, in Consolidation, changes []Change) error {
	err := s.consolidationStore.CommitConsolidation(ctx, scope, in, changes)
	close(s.committed)
	return err
}

func TestWorkerFailureIsSecondaryAndLeavesPendingEvidence(t *testing.T) {
	w := NewConsolidationWorker(1)
	s := &consolidationStore{}
	reported := make(chan struct{})
	c := &curation{store: s, consolidation: s, scopeID: "scope", worker: w, consolidator: func(context.Context, Consolidation) ([]Change, error) { return nil, errors.New("provider unavailable") }, onConsolidationError: func(context.Context, error) { close(reported) }}
	if err := c.FinalizeRun(context.Background(), agentcore.RunResult{Turns: 1, Final: "successful answer", Messages: []agentcore.Message{{Role: agentcore.RoleAssistant, Content: "done"}}}, nil); err != nil {
		t.Fatal("secondary error failed primary run", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { w.Run(ctx); close(done) }()
	select {
	case <-reported:
	case <-time.After(time.Second):
		cancel()
		<-done
		t.Fatal("failure not reported")
	}
	cancel()
	<-done
	if len(s.pending) != 1 || s.commits != 0 {
		t.Fatal("failed model call consumed evidence")
	}
}
