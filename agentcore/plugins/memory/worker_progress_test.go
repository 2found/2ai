package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry"
)

// Unlike the single-batch fixture, this store honors the batch limit and removes
// only committed evidence. Its lock allows admission while a model is in flight.
type progressStore struct {
	mu        sync.Mutex
	pending   []Rollout
	consumed  int
	reads     int
	committed chan struct{}
}

func newProgressStore(n int) *progressStore {
	s := &progressStore{committed: make(chan struct{}, 128)}
	for i := 0; i < n; i++ {
		s.pending = append(s.pending, Rollout{ID: fmt.Sprintf("%064d", i), Final: "verified evidence"})
	}
	return s
}
func (s *progressStore) Recall(context.Context, string, string, int) ([]agentcore.MemoryEntry, error) {
	return nil, nil
}
func (s *progressStore) Remember(context.Context, agentcore.MemoryEntry) error { return nil }
func (s *progressStore) StageRollout(_ context.Context, _ string, r Rollout) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pending = append(s.pending, r)
	return nil
}
func (s *progressStore) PendingRollouts(_ context.Context, _ string, limit int) ([]Rollout, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reads++
	return append([]Rollout(nil), s.pending[:min(limit, len(s.pending))]...), nil
}
func (s *progressStore) CommitConsolidation(ctx context.Context, scope string, in Consolidation, changes []Change) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ValidateConsolidation(scope, in, changes); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range in.Rollouts {
		found := -1
		for i, p := range s.pending {
			if p.ID == r.ID {
				found = i
				break
			}
		}
		if found < 0 {
			return fmt.Errorf("duplicate or missing commit")
		}
		s.pending = append(s.pending[:found], s.pending[found+1:]...)
		s.consumed++
	}
	s.committed <- struct{}{}
	return nil
}
func (s *progressStore) counts() (int, int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.pending), s.consumed, s.reads
}
func progressCuration(scope string, s *progressStore, f Consolidator) *curation {
	return &curation{scopeID: scope, store: s, consolidation: s, consolidator: f}
}
func runProgressWorker(t *testing.T, w *ConsolidationWorker) context.CancelFunc {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { w.Run(ctx); close(done) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("worker did not join")
		}
	})
	return cancel
}
func awaitProgress(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(2 * time.Second):
		t.Fatal("pending memory work did not progress")
	}
}

func TestWorkerDrainsAllBatchesWithoutAnotherPrimaryRun(t *testing.T) {
	s := newProgressStore(9)
	w := NewConsolidationWorker(4)
	var calls atomic.Int32
	if !w.submit(progressCuration("scope", s, func(context.Context, Consolidation) ([]Change, error) { calls.Add(1); return []Change{}, nil })) {
		t.Fatal("admission")
	}
	runProgressWorker(t, w)
	for i := 0; i < 3; i++ {
		awaitProgress(t, s.committed)
	}
	pending, consumed, _ := s.counts()
	if pending != 0 || consumed != 9 || calls.Load() != 3 {
		t.Fatalf("stranded or repeated evidence: %d/%d calls=%d", pending, consumed, calls.Load())
	}
	// A drained worker must block, not repeatedly query an empty store.
	time.Sleep(20 * time.Millisecond)
	_, _, before := s.counts()
	time.Sleep(40 * time.Millisecond)
	_, _, after := s.counts()
	if before != after {
		t.Fatalf("idle storage polling: %d -> %d", before, after)
	}
}

func TestWorkerSlowScopeDoesNotBlockOthersAndNeverOverlapsSameScope(t *testing.T) {
	w := NewConsolidationWorker(8)
	slow := newProgressStore(1)
	fast := newProgressStore(1)
	entered := make(chan struct{}, 4)
	release := make(chan struct{})
	var running, peak atomic.Int32
	c := progressCuration("slow", slow, func(ctx context.Context, in Consolidation) ([]Change, error) {
		n := running.Add(1)
		defer running.Add(-1)
		for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
		}
		entered <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return []Change{}, nil
	})
	w.submit(c)
	runProgressWorker(t, w)
	awaitProgress(t, entered)
	slow.StageRollout(context.Background(), "slow", Rollout{ID: fmt.Sprintf("%064d", 99), Final: "new evidence"})
	w.submit(c)
	w.submit(progressCuration("fast", fast, func(context.Context, Consolidation) ([]Change, error) { return []Change{}, nil }))
	awaitProgress(t, fast.committed)
	if peak.Load() != 1 {
		t.Fatal("overlapping same-scope model snapshots")
	}
	close(release)
	awaitProgress(t, slow.committed)
	awaitProgress(t, slow.committed)
	if p, n, _ := slow.counts(); p != 0 || n != 2 || peak.Load() != 1 {
		t.Fatalf("coalesced arrival lost or overlapping: %d/%d peak=%d", p, n, peak.Load())
	}
}

func TestWorkerBoundsFiftyScopesAndJoinsCancellation(t *testing.T) {
	w := NewConsolidationWorker(64)
	w.coalesceDelay = 0
	entered := make(chan struct{}, 64)
	var active, peak atomic.Int32
	stores := make([]*progressStore, 50)
	for i := range stores {
		stores[i] = newProgressStore(1)
		w.submit(progressCuration(fmt.Sprint(i), stores[i], func(ctx context.Context, _ Consolidation) ([]Change, error) {
			n := active.Add(1)
			defer active.Add(-1)
			for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
			}
			entered <- struct{}{}
			<-ctx.Done()
			return nil, ctx.Err()
		}))
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { w.Run(ctx); close(done) }()
	defer cancel()
	for i := 0; i < 4; i++ {
		awaitProgress(t, entered)
	}
	select {
	case <-entered:
		cancel()
		t.Fatal("unbounded parallel model calls")
	case <-time.After(30 * time.Millisecond):
	}
	cancel()
	awaitProgress(t, done)
	if active.Load() != 0 || peak.Load() != 4 {
		t.Fatalf("shutdown did not join bounded work: active=%d peak=%d", active.Load(), peak.Load())
	}
	for _, s := range stores {
		if p, n, _ := s.counts(); p != 1 || n != 0 {
			t.Fatal("cancelled or queued evidence consumed")
		}
	}
	if w.submit(progressCuration("late", newProgressStore(1), nil)) {
		t.Fatal("shutdown admitted work")
	}
}

func TestWorkerDeadlineRetryIsDeferredAndBounded(t *testing.T) {
	for _, recoverable := range []bool{true, false} {
		t.Run(fmt.Sprint(recoverable), func(t *testing.T) {
			w := NewConsolidationWorker(2)
			w.parallelism = 1
			w.coalesceDelay = 0
			w.passTimeout = 20 * time.Millisecond
			w.retryDelay = 80 * time.Millisecond
			s := newProgressStore(1)
			healthy := newProgressStore(1)
			failed := make(chan struct{}, 4)
			var calls atomic.Int32
			c := progressCuration("timeout", s, func(ctx context.Context, _ Consolidation) ([]Change, error) {
				if calls.Add(1) == 2 && recoverable {
					return []Change{}, nil
				}
				<-ctx.Done()
				return nil, ctx.Err()
			})
			c.onConsolidationError = func(context.Context, error) { failed <- struct{}{} }
			w.submit(c)
			runProgressWorker(t, w)
			awaitProgress(t, failed)
			w.submit(progressCuration("healthy", healthy, func(context.Context, Consolidation) ([]Change, error) { return []Change{}, nil }))
			awaitProgress(t, healthy.committed)
			if calls.Load() != 1 {
				t.Fatal("retry occupied a worker during backoff")
			}
			if recoverable {
				awaitProgress(t, s.committed)
			} else {
				awaitProgress(t, failed)
			}
			time.Sleep(3 * w.retryDelay)
			p, n, _ := s.counts()
			if calls.Load() != 2 {
				t.Fatalf("deadline retry not bounded: %d", calls.Load())
			}
			if recoverable && (p != 0 || n != 1) || !recoverable && (p != 1 || n != 0) {
				t.Fatalf("retry consumed invalid evidence: %d/%d", p, n)
			}
		})
	}
}

func TestWorkerValidationFailureDoesNotRetryAndContinuationIsFair(t *testing.T) {
	w := NewConsolidationWorker(4)
	w.parallelism = 1
	w.coalesceDelay = 0
	w.retryDelay = time.Millisecond
	big := newProgressStore(9)
	small := newProgressStore(1)
	invalid := newProgressStore(1)
	order := make(chan string, 8)
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	var calls atomic.Int32
	w.submit(progressCuration("big", big, func(ctx context.Context, _ Consolidation) ([]Change, error) {
		if calls.Add(1) == 1 {
			started <- struct{}{}
			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		order <- "big"
		return []Change{}, nil
	}))
	runProgressWorker(t, w)
	awaitProgress(t, started)
	w.submit(progressCuration("small", small, func(context.Context, Consolidation) ([]Change, error) { order <- "small"; return []Change{}, nil }))
	bad := progressCuration("invalid", invalid, func(context.Context, Consolidation) ([]Change, error) {
		order <- "invalid"
		return []Change{{Entry: &agentcore.MemoryEntry{ScopeID: "another", Content: "forbidden"}}}, nil
	})
	rejected := make(chan struct{}, 8)
	bad.onConsolidationError = func(context.Context, error) { rejected <- struct{}{} }
	w.submit(bad)
	close(release)
	for i := 0; i < 3; i++ {
		awaitProgress(t, big.committed)
	}
	awaitProgress(t, small.committed)
	awaitProgress(t, rejected)
	if <-order != "big" || <-order != "small" || <-order != "invalid" {
		t.Fatal("large backlog starved waiting scopes")
	}
	time.Sleep(20 * time.Millisecond)
	if p, n, _ := invalid.counts(); p != 1 || n != 0 {
		t.Fatal("invalid scope proposal committed")
	}
	select {
	case <-rejected:
		t.Fatal("validation errors retried")
	default:
	}
}

func TestWorkerContinuationDoesNotInventOriginForLastRollout(t *testing.T) {
	observer, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{})
	if err != nil {
		t.Fatal(err)
	}
	origin, err := telemetry.WithSafeObserver(context.Background(), observer, telemetry.SafeExecution{ExecutionID: "latest-run"})
	if err != nil {
		t.Fatal(err)
	}
	s := newProgressStore(5)
	w := NewConsolidationWorker(4)
	w.coalesceDelay = 0
	response := ai.Message{Role: "assistant", StopReason: "stop", Content: ai.TextContent(`{"changes":[]}`)}
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"fixture"}`), Stream: ai.ScriptedStream(response, response)}}}
	c := progressCuration("scope", s, nativeConsolidator(provider, agentcore.RunInfo{}))
	c.onConsolidationError = func(_ context.Context, err error) { t.Error(err) }
	w.submitOrigin(c, telemetry.CaptureSafeOrigin(origin))
	stop := runProgressWorker(t, w)
	awaitProgress(t, s.committed)
	awaitProgress(t, s.committed)
	stop()
	attempts := 0
	for _, r := range observer.Drain(128) {
		if r.Kind == "attempt_settled" {
			attempts++
			if r.Category != telemetry.CategoryMemory || r.OriginExecutionID != nil {
				t.Fatalf("backlog attributed to latest root: %+v", r)
			}
		}
	}
	if attempts != 2 {
		t.Fatalf("background attempts missing: %d", attempts)
	}
}
