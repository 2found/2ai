package memory

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/2found/2ai/telemetry"
)

const consolidationPassTimeout = 60 * time.Second

type consolidationTask struct {
	*curation
	origin         telemetry.SafeOrigin
	due            time.Time
	order          uint64
	running, dirty bool
	timeouts       int
}
type consolidationResult struct {
	task      *consolidationTask
	origin    telemetry.SafeOrigin
	processed int
	err       error
	timedOut  bool
}

// ConsolidationWorker bounds both admitted scopes and concurrent model work.
// Different scopes progress independently; one scope never has two snapshots in
// flight. A host calls Run once and cancels/joins it before closing its stores.
// Pending evidence remains durable on overflow, failure and shutdown.
type ConsolidationWorker struct {
	mu                    sync.Mutex
	queued                map[string]*consolidationTask
	wake                  chan struct{}
	capacity, parallelism int
	sequence              uint64
	closed                bool
	// Fixed defaults; kept on the worker so scheduler tests need no wall-clock
	// minute waits. These are immutable after Run starts.
	coalesceDelay, retryDelay, passTimeout time.Duration
}

func NewConsolidationWorker(capacity int) *ConsolidationWorker {
	capacity = max(1, capacity)
	return &ConsolidationWorker{capacity: capacity, parallelism: min(4, capacity), queued: map[string]*consolidationTask{}, wake: make(chan struct{}, 1), coalesceDelay: 200 * time.Millisecond, retryDelay: 5 * time.Second, passTimeout: consolidationPassTimeout}
}
func (w *ConsolidationWorker) submit(c *curation) bool {
	return w.submitOrigin(c, telemetry.SafeOrigin{})
}
func (w *ConsolidationWorker) submitOrigin(c *curation, origin telemetry.SafeOrigin) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return false
	}
	if task := w.queued[c.scopeID]; task != nil {
		task.origin = task.origin.Merge(origin)
		// The pass may already have read its snapshot. Remember this notification
		// without admitting a second pass for the same scope.
		task.dirty = true
		return true
	}
	if len(w.queued) >= w.capacity {
		return false
	}
	w.sequence++
	w.queued[c.scopeID] = &consolidationTask{curation: c, origin: origin, due: time.Now().Add(w.coalesceDelay), order: w.sequence}
	select {
	case w.wake <- struct{}{}:
	default:
	}
	return true
}

func (w *ConsolidationWorker) Run(ctx context.Context) {
	results := make(chan consolidationResult, w.parallelism)
	var wg sync.WaitGroup
	timer := time.NewTimer(time.Hour)
	timer.Stop()
	defer func() {
		timer.Stop()
		w.mu.Lock()
		w.closed = true
		clear(w.queued)
		w.mu.Unlock()
		// Each active pass can always deliver its one result into the bounded
		// buffer, including after the scheduler has observed cancellation.
		wg.Wait()
	}()
	active := 0
	for ctx.Err() == nil {
		now := time.Now()
		w.mu.Lock()
		var next *consolidationTask
		if active < w.parallelism {
			for _, task := range w.queued {
				if task.running {
					continue
				}
				if next == nil || task.due.Before(next.due) || (task.due.Equal(next.due) && task.order < next.order) {
					next = task
				}
			}
		}
		if next != nil && !next.due.After(now) {
			next.running = true
			next.dirty = false
			origin := next.origin
			w.mu.Unlock()
			active++
			wg.Add(1)
			go func(task *consolidationTask, origin telemetry.SafeOrigin) {
				defer wg.Done()
				workCtx, cancel := context.WithTimeout(ctx, w.passTimeout)
				workCtx = telemetry.BindSafeOrigin(workCtx, origin, telemetry.CategoryMemory)
				n, err := task.consolidate(workCtx)
				timedOut := errors.Is(workCtx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded)
				cancel()
				results <- consolidationResult{task: task, origin: origin, processed: n, err: err, timedOut: timedOut}
			}(next, origin)
			continue
		}
		var due <-chan time.Time
		if next != nil {
			timer.Reset(max(0, time.Until(next.due)))
			due = timer.C
		}
		w.mu.Unlock()
		select {
		case <-ctx.Done():
			return
		case <-w.wake:
		case <-due:
		case result := <-results:
			active--
			w.mu.Lock()
			task := result.task
			again := false
			delay := time.Duration(0)
			if result.err == nil {
				task.timeouts = 0
				again = result.processed == 4 || task.dirty
			} else if result.timedOut && task.timeouts == 0 && ctx.Err() == nil {
				// One deferred retry for a deadline failure, not an unbounded retry
				// loop or another retry budget for provider/validation/auth errors.
				task.timeouts++
				again = true
				delay = w.retryDelay
			}
			if again {
				task.running = false
				task.dirty = false
				// Continuation can contain older durable evidence; don't invent a
				// singular originating run even when its last batch has one rollout.
				task.origin = result.origin.Merge(telemetry.SafeOrigin{})
				w.sequence++
				task.order = w.sequence
				task.due = time.Now().Add(delay)
			} else {
				delete(w.queued, task.scopeID)
			}
			w.mu.Unlock()
		}
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
	}
}
