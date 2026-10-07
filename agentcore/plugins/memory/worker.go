package memory

import (
	"context"
	"github.com/2found/2ai/telemetry"
	"sync"
	"time"
)

// ConsolidationWorker runs secondary memory work outside primary run deadlines
// and slots. One host starts Run once, cancels it on shutdown and joins it before
// closing stores. Queue overflow/shutdown never loses the durable rollouts: a
// later successful run in that scope schedules another attempt.
type ConsolidationWorker struct {
	mu     sync.Mutex
	queue  chan *curation
	queued map[string]*curation
	closed bool
}

func NewConsolidationWorker(capacity int) *ConsolidationWorker {
	return &ConsolidationWorker{queue: make(chan *curation, max(1, capacity)), queued: map[string]*curation{}}
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
	if queued := w.queued[c.scopeID]; queued != nil {
		queued.safeOrigin = queued.safeOrigin.Merge(origin)
		return true
	}
	select {
	case w.queue <- c:
		c.safeOrigin = origin
		w.queued[c.scopeID] = c
		return true
	default:
		return false
	}
}

func (w *ConsolidationWorker) Run(ctx context.Context) {
	defer func() {
		w.mu.Lock()
		defer w.mu.Unlock()
		w.closed = true
		clear(w.queued)
		for len(w.queue) > 0 {
			<-w.queue
		}
	}()
	for ctx.Err() == nil {
		select {
		case <-ctx.Done():
			return
		case c := <-w.queue:
			w.mu.Lock()
			origin := c.safeOrigin
			delete(w.queued, c.scopeID)
			w.mu.Unlock()
			// Bound each background pass independently of the originating run.
			workCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
			workCtx = telemetry.BindSafeOrigin(workCtx, origin, telemetry.CategoryMemory)
			c.consolidate(workCtx)
			cancel()
		}
	}
}
