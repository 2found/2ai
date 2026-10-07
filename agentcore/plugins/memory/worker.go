package memory

import (
	"context"
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
	queued map[string]bool
	closed bool
}

func NewConsolidationWorker(capacity int) *ConsolidationWorker {
	return &ConsolidationWorker{queue: make(chan *curation, max(1, capacity)), queued: map[string]bool{}}
}

func (w *ConsolidationWorker) submit(c *curation) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return false
	}
	if w.queued[c.scopeID] {
		return true
	}
	select {
	case w.queue <- c:
		w.queued[c.scopeID] = true
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
			delete(w.queued, c.scopeID)
			w.mu.Unlock()
			// Bound each background pass independently of the originating run.
			workCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
			c.consolidate(workCtx)
			cancel()
		}
	}
}
