package ai

import (
	"context"
	"fmt"
	"sync"
)

// providerOperationQueue serializes complete operations per provider. Canceling
// a waiter never releases an active callback's queue slot. Failure/panic does
// release it when the callback settles, without poisoning the next operation.
type providerOperationQueue struct {
	mu    sync.Mutex
	tails map[string]chan struct{}
}

func (q *providerOperationQueue) run(ctx context.Context, providerID string, task func() (any, error)) (any, error) {
	q.mu.Lock()
	if q.tails == nil {
		q.tails = make(map[string]chan struct{})
	}
	previous := q.tails[providerID]
	tail := make(chan struct{})
	q.tails[providerID] = tail
	q.mu.Unlock()
	type outcome struct {
		value any
		err   error
	}
	ready := make(chan outcome, 1)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				err, ok := recovered.(error)
				if !ok {
					err = fmt.Errorf("%v", recovered)
				}
				if cause := context.Cause(ctx); cause != nil {
					err = cause
				}
				ready <- outcome{err: err}
			}
			q.mu.Lock()
			if q.tails[providerID] == tail {
				delete(q.tails, providerID)
			}
			close(tail)
			q.mu.Unlock()
		}()
		if previous != nil {
			<-previous
		}
		if err := context.Cause(ctx); err != nil {
			ready <- outcome{err: err}
			return
		}
		value, err := task()
		if cause := context.Cause(ctx); cause != nil {
			err = cause
		}
		ready <- outcome{value: value, err: err}
	}()
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	select {
	case done := <-ready:
		return done.value, done.err
	case <-ctx.Done():
		return nil, context.Cause(ctx)
	}
}
