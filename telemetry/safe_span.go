package telemetry

import (
	"context"
	"sync/atomic"
	"time"
)

type safeSpanState struct {
	failed  atomic.Bool
	outcome atomic.Pointer[safeSpanOutcome]
}
type safeSpanOutcome struct{ status, failure string }

// SetSafeOutcome binds only bounded structural status metadata. It never reads
// a private status/error or changes the legacy recorder's status precedence.
func (s *Span) SetSafeOutcome(status, failure string) {
	if s != nil && s.safeState != nil {
		s.safeState.outcome.Store(&safeSpanOutcome{safeStatus(status), SafeFailureKind(failure)})
	}
}

func (c Context) startSafeSpan(options SpanOptions, callback func(*Span) error) (err error) {
	scope := *c.safe
	c.safe = nil
	started := time.Now()
	// Preserve an origin parent when starting the first detached span.
	if scope.span != "" {
		scope.parentSpan, scope.parentExecution = scope.span, scope.execution
	}
	scope.span = scope.observer.id()
	kind := "unknown"
	switch options.Name {
	case "agentray.agent.run":
		kind = "agent"
	case "agentray.ai.request":
		kind = "model"
	case "agentray.ai.compaction":
		kind = "model"
		scope.category = CategoryCompaction
	case "agentray.ai.advisor":
		kind = "model"
		scope.category = CategoryAdvisor
	case "agentray.ai.memory_consolidation":
		kind = "model"
		scope.category = CategoryMemory
	case "agentray.tool.execute":
		kind = "tool"
	}
	r := scope.record("span_settled", started)
	r.SpanKind = kind
	r.ToolID = ptrID(options.SafeLabels.tool)
	if r.ToolID != nil {
		r.LabelsState = "approved"
	}
	state := &safeSpanState{}
	returned := false
	defer func() {
		r.SettledAt, r.Duration = time.Now().UTC(), time.Since(started)
		r.Status, r.FailureKind = "completed", "none"
		if !returned || err != nil || state.failed.Load() {
			r.Status, r.FailureKind = "failed", "unknown"
		}
		if err == context.Canceled {
			r.Status, r.FailureKind = "cancelled", "request_cancelled"
		}
		if err == context.DeadlineExceeded {
			r.Status, r.FailureKind = "cancelled", "request_timeout"
		}
		if outcome := state.outcome.Load(); returned && err == nil && outcome != nil {
			r.Status, r.FailureKind = outcome.status, outcome.failure
		}
		scope.observer.admit(r)
	}()
	err = c.StartSpan(options, func(private *Span) error {
		span := *private // never mutate the global no-op span or private backend
		span.context.safe = &scope
		span.safeState = state
		return callback(&span)
	})
	returned = true
	return err
}
