// Package export delivers completed telemetry scopes to host-owned sinks.
// The Pi-compatible recorder owns span lifecycle and parentage; export only
// supplies timing and a passive delivery boundary.
package export

import (
	"time"

	"github.com/2found/2ai/telemetry"
)

type Batch struct {
	StartedAt time.Time                `json:"started_at"`
	Duration  time.Duration            `json:"duration_ns"`
	Spans     []telemetry.RecordedSpan `json:"spans"`
}

// New records one isolated batch per root callback. Child spans share its
// recorder. The host must join child work before the root callback returns.
// Sink failures never alter the callback's value, error, or panic.
func New(sink func(Batch)) telemetry.Context {
	if sink == nil {
		return telemetry.Context{}
	}
	return telemetry.NewContext(func(options telemetry.SpanOptions, callback func(*telemetry.Span) error) error {
		started := time.Now()
		recorder := telemetry.NewInMemory()
		defer func() {
			func() {
				defer func() { _ = recover() }()
				sink(Batch{StartedAt: started, Duration: time.Since(started), Spans: recorder.GetSpans()})
			}()
		}()
		return recorder.StartSpan(options, callback)
	})
}
