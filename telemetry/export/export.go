// Package export delivers telemetry to host-owned sinks. The Pi-compatible
// recorder owns span lifecycle and parentage; export supplies timing and a
// passive delivery boundary.
//
// New hands a host one whole tree per root span, synchronously at root end,
// for in-process consumers that need the complete tree (diagnostics, tests).
// Exporter is the production path: spans leave as they settle, in batches, on
// a background goroutine, with head sampling, heartbeats and flush on close.
package export

import (
	"time"

	"github.com/2found/2ai/telemetry"
)

type Batch struct {
	StartedAt   time.Time                `json:"started_at"`
	Duration    time.Duration            `json:"duration_ns"`
	Spans       []telemetry.RecordedSpan `json:"spans"`
	SpanTimings []telemetry.SpanTiming   `json:"span_timings,omitempty"`
	Dropped     telemetry.Drops          `json:"dropped,omitzero"`
}

// New records one isolated batch per root callback. Child spans share its
// recorder, bounded by limits (payload capture stays off unless limits opt
// in). The host must join child work before the root callback returns. Sink
// failures never alter the callback's value, error, or panic.
func New(sink func(Batch), limits ...telemetry.Limits) telemetry.Context {
	if sink == nil {
		return telemetry.Context{}
	}
	return telemetry.NewContext(func(options telemetry.SpanOptions, callback func(*telemetry.Span) error) error {
		started := time.Now()
		recorder := telemetry.NewInMemory(limits...)
		defer func() {
			func() {
				defer func() { _ = recover() }()
				sink(Batch{StartedAt: started, Duration: time.Since(started), Spans: recorder.GetSpans(), SpanTimings: recorder.GetSpanTimings(), Dropped: recorder.Dropped()})
			}()
		}()
		return recorder.StartSpan(options, callback)
	})
}
