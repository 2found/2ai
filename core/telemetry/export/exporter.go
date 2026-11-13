package export

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	mathrand "math/rand/v2"
	"sync"
	"sync/atomic"
	"time"

	"github.com/2found/2ai/core/telemetry"
)

// Record is one exported span. A settled span is exported once. A heartbeat
// is a snapshot of a root span that is still running: Active is set and
// Timing.Duration is the time elapsed so far.
type Record struct {
	TraceID string                 `json:"trace_id"`
	Span    telemetry.RecordedSpan `json:"span"`
	Timing  telemetry.SpanTiming   `json:"timing"`
	Active  bool                   `json:"active,omitempty"`
}

// Options configures an Exporter. Zero fields use the defaults.
type Options struct {
	// Sink receives each batch on the exporter goroutine. An error or panic
	// counts the batch as failed and is reported to OnError.
	Sink    func(context.Context, []Record) error
	OnError func(error)
	// BatchSize is the most records per delivery; default 256.
	BatchSize int
	// Interval is the longest a record waits for a fuller batch; default 2s.
	Interval time.Duration
	// Queue is how many records may wait for the exporter goroutine; default
	// 4,096. Recording never blocks: a record beyond it is dropped and counted.
	Queue int
	// Sample decides once per root span whether its tree is recorded; nil
	// records every root. See SampleRatio.
	Sample func() bool
	// Heartbeat, when positive, exports a snapshot of every running root this
	// often, so a long or hung run is visible before it ends.
	Heartbeat time.Duration
	// Limits bounds each root's recorder; payload capture is off by default.
	Limits telemetry.Limits
}

// Stats counts an exporter's records and its recorders' drops.
type Stats struct {
	Exported  int64 // records a sink accepted
	Failed    int64 // records in batches a sink rejected
	Dropped   int64 // records refused by a full queue or after Close
	Unsampled int64 // root spans the sampler skipped
	// Recorder sums the Limits drops of finished roots.
	Recorder telemetry.Drops
}

// Exporter streams settled spans to a sink without blocking the recorded work.
type Exporter struct {
	options Options
	queue   chan Record
	flushes chan chan struct{}
	stop    chan struct{}
	done    chan struct{}
	closing sync.Once

	// gate orders enqueues before Close: after stopped is set nothing more
	// enters the queue, so the final drain sees every accepted record.
	gate    sync.RWMutex
	stopped bool

	mu       sync.Mutex
	roots    map[*root]struct{}
	recorder telemetry.Drops

	exported, failed, dropped, unsampled atomic.Int64
}

type root struct {
	trace    string
	started  time.Time
	recorder *telemetry.InMemory
}

// NewExporter starts the exporter goroutine; Close stops it.
func NewExporter(options Options) *Exporter {
	if options.BatchSize <= 0 {
		options.BatchSize = 256
	}
	if options.Interval <= 0 {
		options.Interval = 2 * time.Second
	}
	if options.Queue <= 0 {
		options.Queue = 4096
	}
	e := &Exporter{options: options, queue: make(chan Record, options.Queue), flushes: make(chan chan struct{}), stop: make(chan struct{}), done: make(chan struct{}), roots: map[*root]struct{}{}}
	go e.run()
	return e
}

// SampleRatio keeps about ratio of root spans: 0 keeps none, 1 keeps all.
func SampleRatio(ratio float64) func() bool {
	return func() bool { return mathrand.Float64() < ratio }
}

// Context is the telemetry parent hosts pass to recorded work. Each root span
// gets its own bounded recorder and trace id.
func (e *Exporter) Context() telemetry.Context {
	return telemetry.NewContext(func(options telemetry.SpanOptions, callback func(*telemetry.Span) error) error {
		e.gate.RLock()
		stopped := e.stopped
		e.gate.RUnlock()
		if stopped {
			return telemetry.Context{}.StartSpan(options, callback)
		}
		if e.options.Sample != nil && !e.options.Sample() {
			e.unsampled.Add(1)
			return telemetry.Context{}.StartSpan(options, callback)
		}
		r := &root{trace: newTraceID(), started: time.Now()}
		r.recorder = telemetry.NewStream(func(span telemetry.RecordedSpan, timing telemetry.SpanTiming) {
			e.enqueue(Record{TraceID: r.trace, Span: span, Timing: timing})
		}, e.options.Limits)
		e.mu.Lock()
		e.roots[r] = struct{}{}
		e.mu.Unlock()
		defer func() {
			drops := r.recorder.Dropped()
			e.mu.Lock()
			delete(e.roots, r)
			e.recorder.Spans += drops.Spans
			e.recorder.Events += drops.Events
			e.recorder.Payloads += drops.Payloads
			e.mu.Unlock()
		}()
		return r.recorder.StartSpan(options, callback)
	})
}

func (e *Exporter) enqueue(record Record) {
	e.gate.RLock()
	defer e.gate.RUnlock()
	if e.stopped {
		e.dropped.Add(1)
		return
	}
	select {
	case e.queue <- record:
	default:
		e.dropped.Add(1)
	}
}

// Flush delivers every record queued before the call.
func (e *Exporter) Flush(ctx context.Context) error {
	ack := make(chan struct{})
	select {
	case e.flushes <- ack:
	case <-e.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case <-ack:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close stops recording new roots, delivers what is queued and stops the
// exporter goroutine. Roots still running keep working unrecorded from here;
// their later spans are dropped and counted.
func (e *Exporter) Close(ctx context.Context) error {
	e.closing.Do(func() {
		e.gate.Lock()
		e.stopped = true
		e.gate.Unlock()
		close(e.stop)
	})
	select {
	case <-e.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Stats is a snapshot of the exporter's counters.
func (e *Exporter) Stats() Stats {
	e.mu.Lock()
	recorder := e.recorder
	e.mu.Unlock()
	return Stats{Exported: e.exported.Load(), Failed: e.failed.Load(), Dropped: e.dropped.Load(), Unsampled: e.unsampled.Load(), Recorder: recorder}
}

func (e *Exporter) run() {
	defer close(e.done)
	ticker := time.NewTicker(e.options.Interval)
	defer ticker.Stop()
	var heartbeat <-chan time.Time
	if e.options.Heartbeat > 0 {
		beat := time.NewTicker(e.options.Heartbeat)
		defer beat.Stop()
		heartbeat = beat.C
	}
	pending := make([]Record, 0, e.options.BatchSize)
	// A sink may keep its batch, so a delivered slice is never reused.
	deliver := func() {
		if len(pending) == 0 {
			return
		}
		for start := 0; start < len(pending); start += e.options.BatchSize {
			end := min(start+e.options.BatchSize, len(pending))
			e.deliver(pending[start:end:end])
		}
		pending = make([]Record, 0, e.options.BatchSize)
	}
	drain := func() {
		for {
			select {
			case record := <-e.queue:
				pending = append(pending, record)
			default:
				deliver()
				return
			}
		}
	}
	for {
		select {
		case record := <-e.queue:
			if pending = append(pending, record); len(pending) >= e.options.BatchSize {
				deliver()
			}
		case <-ticker.C:
			deliver()
		case <-heartbeat:
			pending = append(pending, e.heartbeats()...)
			deliver()
		case ack := <-e.flushes:
			drain()
			close(ack)
		case <-e.stop:
			drain()
			return
		}
	}
}

func (e *Exporter) heartbeats() []Record {
	e.mu.Lock()
	roots := make([]*root, 0, len(e.roots))
	for r := range e.roots {
		roots = append(roots, r)
	}
	e.mu.Unlock()
	records := make([]Record, 0, len(roots))
	for _, r := range roots {
		for _, span := range r.recorder.GetSpans() {
			if span.ParentID == nil && !span.Settled {
				records = append(records, Record{TraceID: r.trace, Span: span, Active: true, Timing: telemetry.SpanTiming{SpanID: span.ID, StartedAt: r.started, Duration: time.Since(r.started)}})
			}
		}
	}
	return records
}

func (e *Exporter) deliver(batch []Record) {
	err := func() (err error) {
		defer func() {
			if value := recover(); value != nil {
				err = fmt.Errorf("telemetry sink panic: %v", value)
			}
		}()
		if e.options.Sink == nil {
			return errors.New("telemetry exporter has no sink")
		}
		return e.options.Sink(context.Background(), batch)
	}()
	if err == nil {
		e.exported.Add(int64(len(batch)))
		return
	}
	e.failed.Add(int64(len(batch)))
	if e.options.OnError != nil {
		func() {
			defer func() { _ = recover() }()
			e.options.OnError(err)
		}()
	}
}

func newTraceID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
