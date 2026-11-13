package telemetry

import (
	"errors"
	"sync"
	"time"
)

var errPayloadPanic = errors.New("telemetry payload panicked")

// SpanTiming is a detached timing for a settled span. It is kept separately
// from RecordedSpan so Pi-compatible records retain their original JSON shape.
// EventTimes holds when each of the span's events was recorded, in order.
type SpanTiming struct {
	SpanID     int           `json:"span_id"`
	StartedAt  time.Time     `json:"started_at"`
	Duration   time.Duration `json:"duration_ns"`
	EventTimes []time.Time   `json:"event_times,omitempty"`
}

// Limits bounds one recording scope so a long run cannot grow it without
// limit. Zero fields use the defaults. Payload capture (see
// Span.AddEventWithPayload) is off unless Payloads is positive.
type Limits struct {
	Spans        int // spans held at once; default 10,000
	Events       int // events per span; default 1,000
	Payloads     int // opt-in payloads captured per scope; default 0 (off)
	PayloadBytes int // bytes per payload; default 1 MiB
}

func (l Limits) resolved() Limits {
	if l.Spans <= 0 {
		l.Spans = 10_000
	}
	if l.Events <= 0 {
		l.Events = 1_000
	}
	if l.PayloadBytes <= 0 {
		l.PayloadBytes = 1 << 20
	}
	return l
}

// Drops counts what a recording scope refused because of its Limits. A
// payload is also counted when its producer failed.
type Drops struct {
	Spans    int `json:"spans,omitempty"`
	Events   int `json:"events,omitempty"`
	Payloads int `json:"payloads,omitempty"`
}

type RecordedEvent struct {
	Name       string     `json:"name"`
	Attributes Attributes `json:"attributes"`
}

type RecordedSpan struct {
	ID          int             `json:"id"`
	ParentID    *int            `json:"parentId"`
	Name        string          `json:"name"`
	Attributes  Attributes      `json:"attributes"`
	Events      []RecordedEvent `json:"events"`
	Status      SpanStatus      `json:"status"`
	Settled     bool            `json:"settled"`
	EndSequence *int            `json:"endSequence,omitempty"`
}

type mutableSpan struct {
	RecordedSpan
	explicitStatus bool
	startedAt      time.Time
	duration       time.Duration
	eventTimes     []time.Time
}

type memoryState struct {
	mu              sync.Mutex
	spans           []*mutableSpan
	nextID          int
	nextEndSequence int
	limits          Limits
	payloads        int
	dropped         Drops
	// settled, when set, receives each span as it settles; the span is then
	// forgotten so the scope holds only live spans.
	settled func(RecordedSpan, SpanTiming)
}

// InMemory owns one isolated recording scope. Spans are retained for its
// lifetime, just as in Pi's reference adapter, within its Limits.
type InMemory struct {
	Context
}

func NewInMemory(limits ...Limits) *InMemory {
	return newMemory(nil, limits)
}

// NewStream records like NewInMemory but hands each span to settled as soon
// as it settles and then forgets it, so a long run holds only its live spans
// and a crash loses only those. settled runs on the settling goroutine
// outside the recorder lock and must not block; a panic in it is ignored.
func NewStream(settled func(RecordedSpan, SpanTiming), limits ...Limits) *InMemory {
	return newMemory(settled, limits)
}

func newMemory(settled func(RecordedSpan, SpanTiming), limits []Limits) *InMemory {
	var bound Limits
	if len(limits) > 0 {
		bound = limits[0]
	}
	return &InMemory{Context: Context{state: &memoryState{nextID: 1, nextEndSequence: 1, limits: bound.resolved(), settled: settled}}}
}

// Dropped reports what this scope refused because of its Limits.
func (m *InMemory) Dropped() Drops {
	if m == nil || m.state == nil {
		return Drops{}
	}
	m.state.mu.Lock()
	defer m.state.mu.Unlock()
	return m.state.dropped
}

func (c Context) admit(options SpanOptions) *Span {
	if c.state == nil {
		return noopSpan
	}
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	if c.parent != nil && c.parent.Settled {
		return noopSpan
	}
	if len(c.state.spans) >= c.state.limits.Spans {
		c.state.dropped.Spans++
		return noopSpan
	}
	attributes, ok := copyAttributes(options.Attributes)
	if !ok {
		return noopSpan
	}
	c.state.nextID++
	recorded := &mutableSpan{RecordedSpan: RecordedSpan{
		ID: c.state.nextID - 1, Name: options.Name, Attributes: attributes,
		Events: []RecordedEvent{}, Status: SpanStatus{Status: "ok"},
	}, startedAt: time.Now()}
	if c.parent != nil {
		parentID := c.parent.ID
		recorded.ParentID = &parentID
	}
	c.state.spans = append(c.state.spans, recorded)
	return &Span{context: Context{state: c.state, parent: recorded}}
}

func (s *Span) AddEvent(name string, attributes Attributes) {
	if s != nil && s.callbacks.AddEvent != nil {
		s.callbacks.AddEvent(name, attributes)
		return
	}
	if s == nil || s.context.state == nil || s.context.parent == nil {
		return
	}
	s.context.state.mu.Lock()
	defer s.context.state.mu.Unlock()
	if s.context.parent.Settled {
		return
	}
	if len(s.context.parent.Events) >= s.context.state.limits.Events {
		s.context.state.dropped.Events++
		return
	}
	if copied, ok := copyAttributes(attributes); ok {
		s.context.parent.Events = append(s.context.parent.Events, RecordedEvent{Name: name, Attributes: copied})
		s.context.parent.eventTimes = append(s.context.parent.eventTimes, time.Now())
	}
}

// CapturesPayloads reports whether this span's recorder opted into payloads
// and can still take one, so a caller can skip snapshotting a payload's
// inputs when nothing would keep them.
func (s *Span) CapturesPayloads() bool {
	if s == nil || s.callbacks.AddEvent != nil || s.context.state == nil || s.context.parent == nil {
		return false
	}
	s.context.state.mu.Lock()
	defer s.context.state.mu.Unlock()
	return !s.context.parent.Settled && s.context.state.payloads < s.context.state.limits.Payloads
}

// AddEventWithPayload records an event and, only when the recorder opted into
// payloads, attaches payload's bytes as the string attribute key. payload runs
// outside the recorder lock and only when its result would be kept. A payload
// beyond the count or size bound, or whose producer fails, is dropped and
// counted, and the event carries key+".dropped" with the reason ("count",
// "size" or "error"). Backends supplied through callbacks receive the event
// without the payload.
func (s *Span) AddEventWithPayload(name string, attributes Attributes, key string, payload func() ([]byte, error)) {
	if s == nil || s.callbacks.AddEvent != nil || s.context.state == nil || s.context.parent == nil || payload == nil {
		s.AddEvent(name, attributes)
		return
	}
	state := s.context.state
	state.mu.Lock()
	enabled, settled := state.limits.Payloads > 0, s.context.parent.Settled
	reserved := enabled && !settled && state.payloads < state.limits.Payloads
	if reserved {
		state.payloads++
	}
	state.mu.Unlock()
	if !enabled || settled {
		s.AddEvent(name, attributes)
		return
	}
	event, ok := copyAttributes(attributes)
	if !ok {
		return
	}
	reason := "count"
	if reserved {
		raw, err := readPayload(payload)
		switch {
		case err != nil:
			reason = "error"
		case len(raw) > state.limits.PayloadBytes:
			reason = "size"
		default:
			event.Set(key, string(raw))
			reason = ""
		}
	}
	if reason != "" {
		event.Set(key+".dropped", reason)
		state.mu.Lock()
		state.dropped.Payloads++
		if reserved {
			state.payloads--
		}
		state.mu.Unlock()
	}
	s.AddEvent(name, event)
}

func readPayload(payload func() ([]byte, error)) (raw []byte, err error) {
	defer func() {
		if recover() != nil {
			raw, err = nil, errPayloadPanic
		}
	}()
	return payload()
}

func (s *Span) SetAttributes(attributes Attributes) {
	if s != nil && s.callbacks.SetAttributes != nil {
		s.callbacks.SetAttributes(attributes)
		return
	}
	if s == nil || s.context.state == nil || s.context.parent == nil {
		return
	}
	s.context.state.mu.Lock()
	defer s.context.state.mu.Unlock()
	if s.context.parent.Settled {
		return
	}
	// Copy the entire incoming payload before merging, so invalid input cannot
	// leave a partially applied mutation.
	if copied, ok := copyAttributes(attributes); ok {
		for _, property := range copied.Entries() {
			s.context.parent.Attributes.Set(property.Name, property.Value)
		}
	}
}

func (s *Span) SetStatus(status SpanStatus) {
	if s != nil && s.safeState != nil && status.Status != "ok" {
		s.safeState.failed.Store(true)
	}
	if s != nil && s.callbacks.SetStatus != nil {
		s.callbacks.SetStatus(status)
		return
	}
	// Pi ignores null because reading status.status throws. A readable empty
	// object is different: it normalizes to an explicit error status. This is
	// the in-memory recorder's policy; supplied backends receive the input.
	if status.nullInput && status.Status == "" && status.Error == nil {
		return
	}
	if s == nil || s.context.state == nil || s.context.parent == nil {
		return
	}
	s.context.state.mu.Lock()
	defer s.context.state.mu.Unlock()
	if s.context.parent.Settled {
		return
	}
	s.context.parent.Status = copyStatus(status)
	s.context.parent.explicitStatus = true
}

// SetStatusFrom reads a status only when recording is active. It is the Go
// equivalent of Pi reading a status object with user-defined getters. A reader
// panic is passive: no outer status/explicit-status change is applied, while
// the reader's own reentrant mutations remain. Readers run without the recorder
// lock, so they may record events, set a nested status or inspect snapshots.
// Concurrent settlement wins over a reader that has not yet returned.
// Callback backends implement admission through SpanCallbacks.SetStatusFrom.
func (s *Span) SetStatusFrom(read func() SpanStatus) {
	if s == nil {
		return
	}
	if s.callbacks.SetStatusFrom != nil {
		s.callbacks.SetStatusFrom(read)
		return
	}
	if s.context.state == nil || s.context.parent == nil {
		return
	}
	s.context.state.mu.Lock()
	settled := s.context.parent.Settled
	s.context.state.mu.Unlock()
	if settled {
		return
	}
	if status, ok := readSpanStatus(read); ok {
		s.SetStatus(status)
	}
}

func readSpanStatus(read func() SpanStatus) (status SpanStatus, ok bool) {
	defer func() { _ = recover() }()
	return read(), true
}

func (s *Span) settle(failed bool, failure any) {
	s.context.state.mu.Lock()
	if s.context.parent.Settled {
		s.context.state.mu.Unlock()
		return
	}
	if failed && !s.context.parent.explicitStatus {
		// Error() is user code. Only inspect it when needed, outside the lock;
		// it can reenter the recorder or panic without replacing the failure.
		s.context.state.mu.Unlock()
		status := automaticErrorStatus(failure)
		s.context.state.mu.Lock()
		// Pi checks explicitStatus before inspecting the error, then assigns
		// the computed status. A reentrant SetStatus during inspection does
		// not cancel that assignment (including when inspection panics).
		s.context.parent.Status = status
	}
	s.context.parent.Settled = true
	s.context.parent.duration = time.Since(s.context.parent.startedAt)
	sequence := s.context.state.nextEndSequence
	s.context.state.nextEndSequence++
	s.context.parent.EndSequence = &sequence
	settled := s.context.state.settled
	if settled == nil {
		s.context.state.mu.Unlock()
		return
	}
	span, timing := s.context.parent.snapshot(), s.context.parent.timing()
	spans := s.context.state.spans
	for i := len(spans) - 1; i >= 0; i-- {
		if spans[i] == s.context.parent {
			s.context.state.spans = append(spans[:i], spans[i+1:]...)
			break
		}
	}
	s.context.state.mu.Unlock()
	func() {
		defer func() { _ = recover() }()
		settled(span, timing)
	}()
}

func (span *mutableSpan) snapshot() RecordedSpan {
	copy := span.RecordedSpan
	copy.ParentID = copyInt(span.ParentID)
	copy.EndSequence = copyInt(span.EndSequence)
	copy.Attributes, _ = copyAttributes(span.Attributes)
	copy.Status = copyStatus(span.Status)
	copy.Events = make([]RecordedEvent, len(span.Events))
	for i, event := range span.Events {
		attributes, _ := copyAttributes(event.Attributes)
		copy.Events[i] = RecordedEvent{Name: event.Name, Attributes: attributes}
	}
	return copy
}

func (span *mutableSpan) timing() SpanTiming {
	timing := SpanTiming{SpanID: span.ID, StartedAt: span.startedAt, Duration: span.duration}
	if len(span.eventTimes) > 0 {
		timing.EventTimes = append([]time.Time(nil), span.eventTimes...)
	}
	return timing
}

// GetSpans returns snapshots in start order, including active spans. Attribute
// maps and outer arrays are detached; nested objects/arrays remain shared, as
// with Pi's shallow copy. Callers must synchronize edits to shared payloads.
func (m *InMemory) GetSpans() []RecordedSpan {
	if m == nil || m.state == nil {
		return []RecordedSpan{}
	}
	m.state.mu.Lock()
	defer m.state.mu.Unlock()
	spans := make([]RecordedSpan, 0, len(m.state.spans))
	for _, span := range m.state.spans {
		spans = append(spans, span.snapshot())
	}
	return spans
}

// GetSpanTimings returns settled span timings in start order. Active and no-op
// spans have no completed timing. Durations use Go's monotonic clock and each
// snapshot is independent of future recording.
func (m *InMemory) GetSpanTimings() []SpanTiming {
	if m == nil || m.state == nil {
		return []SpanTiming{}
	}
	m.state.mu.Lock()
	defer m.state.mu.Unlock()
	timings := make([]SpanTiming, 0, len(m.state.spans))
	for _, span := range m.state.spans {
		if span.Settled {
			timings = append(timings, span.timing())
		}
	}
	return timings
}

func copyInt(value *int) *int {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func copyStatus(status SpanStatus) SpanStatus {
	if status.Status == "ok" {
		return SpanStatus{Status: "ok"}
	}
	copy := SpanStatus{Status: "error"}
	if status.Error != nil {
		details := *status.Error
		copy.Error = &details
	}
	return copy
}

func automaticErrorStatus(failure any) (status SpanStatus) {
	status.Status = "error"
	defer func() { _ = recover() }()
	if details, ok := failure.(*ErrorDetails); ok {
		copied := *details
		status.Error = &copied
		return status
	}
	if err, ok := failure.(error); ok {
		status.Error = &ErrorDetails{Name: "Error", Message: err.Error()}
	}
	return status
}
