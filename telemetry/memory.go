package telemetry

import (
	"sync"
	"time"
)

// SpanTiming is independent of the Pi-compatible RecordedSpan JSON contract.
type SpanTiming struct {
	SpanID    int           `json:"span_id"`
	StartedAt time.Time     `json:"started_at"`
	Duration  time.Duration `json:"duration_ns"`
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
}

type memoryState struct {
	mu              sync.Mutex
	spans           []*mutableSpan
	nextEndSequence int
}

// InMemory owns one isolated recording scope. Spans are retained for its
// lifetime, just as in Pi's reference adapter.
type InMemory struct {
	Context
}

func NewInMemory() *InMemory {
	return &InMemory{Context: Context{state: &memoryState{nextEndSequence: 1}}}
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
	attributes, ok := copyAttributes(options.Attributes)
	if !ok {
		return noopSpan
	}
	recorded := &mutableSpan{RecordedSpan: RecordedSpan{
		ID: len(c.state.spans) + 1, Name: options.Name, Attributes: attributes,
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
	if copied, ok := copyAttributes(attributes); ok {
		s.context.parent.Events = append(s.context.parent.Events, RecordedEvent{Name: name, Attributes: copied})
	}
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
	s.context.state.mu.Unlock()
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
		spans = append(spans, copy)
	}
	return spans
}

func copyInt(value *int) *int {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

// GetSpanTimings returns settled timings in admission order, without changing
// snapshot fields, parent admission, or late-child behavior.
func (m *InMemory) GetSpanTimings() []SpanTiming {
	if m == nil || m.state == nil {
		return []SpanTiming{}
	}
	m.state.mu.Lock()
	defer m.state.mu.Unlock()
	timings := make([]SpanTiming, 0, len(m.state.spans))
	for _, span := range m.state.spans {
		if span.Settled {
			timings = append(timings, SpanTiming{SpanID: span.ID, StartedAt: span.startedAt, Duration: span.duration})
		}
	}
	return timings
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
