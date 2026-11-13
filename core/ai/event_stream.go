package ai

import (
	"context"
	"encoding/json"
	"sync"
)

// EventStream ports Pi's unbounded FIFO stream and independent final-result
// promise. Push never waits for a reader. Next consumes one event; Result does
// not drain events. Multiple readers share the queue, rather than receiving
// broadcast copies. Values retain their identity, including Partial pointers.
//
// Go callers may cancel a Next or Result wait through context. Cancellation of
// a reader does not cancel the provider or discard a queued event.
type EventStream[T, R any] struct {
	mu            sync.Mutex
	queue         []T
	head          int
	waiters       []*streamWaiter[T]
	done          bool
	isComplete    func(T) bool
	extractResult func(T) R
	result        R
	resultReady   chan struct{}
	resultSet     bool
}

type streamItem[T any] struct {
	value T
	ok    bool
}
type streamWaiter[T any] struct {
	ready   chan streamItem[T]
	pending bool
}

func NewEventStream[T, R any](isComplete func(T) bool, extractResult func(T) R) *EventStream[T, R] {
	return &EventStream[T, R]{isComplete: isComplete, extractResult: extractResult, resultReady: make(chan struct{})}
}

func (s *EventStream[T, R]) setResult(result R) {
	if s.resultSet {
		return
	}
	s.result, s.resultSet = result, true
	close(s.resultReady)
}

func (s *EventStream[T, R]) Push(event T) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return
	}
	if s.isComplete(event) {
		s.done = true
		s.setResult(s.extractResult(event))
	}
	if len(s.waiters) > 0 {
		waiter := s.waiters[0]
		s.waiters[0] = nil
		s.waiters = s.waiters[1:]
		waiter.pending = false
		waiter.ready <- streamItem[T]{value: event, ok: true}
	} else {
		s.queue = append(s.queue, event)
	}
}

// End stops production and wakes waiting consumers. Queued events remain
// readable. With no result argument, Result remains pending, exactly as Pi's
// end(undefined). A later End(result) may settle it; the first result wins.
func (s *EventStream[T, R]) End(result ...R) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.done = true
	if s.head == len(s.queue) {
		s.queue = nil
		s.head = 0
	}
	if len(result) > 0 {
		s.setResult(result[0])
	}
	for _, waiter := range s.waiters {
		waiter.pending = false
		waiter.ready <- streamItem[T]{}
	}
	s.waiters = nil
}

func (s *EventStream[T, R]) Next(ctx context.Context) (T, bool, error) {
	s.mu.Lock()
	if s.head < len(s.queue) {
		value := s.queue[s.head]
		var zero T
		s.queue[s.head] = zero
		s.head++
		if s.head == len(s.queue) {
			// Reuse small buffers while streaming instead of allocating for
			// every token. Completed streams and large bursts release storage;
			// consumed payload references have already been cleared above.
			if !s.done && cap(s.queue) <= 32 {
				s.queue = s.queue[:0]
			} else {
				s.queue = nil
			}
			s.head = 0
		} else if s.head >= 1024 && s.head >= len(s.queue)/2 {
			// Reclaim consumed prefixes even when the producer keeps the
			// queue continuously nonempty. Copying is amortized over reads.
			s.queue = append([]T(nil), s.queue[s.head:]...)
			s.head = 0
		}
		s.mu.Unlock()
		return value, true, nil
	}
	if s.done {
		s.mu.Unlock()
		var zero T
		return zero, false, nil
	}
	if err := ctx.Err(); err != nil {
		s.mu.Unlock()
		var zero T
		return zero, false, err
	}
	waiter := &streamWaiter[T]{ready: make(chan streamItem[T], 1), pending: true}
	s.waiters = append(s.waiters, waiter)
	s.mu.Unlock()
	select {
	case item := <-waiter.ready:
		return item.value, item.ok, nil
	case <-ctx.Done():
		s.mu.Lock()
		if !waiter.pending {
			// Delivery won the race. Return it rather than lose the event.
			s.mu.Unlock()
			item := <-waiter.ready
			return item.value, item.ok, nil
		}
		for i, candidate := range s.waiters {
			if candidate == waiter {
				copy(s.waiters[i:], s.waiters[i+1:])
				s.waiters[len(s.waiters)-1] = nil
				s.waiters = s.waiters[:len(s.waiters)-1]
				break
			}
		}
		waiter.pending = false
		s.mu.Unlock()
		var zero T
		return zero, false, ctx.Err()
	}
}

func (s *EventStream[T, R]) Result(ctx context.Context) (R, error) {
	// A settled result is available even if the caller's context was cancelled.
	select {
	case <-s.resultReady:
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.result, nil
	default:
	}
	select {
	case <-s.resultReady:
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.result, nil
	case <-ctx.Done():
		var zero R
		return zero, ctx.Err()
	}
}

// AssistantMessageEvent is Pi's event union. Type selects the fields emitted
// on the wire; Partial is the live accumulator, not an event-time snapshot.
type AssistantMessageEvent struct {
	encoding     *transcriptEncoding
	Type         string                     `json:"type"`
	ContentIndex int                        `json:"contentIndex,omitempty"`
	Delta        string                     `json:"delta,omitempty"`
	Content      string                     `json:"content,omitempty"`
	ToolCall     *ContentBlock              `json:"toolCall,omitempty"`
	Partial      *Message                   `json:"partial,omitempty"`
	Reason       string                     `json:"reason,omitempty"`
	Message      *Message                   `json:"message,omitempty"`
	Error        *Message                   `json:"error,omitempty"`
	Extra        map[string]json.RawMessage `json:"-"`
}

func (e AssistantMessageEvent) MarshalJSON() ([]byte, error) {
	type plain AssistantMessageEvent
	required := map[string]any{}
	switch e.Type {
	case "text_start", "thinking_start", "toolcall_start":
		required["contentIndex"] = e.ContentIndex
	case "text_delta", "thinking_delta", "toolcall_delta":
		required["contentIndex"], required["delta"] = e.ContentIndex, e.Delta
	case "text_end", "thinking_end":
		required["contentIndex"], required["content"] = e.ContentIndex, e.Content
	case "toolcall_end":
		required["contentIndex"] = e.ContentIndex
	}
	raw, err := marshalTranscriptObject(plain(e), e.Extra, required)
	return restoreTranscriptEncoding(raw, err, e.encoding)
}

// AssistantMessageEventStream retains Pi's live message pointers. Producers
// updating published payloads must use Synchronize; consumers either inspect
// them in Synchronize or use SnapshotEvent/SnapshotResult. Queue synchronization
// alone cannot protect a message being changed after Push returns.
type AssistantMessageEventStream struct {
	*EventStream[AssistantMessageEvent, *Message]
	payloadMu *sync.Mutex
	ended     chan struct{}
	endOnce   sync.Once
	hooks     AssistantStreamHooks
}

// AssistantStreamHooks supplies the overridable result/iteration operations
// used by Pi's host callback adapter. Its callback promise can reject after
// a terminal event was already pushed. Ordinary provider streams need no hooks.
// Hooks are immutable after construction and must honor reader cancellation.
type AssistantStreamHooks struct {
	Result   func(context.Context) (*Message, error)
	AfterEnd func(context.Context) error
}

type assistantStreamSynchronizationKey struct{}

// WithAssistantStreamSynchronization lets a host relay native provider events
// without copying Pi's live message pointers. Provider streams constructed with
// NewAssistantMessageEventStreamFor share the relay's payload lock. The context
// carries only synchronization, never a queue, result, or cancellation policy.
func WithAssistantStreamSynchronization(ctx context.Context, relay *AssistantMessageEventStream) context.Context {
	return context.WithValue(ctx, assistantStreamSynchronizationKey{}, relay.payloadMu)
}

// NewAssistantMessageEventStreamFor constructs an independent queue/result using
// a host relay's payload lock when supplied, or its own lock otherwise. Set the
// context before starting a producer; locks are immutable once published.
func NewAssistantMessageEventStreamFor(ctx context.Context) *AssistantMessageEventStream {
	stream := NewAssistantMessageEventStream()
	if shared, ok := ctx.Value(assistantStreamSynchronizationKey{}).(*sync.Mutex); ok {
		stream.payloadMu = shared
	}
	return stream
}

func (s *AssistantMessageEventStream) Result(ctx context.Context) (*Message, error) {
	if s.hooks.Result != nil {
		return s.hooks.Result(ctx)
	}
	return s.EventStream.Result(ctx)
}

func (s *AssistantMessageEventStream) Next(ctx context.Context) (AssistantMessageEvent, bool, error) {
	event, ok, err := s.EventStream.Next(ctx)
	if !ok && err == nil && s.hooks.AfterEnd != nil {
		err = s.hooks.AfterEnd(ctx)
	}
	return event, ok, err
}

// End marks producer settlement separately from the first terminal event.
// A proxy can still mutate its live result after that event while draining
// the response. WaitForEnd is useful before inspecting retained raw pointers.
func (s *AssistantMessageEventStream) End(result ...*Message) {
	s.EventStream.End(result...)
	s.endOnce.Do(func() { close(s.ended) })
}

func (s *AssistantMessageEventStream) WaitForEnd(ctx context.Context) error {
	select {
	case <-s.ended:
		return nil
	default:
	}
	select {
	case <-s.ended:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Synchronize protects live payload access. The callback may Push, but must
// not wait on the stream or call Synchronize/Snapshot methods recursively.
func (s *AssistantMessageEventStream) Synchronize(fn func()) {
	s.payloadMu.Lock()
	defer s.payloadMu.Unlock()
	fn()
}

// SnapshotEvent copies the payload as observed now, not at the original Push.
// Shared message, block and usage objects remain shared within the snapshot.
// Retained raw events still observe subsequent mutations, as in Pi.
func (s *AssistantMessageEventStream) SnapshotEvent(event AssistantMessageEvent) (AssistantMessageEvent, error) {
	s.payloadMu.Lock()
	defer s.payloadMu.Unlock()
	return snapshotAssistantEvent(event), nil
}

func (s *AssistantMessageEventStream) SnapshotResult(ctx context.Context) (*Message, error) {
	message, err := s.Result(ctx)
	if err != nil || message == nil {
		return message, err
	}
	snapshot, err := s.SnapshotEvent(AssistantMessageEvent{Message: message})
	return snapshot.Message, err
}

func NewAssistantMessageEventStream() *AssistantMessageEventStream {
	return NewAssistantMessageEventStreamWithHooks(AssistantStreamHooks{})
}

func NewAssistantMessageEventStreamWithHooks(hooks AssistantStreamHooks) *AssistantMessageEventStream {
	return &AssistantMessageEventStream{EventStream: NewEventStream(func(event AssistantMessageEvent) bool { return event.Type == "done" || event.Type == "error" },
		func(event AssistantMessageEvent) *Message {
			if event.Type == "done" {
				return event.Message
			}
			return event.Error
		}), ended: make(chan struct{}), hooks: hooks, payloadMu: new(sync.Mutex)}
}
