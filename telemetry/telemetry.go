// Package telemetry ports Pi's explicit-parent, callback-scoped telemetry to
// Go. The zero Context is inert; NewInMemory records span snapshots.
// It has no dependency on a JavaScript runtime or on the application.
package telemetry

import (
	"github.com/2found/2ai/jsonjs"
	"runtime"
)

// Attributes records Pi's scalar and array attributes. Like Pi's runtime, it
// also accepts JSON-shaped objects, mixed arrays and nested values without
// imposing the TypeScript declaration as runtime validation. Attribute objects
// and outer arrays are copied; nested objects/arrays retain shared identity.
// NewAttributes and its Get/Lookup/Set/Delete/Entries methods retain property
// insertion order, with numeric index keys enumerated first. The zero value is empty.
// NewArray retains shared length and indexed mutations. Decoded arrays use
// *Array; direct array attributes are spread-copied, while nested arrays stay shared.
// A nil value represents null. Undefined omits an attribute, including during a
// merge; nested object fields omit Undefined and array elements export it as null.
// Function values remain in memory; JSON omits object fields and uses null for
// array elements without invoking the function.
// Numeric Go types share one category; []any may mix numeric widths. JSON uses
// binary64 numbers (including integer rounding), nonfinite values become null,
// and byte slices remain numeric arrays. In-memory snapshots keep Go types.
// JSON decoding creates binary64 values, retaining overflow and signed zero;
// a successful decode replaces the object, and a failed decode leaves it untouched.
// Decoded lone UTF-16 surrogates use WTF-8 bytes in strings and map keys.
// Attribute JSON export preserves them; ordinary Go rune iteration does not.
// Callers must synchronize access to shared input/snapshot maps and slices.
type Attributes = jsonjs.ObjectValue

// Undefined represents JavaScript's undefined in attribute payloads. Unlike nil
// (null), it does not overwrite an existing attribute during SetAttributes.
// Nested snapshots retain the marker; JSON omits object fields carrying it and
// emits null for array elements, matching JSON.stringify.
const Undefined = jsonjs.Undefined

type SpanOptions struct {
	Name       string     `json:"name"`
	Attributes Attributes `json:"attributes,omitzero"`
	SafeLabels SafeLabels `json:"-"`
}

type ErrorDetails struct {
	Name     string `json:"name"`
	Message  string `json:"message"`
	encoding *errorDetailsEncoding
}

// Error preserves an explicitly named failure when adapting serialized Pi
// errors. Ordinary Go errors continue to use the default name "Error".
func (e *ErrorDetails) Error() string { return e.Message }

type SpanStatus struct {
	Status    string        `json:"status"`
	Error     *ErrorDetails `json:"error,omitempty"`
	nullInput bool
}

// Context identifies an explicit parent. Copies share the same recording scope;
// there is no global recorder or implicit goroutine-local parent.
type Context struct {
	safe      *safeScope
	state     *memoryState
	parent    *mutableSpan
	callbacks ContextCallbacks
}

// NewContext binds a backend's synchronous callback-scoped span operation.
// The backend owns settlement, passivity, and explicit child parentage. A nil
// function gives the same no-op behavior as the zero Context.
func NewContext(start func(SpanOptions, func(*Span) error) error) Context {
	return NewContextWithCallbacks(ContextCallbacks{StartSpan: start})
}

// ContextCallbacks supplies eager and deferred span admission for a backend.
// Backends own settlement, read passivity and child parentage. A missing
// operation invokes the user's callback with a no-op span; an absent deferred
// operation also leaves its options reader uncalled.
type ContextCallbacks struct {
	StartSpan     func(SpanOptions, func(*Span) error) error
	StartSpanFrom func(func() SpanOptions, func(*Span) error) error
}

// NewContextWithCallbacks binds a backend that supports deferred option reads.
// NewContext is the equivalent constructor for an eager-only backend.
func NewContextWithCallbacks(callbacks ContextCallbacks) Context {
	return Context{callbacks: callbacks}
}

// SpanCallbacks are the mutation operations supplied by a telemetry backend.
// Provided callbacks override native recorder operations; opaque contexts leave
// missing operations inert. Backends must ignore mutations after settlement and
// suppress recording failures without changing the user's callback result.
type SpanCallbacks struct {
	AddEvent      func(string, Attributes)
	SetAttributes func(Attributes)
	SetStatus     func(SpanStatus)
	// Deferred reads own admission, including skipping reads after settlement
	// and suppressing read failures. Nil retains the supplied context's default
	// behavior (inert for an opaque backend).
	SetStatusFrom     func(func() SpanStatus)
	SetAttributesFrom func(func() Attributes)
	AddEventFrom      func(string, func() Attributes)
}

// Span is valid as a recording parent until its callback returns. Retaining it
// is safe: later mutations are inert, and later child callbacks still execute.
type Span struct {
	safeState *safeSpanState
	context   Context
	callbacks SpanCallbacks
}

// NewSpan constructs a backend span with its explicit child context. It does
// not add lifecycle behavior; that belongs to the backend's start function.
func NewSpan(children Context, callbacks SpanCallbacks) *Span {
	return &Span{context: children, callbacks: callbacks}
}

var noopSpan = &Span{}

// Context returns this span's explicit child context.
func (s *Span) Context() Context { return s.context }

func (s *Span) StartSpan(options SpanOptions, callback func(*Span) error) error {
	return s.context.StartSpan(options, callback)
}

// StartSpan invokes callback exactly once in the calling goroutine. Returning
// an error or panicking settles the span without changing the error/panic value.
// Child work that should belong to this span must start before callback returns.
func (c Context) StartSpan(options SpanOptions, callback func(*Span) error) (err error) {
	if c.safe != nil {
		return c.startSafeSpan(options, callback)
	}
	if c.callbacks.StartSpan != nil {
		return c.callbacks.StartSpan(options, callback)
	}
	span := c.admit(options)
	if span == noopSpan {
		return callback(span)
	}
	returned := false
	defer func() {
		if !returned {
			value := recover()
			failure := value
			if _, nilPanic := value.(*runtime.PanicNilError); nilPanic {
				// Go wraps panic(nil), but Pi's thrown null is not an Error.
				// Normalize recording only; preserve the recovered panic below.
				failure = nil
			}
			span.settle(true, failure)
			if value != nil {
				panic(value)
			}
			return // runtime.Goexit still runs defers without becoming a panic.
		}
		span.settle(err != nil, err)
	}()
	err = callback(span)
	returned = true
	return err
}

// StartSpan preserves a typed callback result as well as its error. The method
// on Context is the equivalent for callbacks that only return an error.
func StartSpan[T any](c Context, options SpanOptions, callback func(*Span) (T, error)) (value T, err error) {
	err = c.StartSpan(options, func(span *Span) error {
		var callbackErr error
		value, callbackErr = callback(span)
		return callbackErr
	})
	return value, err
}
