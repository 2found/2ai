package telemetry

// StartSpanFrom reads options only when this context can admit a span. A read
// panic falls back to no-op telemetry, without allocating an ID or changing the
// callback's result. The callback still runs exactly once. Readers run without
// the recorder lock and may create other spans; those are admitted first.
// Concurrent parent settlement wins over a reader that has not yet returned.
func (c Context) StartSpanFrom(read func() SpanOptions, callback func(*Span) error) error {
	if c.callbacks.StartSpanFrom != nil {
		return c.callbacks.StartSpanFrom(read, callback)
	}
	if c.state == nil {
		return callback(noopSpan)
	}
	c.state.mu.Lock()
	settled := c.parent != nil && c.parent.Settled
	c.state.mu.Unlock()
	if settled {
		return callback(noopSpan)
	}
	options, ok := readSpanOptions(read)
	if !ok {
		return callback(noopSpan)
	}
	return c.StartSpan(options, callback)
}

// StartSpanFrom creates a child using deferred options and explicit parentage.
func (s *Span) StartSpanFrom(read func() SpanOptions, callback func(*Span) error) error {
	return s.context.StartSpanFrom(read, callback)
}

// StartSpanFrom preserves the typed callback value and error, as StartSpan does.
func StartSpanFrom[T any](c Context, read func() SpanOptions, callback func(*Span) (T, error)) (value T, err error) {
	err = c.StartSpanFrom(read, func(span *Span) error {
		var callbackErr error
		value, callbackErr = callback(span)
		return callbackErr
	})
	return value, err
}

func readSpanOptions(read func() SpanOptions) (options SpanOptions, ok bool) {
	defer func() { _ = recover() }()
	return read(), true
}
