package telemetry

import "context"

type contextKey struct{}

// WithContext carries an explicit telemetry parent through Go host callbacks.
// It creates no span and has no global or goroutine-local state.
func WithContext(ctx context.Context, parent Context) context.Context {
	return context.WithValue(ctx, contextKey{}, parent)
}

func FromContext(ctx context.Context) Context {
	parent, _ := ctx.Value(contextKey{}).(Context)
	return parent
}

// IsZero reports whether this context has no recording backend.
func (c Context) IsZero() bool {
	return c.state == nil && c.callbacks.StartSpan == nil && c.callbacks.StartSpanFrom == nil
}
