package agentcore

import (
	"context"
	"encoding/json"
	"errors"
)

// BackgroundLauncher is the run-owned scheduling capability. Its receipt is
// opaque to tools; the installed scheduler supplies status/wait/cancel tools.
// Keeping it here lets contribution plugins cooperate without importing peers.
type BackgroundLauncher func(tool, label string, run func(context.Context) (string, error)) (json.RawMessage, error)
type backgroundKey struct{}

type runProgressKey struct{}

// WithRunProgress binds a host's run-lived observer. Background tools must use
// this instead of retaining a tool-lived streaming emitter. The host serializes
// callbacks and fences delivery when the run ends.
func WithRunProgress(ctx context.Context, progress func(string)) context.Context {
	return context.WithValue(ctx, runProgressKey{}, progress)
}

// ReportProgress publishes a display note, never transcript content or a tool
// result. It is inert without an observer or after cancellation.
func ReportProgress(ctx context.Context, note string) {
	if ctx.Err() != nil || note == "" {
		return
	}
	progress, _ := ctx.Value(runProgressKey{}).(func(string))
	if progress != nil {
		progress(note)
	}
}

func WithBackgroundLauncher(ctx context.Context, launch BackgroundLauncher) context.Context {
	return context.WithValue(ctx, backgroundKey{}, launch)
}
func LaunchBackground(ctx context.Context, tool, label string, run func(context.Context) (string, error)) (json.RawMessage, error) {
	launch, _ := ctx.Value(backgroundKey{}).(BackgroundLauncher)
	if launch == nil {
		return nil, errors.New("background work is not enabled for this run")
	}
	return launch(tool, label, run)
}
