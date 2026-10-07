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
