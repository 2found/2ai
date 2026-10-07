package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

// RunController may stop at a settled turn boundary, before another provider
// request. A nonempty reason suspends execution without a synthetic wrap-up.
// Unlike StopInterceptor this also runs after tool turns and before turn one.
// Usage includes child/reviewer work. Controllers must not perform workload effects.
type RunController interface {
	ControlRun(context.Context, StepInfo) (reason string, err error)
}

// RunCommandHandler accepts host-authorized commands after checkpoint restore.
// Commands are never inferred from transcript text by the kernel.
type RunCommandHandler interface {
	HandleRunCommand(context.Context, json.RawMessage) (json.RawMessage, error)
}

// RunFinalizer settles plugin state in reverse registration order before it is checkpointed. Result is a
// read-only snapshot; failure describes the primary execution error. Returning
// an error prevents the host from treating the checkpoint as successful.
type RunFinalizer interface {
	FinalizeRun(context.Context, RunResult, error) error
}

// ControlPiRun checks the composed controllers without executing a provider.
func (h *PiToolHost) ControlPiRun(ctx context.Context, info StepInfo) (string, error) {
	h.gate.RLock()
	defer h.gate.RUnlock()
	if h.closed {
		return "", errors.New("Pi tool host is closed")
	}
	return h.controlRun(ctx, info)
}
func (h *PiToolHost) controlRun(ctx context.Context, info StepInfo) (string, error) {
	info.Usage = addUsage(info.Usage, h.agent.peekChildUsage())
	for _, ext := range h.exts.all {
		if c, ok := ext.(RunController); ok {
			var reason string
			var err error
			if panicErr := safe(func() { reason, err = c.ControlRun(ctx, info) }); panicErr != nil {
				return "", panicErr
			}
			if err != nil {
				return "", fmt.Errorf("control %s: %w", ext.Name(), err)
			}
			if reason != "" {
				return reason, nil
			}
		}
	}
	return "", nil
}
