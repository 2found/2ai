package memory

import (
	"context"
	"errors"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/telemetry"
)

// Recover admits durable evidence without replaying a primary run. The host
// resolves current authority/provider policy before calling it. It waits only
// for capacity notifications, never polls storage. Errors leave evidence intact.
func (w *ConsolidationWorker) Recover(ctx context.Context, p Plugin, scope string) error {
	p.Worker = w
	ext, err := p.BeginRun(ctx, agentcore.RunInfo{ScopeID: scope})
	if err != nil {
		return err
	}
	c, ok := ext.(*curation)
	if !ok || c.consolidation == nil || c.consolidator == nil {
		return errors.New("memory recovery unavailable")
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		w.mu.Lock()
		changed, closed := w.changed, w.closed
		w.mu.Unlock()
		if closed {
			return errors.New("memory worker closed")
		}
		if w.submitOrigin(c, telemetry.CaptureSafeOrigin(ctx)) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-changed:
		}
	}
}

func (w *ConsolidationWorker) signalSpaceLocked() {
	close(w.changed)
	w.changed = make(chan struct{})
}
