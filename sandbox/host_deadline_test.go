package sandbox

import (
	"context"
	"io"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

func TestHostUnlimitedStillCancelsProcessGroup(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	proc, err := NewHostSandbox().Start(ctx, agentcore.SandboxExec{Argv: []string{"/bin/sh", "-c", "echo ready; sleep 60 & wait"}, Constraints: agentcore.SandboxLimits{TimeoutSeconds: -1}})
	if err != nil {
		t.Fatal(err)
	}
	ready := make([]byte, len("ready\n"))
	if _, err := io.ReadFull(proc.Stdout(), ready); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	cancel()
	_, _ = proc.Wait()
	if time.Since(started) > 2*time.Second {
		t.Fatal("unlimited process ignored cancellation")
	}
}

func TestHostTimeoutModes(t *testing.T) {
	for _, seconds := range []float64{-1, 0, 0.01} {
		ctx, cancel := hostRunContext(context.Background(), seconds)
		deadline, ok := ctx.Deadline()
		if (seconds < 0 && ok) || (seconds >= 0 && !ok) {
			t.Fatalf("deadline for %g: %v", seconds, ok)
		}
		if seconds == 0 && time.Until(deadline) < 29*time.Second {
			t.Fatal("lost default timeout")
		}
		cancel()
	}
}
