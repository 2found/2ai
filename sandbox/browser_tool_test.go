package sandbox

import (
	"context"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
)

func TestBrowserToolRunsThroughSandboxWithWorkspaceMount(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	stub := &stubSandbox{result: agentcore.SandboxResult{ExitCode: 0, Stdout: "ok\n"}}
	tool := NewBrowserTool(stub, ws, BrowserUseLimits(), "agentray-browser:test")
	out, err := tool.Run(context.Background(), `{"command":"agent-browser open https://example.com"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	want := []string{"/bin/sh", "-c", "agent-browser open https://example.com"}
	if strings.Join(stub.last.Argv, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("argv = %v", stub.last.Argv)
	}
	if !stub.last.Constraints.Network || !stub.last.Constraints.WritableFS {
		t.Fatalf("expected browser network + writable fs, got %+v", stub.last.Constraints)
	}
	if stub.last.Image != "agentray-browser:test" {
		t.Fatalf("expected browser image to be threaded, got %q", stub.last.Image)
	}
	if len(stub.last.Mounts) != 1 || stub.last.Mounts[0].Source != ws.Root() || stub.last.Mounts[0].Target != browserWorkdir || stub.last.Mounts[0].ReadOnly {
		t.Fatalf("mounts = %+v", stub.last.Mounts)
	}
	if !strings.Contains(out, "ok") {
		t.Fatalf("output = %q", out)
	}
}

func TestBrowserToolThreadsBrowserScopedSession(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	stub := &stubSandbox{result: agentcore.SandboxResult{ExitCode: 0}}
	tool := NewBrowserTool(stub, ws, BrowserUseLimits(), "agentray-browser:test")

	// Without a conversation session id, the call degrades to ephemeral.
	if _, err := tool.Run(context.Background(), `{"command":"agent-browser snapshot"}`); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if stub.last.Session != "" {
		t.Fatalf("expected ephemeral (empty) session without a conversation id, got %q", stub.last.Session)
	}

	// With a conversation session, the browser must get a distinct, browser-scoped
	// session id so it does not collide with the computer_use container.
	ctx := agentcore.WithSandboxSession(context.Background(), "conv-42")
	if _, err := tool.Run(ctx, `{"command":"agent-browser snapshot"}`); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if want := "conv-42" + browserSessionSuffix; stub.last.Session != want {
		t.Fatalf("session = %q, want %q", stub.last.Session, want)
	}
	if stub.last.Session == "conv-42" {
		t.Fatal("browser session must not share the bare conversation id with computer_use")
	}
}

func TestBrowserToolRejectsMissingDepsAndBadArgs(t *testing.T) {
	// The sandbox is optional (a nil one falls back to the host), but the
	// workspace is not: artifacts have nowhere to land without it.
	if _, err := NewBrowserTool(&stubSandbox{}, nil, BrowserUseLimits(), "").Run(context.Background(), `{"command":"x"}`); err == nil {
		t.Fatal("expected missing workspace error")
	}
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	if _, err := NewBrowserTool(&stubSandbox{}, ws, BrowserUseLimits(), "").Run(context.Background(), `{"command":" "}`); err == nil {
		t.Fatal("expected empty command error")
	}
}
