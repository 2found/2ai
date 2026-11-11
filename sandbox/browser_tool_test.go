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
	tool := NewBrowserTool(stub, ws, BrowserUseLimits(AnyHost), "agentray-browser:test")
	// A shell metacharacter is just an argument byte: there is no shell.
	out, err := tool.Run(context.Background(), `{"args":["open","https://example.com/?q=$(id);x"]}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	want := []string{"agent-browser", "open", "https://example.com/?q=$(id);x"}
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
	if _, err := tool.Run(context.Background(), `{"args":["snapshot"]}`); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if stub.last.Session != "" {
		t.Fatalf("expected ephemeral (empty) session without a conversation id, got %q", stub.last.Session)
	}

	// With a conversation session, the browser must get a distinct, browser-scoped
	// session id so it does not collide with the computer_use container.
	ctx := agentcore.WithSandboxSession(context.Background(), "conv-42")
	if _, err := tool.Run(ctx, `{"args":["snapshot"]}`); err != nil {
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
	if _, err := NewBrowserTool(&stubSandbox{}, nil, BrowserUseLimits(), "").Run(context.Background(), `{"args":["snapshot"]}`); err == nil {
		t.Fatal("expected missing workspace error")
	}
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	if _, err := NewBrowserTool(&stubSandbox{}, ws, BrowserUseLimits(), "").Run(context.Background(), `{"args":[]}`); err == nil {
		t.Fatal("expected empty command error")
	}
}

func TestBrowserUseLimitsNetworkIsExplicit(t *testing.T) {
	if lim := BrowserUseLimits(); lim.Network || len(lim.NetworkAllow) != 0 {
		t.Fatalf("no allowlist must mean offline, got %+v", lim)
	}
	if lim := BrowserUseLimits("example.com"); !lim.Network || lim.NetworkAllow[0] != "example.com" {
		t.Fatalf("allowlist not threaded: %+v", lim)
	}
}

func TestBrowserToolRefusesUnsafeArgs(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	for name, args := range map[string]string{
		"install":              `["install"]`,
		"batch nests commands": `["batch","open x"]`,
		"upload reads files":   `["upload","@e1","/etc/passwd"]`,
		"unknown command":      `["sh","-c","id"]`,
		"executable path":      `["--executable-path","/bin/sh","open","x"]`,
		"flag after command":   `["open","x","--args=--no-sandbox"]`,
		"flag equals form":     `["--profile=/root","open","x"]`,
		"cdp attach":           `["open","x","--cdp","9222"]`,
		"option before cmd":    `["--headless","open","x"]`,
		"session without name": `["--session"]`,
		"session eats flag":    `["--session","--json","open","x"]`,
		"only global flags":    `["--session","s","--json"]`,
		"NUL byte":             `["open","x\u0000y"]`,
		"empty":                `[]`,
	} {
		stub := &stubSandbox{}
		tool := NewBrowserTool(stub, ws, BrowserUseLimits(), "")
		if _, err := tool.Run(context.Background(), `{"args":`+args+`}`); err == nil {
			t.Errorf("%s: %s accepted", name, args)
		}
		if stub.last.Argv != nil {
			t.Errorf("%s: refused args still reached the sandbox: %v", name, stub.last.Argv)
		}
	}
	for _, args := range []string{`["--session","s","snapshot","-i"]`, `["--session=s","--json","click","@e1"]`, `["screenshot","page.png","--full"]`} {
		if _, err := NewBrowserTool(&stubSandbox{}, ws, BrowserUseLimits(), "").Run(context.Background(), `{"args":`+args+`}`); err != nil {
			t.Errorf("%s refused: %v", args, err)
		}
	}
}
