package sandbox

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

// stubSandbox records the last request and returns a canned result, so the
// shell tool's contract can be tested without a container runtime.
type stubSandbox struct {
	last   agentcore.SandboxExec
	result agentcore.SandboxResult
	err    error
}

func (s *stubSandbox) Exec(_ context.Context, req agentcore.SandboxExec) (agentcore.SandboxResult, error) {
	s.last = req
	return s.result, s.err
}

func TestShellToolRunsThroughSandbox(t *testing.T) {
	stub := &stubSandbox{result: agentcore.SandboxResult{ExitCode: 0, Stdout: "hi\n"}}
	tool := NewShellTool(stub, agentcore.SandboxLimits{}, nil)

	if tool.Name() != ToolRunShell {
		t.Fatalf("name = %q, want %q", tool.Name(), ToolRunShell)
	}

	out, err := tool.Run(context.Background(), `{"command":"echo hi"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// The command must be executed via the sandbox, never the host.
	wantArgv := []string{"/bin/sh", "-c", "echo hi"}
	if strings.Join(stub.last.Argv, "\x00") != strings.Join(wantArgv, "\x00") {
		t.Fatalf("argv = %v, want %v", stub.last.Argv, wantArgv)
	}
	// Default limits are fail-closed: no network, read-only fs.
	if stub.last.Constraints.Network || stub.last.Constraints.WritableFS {
		t.Fatalf("expected fail-closed constraints, got %+v", stub.last.Constraints)
	}
	if !strings.Contains(out, "exit_code: 0") || !strings.Contains(out, "hi") {
		t.Fatalf("formatted output missing fields: %q", out)
	}
}

func TestShellToolMapsHostUserForWritableWorkspace(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	stub := &stubSandbox{result: agentcore.SandboxResult{ExitCode: 0}}
	tool := NewShellTool(stub, agentcore.SandboxLimits{}, ws)
	if _, err := tool.Run(context.Background(), `{"command":"touch output.txt"}`); err != nil {
		t.Fatal(err)
	}
	if !stub.last.Constraints.RunAsHostUser || stub.last.Constraints.WritableFS {
		t.Fatalf("workspace identity/root policy = %+v", stub.last.Constraints)
	}
}

// computer_use is the persistent, network-enabled, writable profile and must
// thread the conversation session id from the context onto the exec so the
// backend reuses one container across calls.
func TestComputerUseToolThreadsSessionAndLimits(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	stub := &stubSandbox{result: agentcore.SandboxResult{ExitCode: 0, Stdout: "ok\n"}}
	tool := NewComputerUseTool(stub, ws)

	if tool.Name() != ToolComputerUse {
		t.Fatalf("name = %q, want %q", tool.Name(), ToolComputerUse)
	}

	ctx := agentcore.WithSandboxSession(context.Background(), "conv-123")
	if _, err := tool.Run(ctx, `{"command":"pip install python-docx"}`); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if stub.last.Session != "conv-123" {
		t.Fatalf("session = %q, want conv-123", stub.last.Session)
	}
	// The computer-use envelope is the opposite of run_shell's lock: network on,
	// writable fs, so installs work.
	if !stub.last.Constraints.Network || !stub.last.Constraints.WritableFS {
		t.Fatalf("expected networked writable constraints, got %+v", stub.last.Constraints)
	}
	// Workspace is mounted read-write so produced files persist on the host.
	if len(stub.last.Mounts) != 1 || stub.last.Mounts[0].ReadOnly {
		t.Fatalf("expected one rw workspace mount, got %+v", stub.last.Mounts)
	}
}

// Without a session on the context the computer_use tool still runs (degrades to
// an ephemeral execution) rather than failing.
func TestComputerUseToolWithoutSessionDegrades(t *testing.T) {
	ws, _ := NewWorkspace(t.TempDir())
	stub := &stubSandbox{result: agentcore.SandboxResult{ExitCode: 0}}
	tool := NewComputerUseTool(stub, ws)
	if _, err := tool.Run(context.Background(), `{"command":"echo hi"}`); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if stub.last.Session != "" {
		t.Fatalf("expected empty session, got %q", stub.last.Session)
	}
}

// TestShellToolSpillsOversizedOutput: output past the visible cap is persisted
// to the workspace with a tail note naming the path (which middle-truncation
// preserves), so the overflow is recoverable instead of lost.
func TestShellToolSpillsOversizedOutput(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	huge := strings.Repeat("build log line\n", 4000) // ~60KB, over the 24KB spill threshold
	stub := &stubSandbox{result: agentcore.SandboxResult{ExitCode: 1, Stdout: huge, Stderr: "final error: it broke"}}
	tool := NewShellTool(stub, agentcore.SandboxLimits{}, ws)

	out, err := tool.Run(context.Background(), `{"command":"make"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, "full output saved to "+shellLogDir+"/") {
		t.Fatalf("expected spill note in output tail: %q", out[len(out)-300:])
	}
	// The note names a real, readable file containing the complete output.
	start := strings.Index(out, shellLogDir+"/")
	rel := out[start:]
	rel = rel[:strings.IndexAny(rel, "; ")]
	logOut, rerr := NewReadFileTool(nil, ws).Run(context.Background(), `{"path":"`+rel+`","offset":1,"limit":2}`)
	if rerr != nil {
		t.Fatalf("spilled log not readable via read_file: %v", rerr)
	}
	if !strings.Contains(logOut, "exit_code: 1") {
		t.Fatalf("spilled log missing content: %q", logOut)
	}
}

// TestShellToolSmallOutputDoesNotSpill: in-cap output must stay exactly as
// before — no note, no log file.
func TestShellToolSmallOutputDoesNotSpill(t *testing.T) {
	ws, _ := NewWorkspace(t.TempDir())
	stub := &stubSandbox{result: agentcore.SandboxResult{ExitCode: 0, Stdout: "ok\n"}}
	tool := NewShellTool(stub, agentcore.SandboxLimits{}, ws)
	out, err := tool.Run(context.Background(), `{"command":"echo ok"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if strings.Contains(out, "full output saved") {
		t.Fatalf("small output must not spill: %q", out)
	}
}

func TestShellToolRejectsEmptyAndBadArgs(t *testing.T) {
	tool := NewShellTool(&stubSandbox{}, agentcore.SandboxLimits{}, nil)
	if _, err := tool.Run(context.Background(), `{"command":"  "}`); err == nil {
		t.Fatal("expected error on empty command")
	}
	if _, err := tool.Run(context.Background(), `not json`); err == nil {
		t.Fatal("expected error on invalid JSON")
	}
}

// The sandbox is optional in both directions: with none injected, run_shell
// runs the command directly on the host machine instead of refusing.
func TestShellToolWithoutSandboxRunsOnHost(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	tool := NewShellTool(nil, agentcore.SandboxLimits{}, ws)
	out, err := tool.Run(context.Background(), `{"command":"echo hi"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, "exit_code: 0") || !strings.Contains(out, "hi") {
		t.Fatalf("host run output = %q, want exit 0 and 'hi'", out)
	}
}

// D2: host mode is guarded, not raw. A command that runs on the host must see
// only the env the SandboxExec declared — never the server process's own, which
// holds DB credentials and API keys a prompt-injected command would love.
func TestShellToolHostModeHidesHostEnv(t *testing.T) {
	t.Setenv("AGENTRAY_TEST_SECRET", "leaked-db-password")
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	tool := NewShellTool(nil, agentcore.SandboxLimits{}, ws)
	out, err := tool.Run(context.Background(), `{"command":"echo \"[$AGENTRAY_TEST_SECRET]\"; env"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if strings.Contains(out, "leaked-db-password") {
		t.Fatalf("host-mode shell leaked a host env var:\n%s", out)
	}
	if !strings.Contains(out, "[]") {
		t.Fatalf("expected the unset variable to expand to nothing, got:\n%s", out)
	}
}

// The host substrate runs in the workspace directory, so run_shell and the file
// tools keep sharing one filesystem without a bind mount to arrange it.
func TestShellToolHostModeRunsInWorkspace(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "marker.txt"), []byte("x"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	ws, err := NewWorkspace(dir)
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	out, err := NewShellTool(nil, agentcore.SandboxLimits{}, ws).Run(context.Background(), `{"command":"ls"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, "marker.txt") {
		t.Fatalf("host run did not start in the workspace: %q", out)
	}
}

// Without a workspace the host substrate must not start in the server's own
// working directory — it gets a throwaway scratch dir, the analogue of the
// backend's ephemeral tmpfs workdir.
func TestShellToolHostModeWithoutWorkspaceUsesScratch(t *testing.T) {
	out, err := NewShellTool(nil, agentcore.SandboxLimits{}, nil).Run(context.Background(), `{"command":"pwd; ls | wc -l"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	cwd, _ := os.Getwd()
	if strings.Contains(out, cwd) {
		t.Fatalf("host run started in the server's working directory:\n%s", out)
	}
	if !strings.Contains(out, "0") {
		t.Fatalf("expected an empty scratch dir, got:\n%s", out)
	}
}

// TimeoutSeconds is the one limit the host substrate can enforce, and it must:
// an unbounded command would otherwise pin the process forever.
func TestShellToolHostModeEnforcesTimeout(t *testing.T) {
	tool := NewShellTool(nil, agentcore.SandboxLimits{TimeoutSeconds: 0.25}, nil)
	out, err := tool.Run(context.Background(), `{"command":"sleep 5"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, "killed:") {
		t.Fatalf("expected the command to be killed, got:\n%s", out)
	}
}

// When a workspace is configured the shell must share it with the file tools:
// the workspace is bind-mounted read-write and is the command's working dir, so
// write_file → run_shell (and back) operate on one filesystem.
func TestShellToolMountsWorkspace(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	stub := &stubSandbox{result: agentcore.SandboxResult{ExitCode: 0}}
	tool := NewShellTool(stub, agentcore.SandboxLimits{}, ws)

	if _, err := tool.Run(context.Background(), `{"command":"ls"}`); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if stub.last.Workdir != shellWorkdir {
		t.Fatalf("workdir = %q, want %q", stub.last.Workdir, shellWorkdir)
	}
	if len(stub.last.Mounts) != 1 {
		t.Fatalf("expected 1 mount, got %d", len(stub.last.Mounts))
	}
	m := stub.last.Mounts[0]
	if m.Source != ws.Root() || m.Target != shellWorkdir || m.ReadOnly {
		t.Fatalf("mount = %+v, want rw %s->%s", m, ws.Root(), shellWorkdir)
	}
}

// The timeout must reap the whole process group, not just the direct child. A
// shell that backgrounded a job would otherwise keep the inherited stdout pipe
// open, so the "killed" run blocks until the orphan finishes.
func TestShellToolHostModeKillsBackgroundedChildren(t *testing.T) {
	tool := NewShellTool(nil, agentcore.SandboxLimits{TimeoutSeconds: 0.5}, nil)
	start := time.Now()
	out, err := tool.Run(context.Background(), `{"command":"sleep 30 & sleep 30"}`)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, "killed:") {
		t.Fatalf("expected the command to be killed, got:\n%s", out)
	}
	if elapsed > 10*time.Second {
		t.Fatalf("run took %s — the backgrounded child outlived the timeout", elapsed)
	}
}
