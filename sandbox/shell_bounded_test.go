package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
)

func TestShellTimeoutArgument(t *testing.T) {
	stub := &stubSandbox{}
	tool := NewShellTool(stub, agentcore.SandboxLimits{}, nil)
	run := func(args string) error { _, err := tool.Run(context.Background(), args); return err }

	if err := run(`{"command":"true"}`); err != nil || stub.last.Constraints.TimeoutSeconds != shellDefaultTimeoutS {
		t.Fatalf("default timeout = %v (%v), want %v", stub.last.Constraints.TimeoutSeconds, err, shellDefaultTimeoutS)
	}
	if err := run(`{"command":"true","timeout_seconds":30}`); err != nil || stub.last.Constraints.TimeoutSeconds != 30 {
		t.Fatalf("explicit timeout = %v (%v), want 30", stub.last.Constraints.TimeoutSeconds, err)
	}
	if err := run(`{"command":"true","timeout_seconds":600}`); err != nil {
		t.Fatalf("the maximum itself must be accepted: %v", err)
	}
	for _, bad := range []string{"0", "-5", "601", "100000"} {
		stub.last = agentcore.SandboxExec{}
		if err := run(`{"command":"true","timeout_seconds":` + bad + `}`); err == nil || !strings.Contains(err.Error(), "between 1 and 600") {
			t.Errorf("timeout_seconds=%s: err = %v, want a bounds error", bad, err)
		}
		if stub.last.Argv != nil {
			t.Errorf("timeout_seconds=%s: refused call still ran", bad)
		}
	}

	// A configured timeout is the default, and raises the maximum when larger.
	long := NewShellTool(stub, agentcore.SandboxLimits{TimeoutSeconds: 900}, nil)
	if _, err := long.Run(context.Background(), `{"command":"true","timeout_seconds":900}`); err != nil {
		t.Fatalf("configured 900s must allow 900: %v", err)
	}
	// A negative configured timeout hands the deadline to the host: no cap.
	unbounded := NewShellTool(stub, agentcore.SandboxLimits{TimeoutSeconds: -1}, nil)
	if _, err := unbounded.Run(context.Background(), `{"command":"true"}`); err != nil || stub.last.Constraints.TimeoutSeconds != -1 {
		t.Fatalf("host-owned deadline = %v (%v), want -1", stub.last.Constraints.TimeoutSeconds, err)
	}
	if _, err := unbounded.Run(context.Background(), `{"command":"true","timeout_seconds":5000}`); err != nil {
		t.Fatalf("no maximum with a host-owned deadline: %v", err)
	}
}

func TestHeadTailKeepsEndsAndCountsTheMiddle(t *testing.T) {
	b := headTail{headMax: 4, tailMax: 4}
	for _, chunk := range []string{"ab", "cdef", strings.Repeat("x", 100), "ghij"} {
		b.write([]byte(chunk))
	}
	if got, want := b.String(), "abcd\n[... 102 bytes omitted ...]\nghij"; got != want {
		t.Fatalf("String() = %q, want %q", got, want)
	}
	small := headTail{headMax: 4, tailMax: 4}
	small.write([]byte("abcdefg"))
	if got := small.String(); got != "abcdefg" {
		t.Fatalf("no marker when nothing is dropped: %q", got)
	}
	// Cuts never split a multi-byte rune; the split bytes count as omitted.
	u := headTail{headMax: 5, tailMax: 5}
	u.write([]byte("ééé" + strings.Repeat("-", 20) + "ééé"))
	got := u.String()
	if !utf8.ValidString(got) {
		t.Fatalf("output is not valid UTF-8: %q", got)
	}
	if !strings.HasPrefix(got, "éé\n[... 24 bytes omitted ...]\n") || !strings.HasSuffix(got, "éé") {
		t.Fatalf("UTF-8 cut = %q", got)
	}
}

func TestShellSmallOutputLeavesNoLog(t *testing.T) {
	ws := mustWorkspace(t)
	stub := &stubSandbox{result: agentcore.SandboxResult{Stdout: "ok\n"}}
	if _, err := NewShellTool(stub, agentcore.SandboxLimits{}, ws).Run(context.Background(), `{"command":"echo ok"}`); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(filepath.Join(ws.Root(), shellLogDir))
	if len(entries) != 0 {
		t.Fatalf("a short output left %d log files", len(entries))
	}
}

// Streaming from the host: memory is bounded by the windows, the visible result
// is head + marker + tail, and the log holds every byte.
func TestShellHostStreamsHugeOutputBounded(t *testing.T) {
	skipOnWindows(t)
	ws := mustWorkspace(t)
	tool := NewShellTool(nil, agentcore.SandboxLimits{}, ws)
	out, err := tool.Run(context.Background(), `{"command":"yes 0123456789 | head -c 2000000; echo; echo the-end >&2"}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) > shellHeadBytes+shellTailBytes+1024 {
		t.Fatalf("visible output is %d bytes", len(out))
	}
	if !strings.Contains(out, "bytes omitted ...]") || !strings.Contains(out, "stderr:\nthe-end") {
		t.Fatalf("output = %q", out[len(out)-400:])
	}
	start := strings.Index(out, shellLogDir+"/")
	rel := out[start : start+strings.IndexByte(out[start:], ';')]
	info, err := os.Stat(filepath.Join(ws.Root(), rel))
	if err != nil {
		t.Fatalf("log: %v", err)
	}
	if want := int64(2000000 + 1 + len("the-end\n")); info.Size() != want {
		t.Fatalf("log holds %d bytes, want %d", info.Size(), want)
	}
}

// A timeout kills the whole process group, not just the shell: a backgrounded
// child must not outlive the call.
func TestShellHostTimeoutKillsProcessGroup(t *testing.T) {
	skipOnWindows(t)
	ws := mustWorkspace(t)
	tool := NewShellTool(nil, agentcore.SandboxLimits{}, ws)
	start := time.Now()
	out, err := tool.Run(context.Background(), `{"command":"sleep 60 & echo child=$!; wait","timeout_seconds":1}`)
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(start) > 4*time.Second {
		t.Fatalf("timeout took %v", time.Since(start))
	}
	if !strings.Contains(out, "killed: "+hostDeadlineReason) {
		t.Fatalf("output = %q", out)
	}
	i := strings.Index(out, "child=")
	if i < 0 {
		t.Fatalf("no child pid in %q", out)
	}
	pid, _ := strconv.Atoi(strings.Fields(out[i+len("child="):])[0])
	deadline := time.Now().Add(3 * time.Second)
	for syscall.Kill(pid, 0) == nil {
		if time.Now().After(deadline) {
			_ = syscall.Kill(pid, syscall.SIGKILL)
			t.Fatalf("child %d survived the timeout", pid)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestShellHostCancellationIsAnError(t *testing.T) {
	skipOnWindows(t)
	tool := NewShellTool(nil, agentcore.SandboxLimits{}, mustWorkspace(t))
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := tool.Run(ctx, `{"command":"sleep 30"}`); err == nil {
		t.Fatal("a cancelled call must return an error, not a result")
	}
}

func TestShellStreamingReportsProgress(t *testing.T) {
	skipOnWindows(t)
	tool := NewShellTool(nil, agentcore.SandboxLimits{}, mustWorkspace(t))
	var mu sync.Mutex
	var notes []string
	out, err := tool.RunStreaming(context.Background(), `{"command":"echo first; sleep 1.2; echo second"}`, func(s string) {
		mu.Lock()
		notes = append(notes, s)
		mu.Unlock()
	})
	if err != nil || !strings.Contains(out, "first\nsecond") {
		t.Fatalf("out = %q, err = %v", out, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(notes) == 0 || !strings.Contains(notes[0], "first") || strings.Contains(notes[0], "second") {
		t.Fatalf("progress = %q, want an early note with only the first line", notes)
	}
}

// hungProcess is a backend whose pipes stay open after Kill, as when a daemon
// escaped the process group and inherited them.
type hungProcess struct {
	stdoutR, stderrR *io.PipeReader
	stdoutW, stderrW *io.PipeWriter
	killed           chan struct{}
	once             sync.Once
}

type hungSandbox struct{ proc *hungProcess }

func (h *hungSandbox) Exec(context.Context, agentcore.SandboxExec) (agentcore.SandboxResult, error) {
	return agentcore.SandboxResult{}, fmt.Errorf("unused")
}

func (h *hungSandbox) Start(context.Context, agentcore.SandboxExec) (agentcore.SandboxProcess, error) {
	or, ow := io.Pipe()
	er, ew := io.Pipe()
	h.proc = &hungProcess{stdoutR: or, stdoutW: ow, stderrR: er, stderrW: ew, killed: make(chan struct{})}
	return h.proc, nil
}

func (p *hungProcess) Stdin() io.WriteCloser { return nopWriteCloser{io.Discard} }
func (p *hungProcess) Stdout() io.ReadCloser { return p.stdoutR }
func (p *hungProcess) Stderr() io.ReadCloser { return p.stderrR }
func (p *hungProcess) Kill() error           { p.once.Do(func() { close(p.killed) }); return nil }
func (p *hungProcess) Wait() (agentcore.SandboxResult, error) {
	<-p.killed
	return agentcore.SandboxResult{Killed: true, KillReason: "exceeded 0.2s timeout"}, nil
}

func TestShellWatchdogClosesPipesHeldPastTheKill(t *testing.T) {
	grace := shellPipeGrace
	shellPipeGrace = 100 * time.Millisecond
	t.Cleanup(func() { shellPipeGrace = grace })
	sb := &hungSandbox{}
	tool := NewShellTool(sb, agentcore.SandboxLimits{}, nil)
	done := make(chan string, 1)
	go func() {
		out, _ := tool.Run(context.Background(), `{"command":"daemon","timeout_seconds":0.2}`)
		done <- out
	}()
	select {
	case out := <-done:
		if !strings.Contains(out, "killed: exceeded 0.2s timeout") {
			t.Fatalf("out = %q", out)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("run_shell hung on pipes held open past the kill")
	}
}

func TestShellBackgroundNeedsALauncher(t *testing.T) {
	ws := mustWorkspace(t)
	stub := &stubSandbox{}
	_, err := NewShellTool(stub, agentcore.SandboxLimits{}, ws).Run(context.Background(), `{"command":"sleep 1","run_in_background":true}`)
	if err == nil || !strings.Contains(err.Error(), "not enabled") {
		t.Fatalf("err = %v, want background-not-enabled", err)
	}
	if stub.last.Argv != nil {
		t.Fatal("the command ran in the foreground anyway")
	}
	if entries, _ := os.ReadDir(filepath.Join(ws.Root(), shellLogDir)); len(entries) != 0 {
		t.Fatalf("a refused background run left %d log files", len(entries))
	}
}

func TestShellBackgroundStreamsToItsLog(t *testing.T) {
	ws := mustWorkspace(t)
	stub := &stubSandbox{result: agentcore.SandboxResult{Stdout: "serving on :8080\n"}}
	var job func(context.Context) (string, error)
	var gotTool, gotLabel string
	ctx := agentcore.WithBackgroundLauncher(context.Background(), func(tool, label string, run func(context.Context) (string, error)) (json.RawMessage, error) {
		gotTool, gotLabel, job = tool, label, run
		return json.RawMessage(`{"job_id":"j1"}`), nil
	})
	cmd := "python3 -m http.server " + strings.Repeat("x", 100)
	out, err := NewShellTool(stub, agentcore.SandboxLimits{}, ws).Run(ctx, `{"command":"`+cmd+`","run_in_background":true}`)
	if err != nil {
		t.Fatal(err)
	}
	if gotTool != ToolRunShell || len(gotLabel) > 80 || !strings.HasPrefix(gotLabel, "python3 -m http.server") {
		t.Fatalf("launch(%q, %q)", gotTool, gotLabel)
	}
	if !strings.Contains(out, `{"job_id":"j1"}`) || !strings.Contains(out, "live output: "+shellLogDir+"/") {
		t.Fatalf("receipt = %q", out)
	}
	if stub.last.Argv != nil {
		t.Fatal("the command must run in the job, not before the receipt")
	}
	res, err := job(context.Background())
	if err != nil || !strings.Contains(res, "serving on :8080") {
		t.Fatalf("job = %q, %v", res, err)
	}
	if stub.last.Constraints.TimeoutSeconds != shellMaxTimeoutS {
		t.Fatalf("background timeout = %v, want the maximum", stub.last.Constraints.TimeoutSeconds)
	}
	start := strings.Index(out, shellLogDir+"/")
	rel := strings.Fields(out[start:])[0]
	// Kept even though short: its path was already handed to the model.
	if data, err := os.ReadFile(filepath.Join(ws.Root(), rel)); err != nil || string(data) != "serving on :8080\n" {
		t.Fatalf("log = %q, %v", data, err)
	}
}

func skipOnWindows(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("POSIX shell")
	}
}

// Inside a background job the command's output streams to the job as it is
// produced, bounded by the job's own buffer; the same command run in the
// foreground publishes nothing.
func TestShellBackgroundPublishesLiveOutputToTheJob(t *testing.T) {
	skipOnWindows(t)
	var job func(context.Context) (string, error)
	ctx := agentcore.WithBackgroundLauncher(context.Background(), func(_, _ string, run func(context.Context) (string, error)) (json.RawMessage, error) {
		job = run
		return json.RawMessage(`{"id":"job_1"}`), nil
	})
	tool := NewShellTool(nil, agentcore.SandboxLimits{}, nil)
	if _, err := tool.Run(ctx, `{"command":"echo first; sleep 0.3; echo second >&2; sleep 0.3; echo third","run_in_background":true}`); err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var published strings.Builder
	var sawFirstBeforeEnd bool
	jobCtx := agentcore.WithBackgroundJob(context.Background(), agentcore.BackgroundJob{Output: func(text string) {
		mu.Lock()
		defer mu.Unlock()
		published.WriteString(text)
	}})
	done := make(chan string, 1)
	go func() { res, _ := job(jobCtx); done <- res }()
	deadline := time.After(10 * time.Second)
	for !sawFirstBeforeEnd {
		select {
		case <-done:
			t.Fatal("the job ended before its first output was visible")
		case <-deadline:
			t.Fatal("no live output")
		case <-time.After(10 * time.Millisecond):
		}
		mu.Lock()
		sawFirstBeforeEnd = strings.Contains(published.String(), "first")
		mu.Unlock()
	}
	res := <-done
	mu.Lock()
	got := published.String()
	mu.Unlock()
	for _, want := range []string{"first", "second", "third"} {
		if !strings.Contains(got, want) || !strings.Contains(res, want) {
			t.Fatalf("%q missing from live output %q or result %q", want, got, res)
		}
	}

	// A foreground call has no job: nothing is published and nothing breaks.
	if out, err := tool.Run(context.Background(), `{"command":"echo sync"}`); err != nil || !strings.Contains(out, "sync") {
		t.Fatalf("foreground = %q, %v", out, err)
	}
	if strings.Contains(published.String(), "sync") {
		t.Fatal("a foreground command published to a job")
	}
}
