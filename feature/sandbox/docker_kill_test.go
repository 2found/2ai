package sandbox

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/core/agentcore"
)

// hangingDocker is a fake docker CLI whose workload commands ("run --rm" and
// "exec -i") hang until killed, while bookkeeping calls return at once.
func hangingDocker(t *testing.T) fakeDocker {
	return newFakeDocker(t, "", `case "$1 $2" in "run --rm"|"exec -i") exec sleep 30 ;; esac`)
}

// Killing the docker CLI does not stop the container, so a one-shot run that
// times out must be removed in the daemon, and the result must say why.
func TestDockerExecTimeoutRemovesContainer(t *testing.T) {
	fake := hangingDocker(t)
	sb := NewDockerSandbox(WithDockerBinary(fake.bin))
	start := time.Now()
	res, err := sb.Exec(context.Background(), agentcore.SandboxExec{
		Argv:        []string{"sleep", "30"},
		Constraints: agentcore.SandboxLimits{TimeoutSeconds: 0.3},
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if !res.Killed || res.KillReason != "exceeded 0.3s timeout" {
		t.Fatalf("result = %+v, want killed by the timeout", res)
	}
	if time.Since(start) > 5*time.Second {
		t.Fatalf("Exec took %v after the deadline", time.Since(start))
	}
	// The timeout can kill the fake CLI before its shell logs the "run" line,
	// so the run's name is checked only when it was logged.
	name := "agentray-sbx-"
	for _, c := range fake.calls(t) {
		if f := strings.Fields(c); len(f) > 4 && f[0] == "run" && f[3] == "--name" {
			name = f[4]
		}
	}
	if fake.called(t, "rm -f "+name) == 0 {
		t.Fatalf("container %q not removed; calls:\n%s", name, strings.Join(fake.calls(t), "\n"))
	}
}

// A caller's cancellation is an error (the run was abandoned, not finished),
// and still removes the container.
func TestDockerStartCancelRemovesContainer(t *testing.T) {
	fake := hangingDocker(t)
	sb := NewDockerSandbox(WithDockerBinary(fake.bin))
	ctx, cancel := context.WithCancel(context.Background())
	proc, err := sb.Start(ctx, agentcore.SandboxExec{Argv: []string{"sleep", "30"}})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	_ = proc.Stdin().Close()
	go func() { _, _ = io.Copy(io.Discard, proc.Stdout()) }()
	go func() { _, _ = io.Copy(io.Discard, proc.Stderr()) }()
	time.AfterFunc(100*time.Millisecond, cancel)
	res, _ := proc.Wait()
	if !res.Killed || res.KillReason != "cancelled" {
		t.Fatalf("result = %+v, want cancelled", res)
	}
	if fake.called(t, "rm -f agentray-sbx-") == 0 {
		t.Fatalf("container not removed; calls:\n%s", strings.Join(fake.calls(t), "\n"))
	}
}

// A session exec runs under the pid-recording wrapper, and a timeout kills its
// process tree inside the container (the container itself must survive: it
// holds the session's state).
func TestDockerSessionExecTimeoutKillsTreeNotContainer(t *testing.T) {
	fake := hangingDocker(t)
	sb := NewDockerSandbox(WithDockerBinary(fake.bin))
	res, err := sb.Exec(context.Background(), agentcore.SandboxExec{
		Argv:        []string{"sleep", "30"},
		Session:     "conv-1",
		Constraints: agentcore.SandboxLimits{TimeoutSeconds: 0.3, WritableFS: true},
	})
	if err != nil || !res.Killed {
		t.Fatalf("Exec = %+v, %v; want killed", res, err)
	}
	calls := strings.Join(fake.calls(t), "\n")
	if !strings.Contains(calls, "--tmpfs "+sessionRunDir+":rw,size=1m,mode=1777") {
		t.Fatalf("session container lacks the pidfile tmpfs:\n%s", calls)
	}
	var pidfile string
	for _, c := range fake.calls(t) {
		if strings.HasPrefix(c, "exec -i agentray-ses-conv-1 /bin/sh -c "+sessionExecWrapper+" ") {
			pidfile = strings.Fields(strings.TrimPrefix(c, "exec -i agentray-ses-conv-1 /bin/sh -c "+sessionExecWrapper+" "))[0]
		}
	}
	if !strings.HasPrefix(pidfile, sessionRunDir+"/") {
		t.Fatalf("session exec not wrapped with a pidfile:\n%s", calls)
	}
	// The kill script spans lines, so match it in the raw log.
	if strings.Count(calls, "exec agentray-ses-conv-1 /bin/sh -c "+sessionKillScript+" "+pidfile) != 1 {
		t.Fatalf("tree kill not run for %s:\n%s", pidfile, calls)
	}
	if fake.called(t, "rm -f agentray-ses-conv-1") != 1 { // only the pre-create clear
		t.Fatalf("a timed-out exec must not remove the session container:\n%s", calls)
	}
}

// The wrapper and kill script are shell, so run them for real: the kill must
// reach a grandchild, not just the wrapper. Needs /proc (Linux).
func TestSessionKillScriptKillsGrandchildren(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("needs /proc/<pid>/task/*/children")
	}
	pidfile := filepath.Join(t.TempDir(), "x.pid")
	childPid := filepath.Join(t.TempDir(), "child.pid")
	cmd := exec.Command("/bin/sh", "-c", sessionExecWrapper, pidfile, "/bin/sh", "-c", `sleep 30 & echo $! > "$1"; wait`, "sh", childPid)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		if b, err := os.ReadFile(childPid); err == nil && len(strings.TrimSpace(string(b))) > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("child never started")
		}
		time.Sleep(10 * time.Millisecond)
	}
	b, _ := os.ReadFile(childPid)
	sleeper := strings.TrimSpace(string(b))
	if out, err := exec.Command("/bin/sh", "-c", sessionKillScript, pidfile).CombinedOutput(); err != nil {
		t.Fatalf("kill script: %v: %s", err, out)
	}
	_ = cmd.Wait()
	if _, err := os.Stat("/proc/" + sleeper); err == nil {
		if state, _ := os.ReadFile("/proc/" + sleeper + "/stat"); !strings.Contains(string(state), ") Z ") {
			t.Fatalf("grandchild %s survived the kill", sleeper)
		}
	}
	if _, err := os.Stat(pidfile); !os.IsNotExist(err) {
		t.Fatalf("pidfile not removed: %v", err)
	}
}
