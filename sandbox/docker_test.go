package sandbox

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

// newTestSandbox returns a DockerSandbox, skipping the test when docker or the
// sandbox image is unavailable (CI without a runtime, offline, etc.).
func newTestSandbox(t *testing.T) *DockerSandbox {
	t.Helper()
	sb := NewDockerSandbox()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if !sb.Available(ctx) {
		t.Skip("docker not available — skipping isolation integration test")
	}
	// Ensure the image is present; pull once if not. Skip if the pull fails
	// (offline) rather than failing the suite.
	if exec.CommandContext(ctx, "docker", "image", "inspect", sb.Image()).Run() != nil {
		pull, pcancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer pcancel()
		if out, err := exec.CommandContext(pull, "docker", "pull", sb.Image()).CombinedOutput(); err != nil {
			t.Skipf("cannot pull %s (%v): %s", sb.Image(), err, out)
		}
	}
	return sb
}

func sh(sb *DockerSandbox, t *testing.T, script string, lim agentcore.SandboxLimits) agentcore.SandboxResult {
	t.Helper()
	res, err := sb.Exec(context.Background(), agentcore.SandboxExec{
		Argv:        []string{"/bin/sh", "-c", script},
		Constraints: lim,
	})
	if err != nil {
		t.Fatalf("Exec(%q): %v", script, err)
	}
	return res
}

// The headline guarantee: a secret in the HOST process environment is invisible
// inside the sandbox. This is exactly the agentray risk — agent in-process can
// read os.Getenv (DB creds, API keys); sandboxed, it cannot.
func TestDockerSandboxDoesNotLeakHostEnv(t *testing.T) {
	sb := newTestSandbox(t)
	os.Setenv("AGENTRAY_FAKE_SECRET", "leak-me-please")
	defer os.Unsetenv("AGENTRAY_FAKE_SECRET")

	res := sh(sb, t, "printenv; cat /proc/self/environ 2>/dev/null | tr '\\0' '\\n'", agentcore.SandboxLimits{})
	if strings.Contains(res.Stdout, "leak-me-please") || strings.Contains(res.Stdout, "AGENTRAY_FAKE_SECRET") {
		t.Fatalf("host secret leaked into sandbox env:\n%s", res.Stdout)
	}
}

// Only explicitly-passed env reaches the container.
func TestDockerSandboxPassesExplicitEnvOnly(t *testing.T) {
	sb := newTestSandbox(t)
	res, err := sb.Exec(context.Background(), agentcore.SandboxExec{
		Argv: []string{"/bin/sh", "-c", "echo $GREETING"},
		Env:  map[string]string{"GREETING": "hello-sandbox"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.Stdout, "hello-sandbox") {
		t.Fatalf("explicit env not visible: %q", res.Stdout)
	}
}

// Default is no network: egress must fail.
func TestDockerSandboxNoNetworkByDefault(t *testing.T) {
	sb := newTestSandbox(t)
	res := sh(sb, t, "wget -q -T 3 -O- http://1.1.1.1 >/dev/null 2>&1 && echo HAS_NET || echo NO_NET", agentcore.SandboxLimits{})
	if !strings.Contains(res.Stdout, "NO_NET") {
		t.Fatalf("expected no network egress, got: %q", res.Stdout)
	}
}

// Root filesystem is read-only; only the workdir is writable.
func TestDockerSandboxReadOnlyRoot(t *testing.T) {
	sb := newTestSandbox(t)
	res := sh(sb, t,
		"touch /etc/should_fail 2>/dev/null && echo ROOT_WRITABLE || echo ROOT_RO; "+
			"touch /work/ok 2>/dev/null && echo WORK_WRITABLE || echo WORK_RO",
		agentcore.SandboxLimits{})
	if !strings.Contains(res.Stdout, "ROOT_RO") {
		t.Fatalf("expected read-only root, got: %q", res.Stdout)
	}
	if !strings.Contains(res.Stdout, "WORK_WRITABLE") {
		t.Fatalf("expected writable workdir, got: %q", res.Stdout)
	}
}

// Non-zero exit codes propagate as results, not errors.
func TestDockerSandboxExitCode(t *testing.T) {
	sb := newTestSandbox(t)
	res := sh(sb, t, "exit 7", agentcore.SandboxLimits{})
	if res.ExitCode != 7 {
		t.Fatalf("exit code = %d, want 7", res.ExitCode)
	}
}

// A persistent session reuses one container across Exec calls: a file written in
// the first call is visible in the second. This is the property that lets a
// computer-use agent install a tool, then use it.
func TestDockerSandboxSessionPersistsState(t *testing.T) {
	sb := newTestSandbox(t)
	sess := "test-" + randHex(4)
	defer sb.CloseSession(sess)

	lim := agentcore.SandboxLimits{WritableFS: true, TimeoutSeconds: 30}
	first, err := sb.Exec(context.Background(), agentcore.SandboxExec{
		Argv:        []string{"/bin/sh", "-c", "echo persisted > /tmp/marker"},
		Session:     sess,
		Constraints: lim,
	})
	if err != nil {
		t.Fatalf("first exec: %v", err)
	}
	if first.ExitCode != 0 {
		t.Fatalf("first exec exit = %d, stderr=%q", first.ExitCode, first.Stderr)
	}
	second, err := sb.Exec(context.Background(), agentcore.SandboxExec{
		Argv:        []string{"/bin/sh", "-c", "cat /tmp/marker"},
		Session:     sess,
		Constraints: lim,
	})
	if err != nil {
		t.Fatalf("second exec: %v", err)
	}
	if !strings.Contains(second.Stdout, "persisted") {
		t.Fatalf("state did not persist across session calls: %q / stderr %q", second.Stdout, second.Stderr)
	}
}

// CloseSession reaps the container; a subsequent call transparently recreates a
// fresh one (so state does NOT survive a close).
func TestDockerSandboxCloseSessionResets(t *testing.T) {
	sb := newTestSandbox(t)
	sess := "test-" + randHex(4)
	defer sb.CloseSession(sess)

	lim := agentcore.SandboxLimits{WritableFS: true, TimeoutSeconds: 30}
	_, _ = sb.Exec(context.Background(), agentcore.SandboxExec{
		Argv: []string{"/bin/sh", "-c", "echo x > /tmp/marker"}, Session: sess, Constraints: lim,
	})
	if err := sb.CloseSession(sess); err != nil {
		t.Fatalf("CloseSession: %v", err)
	}
	res, err := sb.Exec(context.Background(), agentcore.SandboxExec{
		Argv: []string{"/bin/sh", "-c", "cat /tmp/marker 2>/dev/null; echo done"}, Session: sess, Constraints: lim,
	})
	if err != nil {
		t.Fatalf("post-close exec: %v", err)
	}
	if strings.Contains(res.Stdout, "x") {
		t.Fatalf("marker survived CloseSession: %q", res.Stdout)
	}
}

// A command exceeding its timeout is killed and reported.
func TestDockerSandboxTimeoutKills(t *testing.T) {
	sb := newTestSandbox(t)
	res := sh(sb, t, "sleep 30", agentcore.SandboxLimits{TimeoutSeconds: 2})
	if !res.Killed {
		t.Fatalf("expected killed result, got %+v", res)
	}
}

// The parallelism guarantee: many agents/runs executing concurrently each get
// their own container, so one session's writes are invisible to every other.
// Each session id keys a distinct container (agentray-ses-<id>), the registry
// map is mutex-guarded, and writable state lives only inside that container — so
// N runs in flight at once never read or clobber a sibling's files. This is what
// lets the platform fan out agents without a shared-sandbox conflict.
func TestDockerSandboxParallelSessionsAreIsolated(t *testing.T) {
	sb := newTestSandbox(t)
	const n = 6
	lim := agentcore.SandboxLimits{WritableFS: true, TimeoutSeconds: 30}

	var wg sync.WaitGroup
	errs := make([]error, n)
	readback := make([]string, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sess := fmt.Sprintf("par-%d-%s", i, randHex(4))
			defer sb.CloseSession(sess)
			// Each goroutine writes a secret unique to its session, then reads it
			// back. Distinct sessions must each see only their own value.
			secret := fmt.Sprintf("secret-%d-%s", i, randHex(6))
			if _, err := sb.Exec(context.Background(), agentcore.SandboxExec{
				Argv:        []string{"/bin/sh", "-c", "echo " + secret + " > /tmp/marker"},
				Session:     sess,
				Constraints: lim,
			}); err != nil {
				errs[i] = err
				return
			}
			res, err := sb.Exec(context.Background(), agentcore.SandboxExec{
				Argv:        []string{"/bin/sh", "-c", "cat /tmp/marker"},
				Session:     sess,
				Constraints: lim,
			})
			if err != nil {
				errs[i] = err
				return
			}
			readback[i] = strings.TrimSpace(res.Stdout)
			if readback[i] != secret {
				errs[i] = fmt.Errorf("session %d read %q, want its own %q", i, readback[i], secret)
			}
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("parallel session %d: %v", i, err)
		}
	}
	// No two sessions read the same value: a cross-session leak would surface as a
	// duplicate marker here even if each read happened to match a (leaked) secret.
	seen := map[string]int{}
	for i, v := range readback {
		if prev, dup := seen[v]; dup {
			t.Fatalf("sessions %d and %d shared marker %q — containers not isolated", prev, i, v)
		}
		seen[v] = i
	}
}
