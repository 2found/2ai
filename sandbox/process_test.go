package sandbox

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
)

func TestHostSandboxInteractiveProcessRoundTrip(t *testing.T) {
	var _ agentcore.ProcessSandbox = NewHostSandbox()
	proc, err := NewHostSandbox().Start(context.Background(), agentcore.SandboxExec{
		Argv: []string{"/bin/sh", "-c", "cat"},
	})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	if _, err := io.WriteString(proc.Stdin(), "framed protocol\n"); err != nil {
		t.Fatalf("stdin: %v", err)
	}
	if err := proc.Stdin().Close(); err != nil {
		t.Fatalf("close stdin: %v", err)
	}
	out, err := io.ReadAll(proc.Stdout())
	if err != nil {
		t.Fatalf("stdout: %v", err)
	}
	res, err := proc.Wait()
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if res.ExitCode != 0 || string(out) != "framed protocol\n" {
		t.Fatalf("result=%+v stdout=%q", res, out)
	}
}

func TestDockerInteractiveUsesHardenedOneShotEnvelope(t *testing.T) {
	var _ agentcore.ProcessSandbox = NewDockerSandbox()
	sb := NewDockerSandbox(WithImage("lsp-image:test"))
	args, err := sb.oneShotArgs(agentcore.SandboxExec{
		Argv:    []string{"gopls", "serve"},
		Env:     map[string]string{"HOME": "/work"},
		Mounts:  []agentcore.SandboxMount{{Source: "/host/work", Target: "/workspace", ReadOnly: true}},
		Workdir: "/workspace",
	}, withDefaults(agentcore.SandboxLimits{}), "lsp-test")
	if err != nil {
		t.Fatalf("oneShotArgs: %v", err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"run --rm -i", "--cap-drop ALL", "--security-opt no-new-privileges",
		"--network none", "--read-only", "readonly", "--workdir /workspace",
		"--env HOME=/work", "lsp-image:test gopls serve",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("docker args missing %q: %s", want, joined)
		}
	}
}

func TestDockerEvalProcessUsesHardenedWritableWorkspaceEnvelope(t *testing.T) {
	sb := NewDockerSandbox(WithImage("fallback:test"))
	args, err := sb.oneShotArgs(agentcore.SandboxExec{
		Argv:    []string{"python3", "-u", "-c", "runner"},
		Env:     map[string]string{"HOME": "/work", "PYTHONUNBUFFERED": "1"},
		Image:   "python-eval:test",
		Mounts:  []agentcore.SandboxMount{{Source: "/host/work", Target: "/workspace"}},
		Workdir: "/workspace",
	}, withDefaults(agentcore.SandboxLimits{TimeoutSeconds: 3600, RunAsHostUser: true}), "eval-test")
	if err != nil {
		t.Fatalf("oneShotArgs: %v", err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"run --rm -i", "--cap-drop ALL", "--security-opt no-new-privileges",
		"--network none", "--read-only", "--workdir /workspace",
		"--mount type=bind,src=/host/work,dst=/workspace",
		"python-eval:test python3 -u -c runner",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("docker args missing %q: %s", want, joined)
		}
	}
	if strings.Contains(joined, "dst=/workspace,readonly") {
		t.Fatalf("eval workspace must be writable: %s", joined)
	}
	if user := sandboxUser(agentcore.SandboxLimits{RunAsHostUser: true}); !strings.Contains(joined, "--user "+user) {
		t.Errorf("docker args missing mapped workspace user %q: %s", user, joined)
	}
	if tmpfs := sandboxTmpfs(agentcore.SandboxLimits{RunAsHostUser: true}); !strings.Contains(joined, "--tmpfs "+tmpfs) {
		t.Errorf("docker args tmpfs ownership does not match mapped user %q: %s", tmpfs, joined)
	}
}
