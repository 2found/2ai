package sandbox

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/2found/2ai/agentcore"
)

// ToolRunShell is the stable tool name the model calls to run a shell command.
// A consumer's Policy must permit this name before the model is shown the tool.
const ToolRunShell = "run_shell"

// ToolComputerUse is the higher-privilege, Claude-Code-level shell. Unlike
// run_shell (locked: ephemeral, read-only, no network), it runs in a persistent
// session container with network egress and a writable filesystem, so the agent
// can install tooling (pip/apt/npm) and have it — plus any files it writes —
// survive across calls: write code, run it, install a PDF/DOCX/XLSX parser, then
// produce a PDF/PPTX/HTML artifact. It is a deliberate, policy-granted capability
// distinct from run_shell so a project opts into it explicitly.
const ToolComputerUse = "computer_use"

// ShellTool lets the agent run a shell command on whichever substrate it was
// built with. With an injected Sandbox the command runs inside it and sees
// nothing of the host: no filesystem, no environment, no network unless
// granted. With no sandbox it falls back to HostSandbox and runs on the host —
// still with only the declared env visible (a prompt-injected `cat
// /proc/self/environ` cannot read the server's DB creds or API keys) and still
// under the timeout, but with the host's filesystem and network. Which
// substrate is appropriate is the caller's decision; see HostSandbox for what
// is and is not enforceable there.
type ShellTool struct {
	sb        agentcore.Sandbox
	limits    agentcore.SandboxLimits
	workspace *Workspace
	// hosted records that no sandbox was injected, so the model-facing
	// description does not promise isolation the substrate is not providing.
	hosted bool

	// name/description let one implementation back both the locked run_shell and
	// the persistent computer_use surface (the only behavioural fork is limits +
	// session awareness).
	name        string
	description string
	// persistent makes the tool reuse one session container across calls (keyed
	// by the conversation session on the context), so installs and files survive.
	persistent bool
}

// ComputerUseLimits is the envelope for the persistent computer_use shell:
// network on (to install tooling), writable filesystem (so package managers can
// write), and generous time/memory so a build, a LibreOffice conversion, or a
// document render can finish. It stays hard-isolated by the backend (no host
// env, all caps dropped, no-new-privileges, resource caps).
func ComputerUseLimits() agentcore.SandboxLimits {
	return agentcore.SandboxLimits{
		Network:        true,
		WritableFS:     true,
		MemoryMB:       2048,
		CPUs:           2,
		PidsLimit:      512,
		TimeoutSeconds: 300,
	}
}

// shellWorkdir is where the workspace is mounted inside the shell sandbox when a
// workspace is configured, so run_shell and the read_file/write_file tools share
// one filesystem (a script written with write_file is runnable, and shell output
// is readable back) — the coherent-workspace behaviour an agent expects.
const shellWorkdir = "/workspace"

const (
	// spillShellOutputBytes mirrors agentcore's default per-tool-result cap: past
	// it the loop middle-truncates what the model sees, so the full output is
	// first persisted to the workspace and the model told where to find it
	// (pi's full-output pattern) instead of the overflow being lost.
	spillShellOutputBytes = 24 * 1024
	// shellLogDir is the workspace-relative directory spilled outputs land in —
	// readable via read_file/grep, and under shellWorkdir inside the shell.
	shellLogDir = ".shell_logs"
)

// NewShellTool builds a run_shell tool over the given sandbox. sb is optional:
// nil runs the command directly on the host machine via HostSandbox (the
// default substrate for an embedded consumer), non-nil runs it inside the
// sandbox. limits is the per-call isolation envelope (the zero value is
// fail-closed: no network, read-only fs, default resource caps); on the host
// substrate only its timeout is enforceable. When ws is non-nil the agent
// workspace is bind-mounted read-write at shellWorkdir and becomes the
// command's working directory, so shell commands see the same files as the file
// tools; when nil the shell runs in an ephemeral, empty scratch dir.
func NewShellTool(sb agentcore.Sandbox, limits agentcore.SandboxLimits, ws *Workspace) *ShellTool {
	hosted := sb == nil
	// A locked container keeps its root read-only, but a workspace bind mount is
	// intentionally writable. Map the host UID/GID so ordinary 0755 workspaces
	// are actually writable and generated files are not owned by container-root.
	if ws != nil && !limits.WritableFS {
		limits.RunAsHostUser = true
	}
	desc := "Run a shell command inside an isolated sandbox (no host " +
		"filesystem, no host environment, no network unless granted). " +
		"Returns the combined exit code, stdout, and stderr."
	if hosted {
		desc = "Run a shell command on this machine, in the agent workspace " +
			"directory. The command does not inherit this process's environment " +
			"variables. Returns the combined exit code, stdout, and stderr."
		sb = NewHostSandbox()
	}
	return &ShellTool{
		sb:          sb,
		limits:      limits,
		workspace:   ws,
		hosted:      hosted,
		name:        ToolRunShell,
		description: desc,
	}
}

// NewComputerUseTool builds the persistent computer_use shell over sb, with the
// agent workspace mounted (required, so artifacts persist on the host) and the
// ComputerUseLimits envelope. sb is optional: nil runs the commands directly on
// the host machine, where "persistent" needs no session container because the
// host filesystem already outlives every call. With a sandbox it reuses one
// session container per conversation, so installs and written files survive
// across calls — the Claude-Code-level "write code, install a tool, run it,
// produce a document" loop.
// networkAllow, when non-empty, confines the session's egress to the listed
// hosts (and their subdomains) via the sandbox's filtering proxy (#5b). Empty
// keeps the current open-network behavior. It has no effect on the host
// substrate, which cannot filter egress.
func NewComputerUseTool(sb agentcore.Sandbox, ws *Workspace, networkAllow ...string) *ShellTool {
	limits := ComputerUseLimits()
	limits.NetworkAllow = networkAllow
	hosted := sb == nil
	desc := "Run a shell command in a persistent, network-enabled Linux " +
		"sandbox with a writable filesystem. State persists across calls in " +
		"the same conversation: install tools (pip/apt/npm), write and run " +
		"code, and produce files in the workspace (parse or generate PDF, " +
		"DOCX, XLSX, PPTX, HTML, etc.). The workspace is the working directory; " +
		"files written there are saved. Returns exit code, stdout, and stderr."
	if hosted {
		desc = "Run a shell command on this machine, in the agent workspace " +
			"directory. State persists across calls: install tools, write and run " +
			"code, and produce files in the workspace (parse or generate PDF, " +
			"DOCX, XLSX, PPTX, HTML, etc.). The command does not inherit this " +
			"process's environment variables. Returns exit code, stdout, and stderr."
		sb = NewHostSandbox()
	}
	return &ShellTool{
		sb:          sb,
		limits:      limits,
		workspace:   ws,
		hosted:      hosted,
		persistent:  true,
		name:        ToolComputerUse,
		description: desc,
	}
}

func (t *ShellTool) Name() string { return t.name }

func (t *ShellTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:        t.name,
		Description: t.description,
		Strict:      agentcore.ToolStrictEnabled,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{
					"type":        "string",
					"description": "The shell command to execute inside the sandbox.",
				},
			},
			"required": []string{"command"},
		},
	}
}

// Run is sequential-only (no ParallelTool): a shell command may mutate the
// session workdir, so concurrent runs are not opted into.
func (t *ShellTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("run_shell: invalid arguments: %w", err)
	}
	if strings.TrimSpace(in.Command) == "" {
		return "", fmt.Errorf("run_shell: empty command")
	}

	exec := agentcore.SandboxExec{
		Argv:        []string{"/bin/sh", "-c", in.Command},
		Constraints: t.limits,
	}
	// A persistent tool reuses one container per conversation so installs and
	// files survive between calls. The session id rides the context (set by the
	// runtime); absent it, the call degrades to an ephemeral run.
	if t.persistent {
		exec.Session = agentcore.SandboxSessionFrom(ctx)
	}
	// Share the agent workspace with the file tools: bind-mount it read-write and
	// run from it, so write_file → run_shell (and back) is coherent. The bind
	// mount stays writable even with the read-only root; isolation (no host env,
	// no network) is unchanged.
	if t.workspace != nil {
		exec.Mounts = []agentcore.SandboxMount{{
			Source:   t.workspace.Root(),
			Target:   shellWorkdir,
			ReadOnly: false,
		}}
		exec.Workdir = shellWorkdir
	}
	res, err := t.sb.Exec(ctx, exec)
	if err != nil {
		return "", fmt.Errorf("run_shell: %w", err)
	}
	out := formatResult(res)
	// Oversized output: persist the full text where the agent can read it before
	// the loop truncates the visible result. The note rides the tail of the
	// result, which middle-truncation preserves.
	if len(out) > spillShellOutputBytes && t.workspace != nil {
		if rel, serr := t.spillOutput(out); serr == nil {
			out += fmt.Sprintf("\n[output is %d bytes and will be truncated above — full output saved to %s; read it with read_file/grep, or as %s/%s in the shell]",
				len(out), rel, shellWorkdir, rel)
		}
	}
	return out, nil
}

// spillOutput writes the full formatted result into shellLogDir in the
// workspace and returns the workspace-relative path.
func (t *ShellTool) spillOutput(out string) (string, error) {
	var suffix [4]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		return "", err
	}
	rel := filepath.ToSlash(filepath.Join(shellLogDir, fmt.Sprintf("shell-%s.log", hex.EncodeToString(suffix[:]))))
	dir := filepath.Join(t.workspace.Root(), shellLogDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(t.workspace.Root(), filepath.FromSlash(rel)), []byte(out), 0o644); err != nil {
		return "", err
	}
	return rel, nil
}

// formatResult renders a SandboxResult into a compact, model-readable block.
func formatResult(res agentcore.SandboxResult) string {
	var b strings.Builder
	if res.Killed {
		fmt.Fprintf(&b, "killed: %s\n", res.KillReason)
	}
	fmt.Fprintf(&b, "exit_code: %d\n", res.ExitCode)
	if out := strings.TrimRight(res.Stdout, "\n"); out != "" {
		fmt.Fprintf(&b, "stdout:\n%s\n", out)
	}
	if errOut := strings.TrimRight(res.Stderr, "\n"); errOut != "" {
		fmt.Fprintf(&b, "stderr:\n%s\n", errOut)
	}
	return strings.TrimRight(b.String(), "\n")
}
