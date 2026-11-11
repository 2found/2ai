package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"
	"time"

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
// writable filesystem (so package managers can write), and generous
// time/memory so a build, a LibreOffice conversion, or a document render can
// finish. Network egress is granted only to networkAllow (hosts and their
// subdomains, via the sandbox's filtering proxy); pass AnyHost for open
// egress. Empty means offline — never silently open. It stays hard-isolated by
// the backend (no host env, all caps dropped, no-new-privileges, resource
// caps).
func ComputerUseLimits(networkAllow ...string) agentcore.SandboxLimits {
	return agentcore.SandboxLimits{
		Network:        len(networkAllow) > 0,
		NetworkAllow:   networkAllow,
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
	// shellDefaultTimeoutS applies when neither the call nor the tool's limits
	// name a timeout; shellMaxTimeoutS is the most a call may ask for (or the
	// configured timeout, if larger). A background command defaults to the max.
	shellDefaultTimeoutS = 120.0
	shellMaxTimeoutS     = 600.0
	// Each stream keeps its first shellHeadBytes and last shellTailBytes; the
	// middle is replaced by an omitted-bytes marker. Both streams together stay
	// under agentcore's default per-tool-result cap, so the loop never has to
	// truncate a shell result itself — and the tail, where errors land, always
	// survives.
	shellHeadBytes = 4 * 1024
	shellTailBytes = 8 * 1024
	// shellLogDir is the workspace-relative directory full outputs land in —
	// readable via read_file/grep, and from the shell's working directory.
	shellLogDir = ".shell_logs"
	// A streaming run reports the newest shellProgressBytes of output at most
	// every shellProgressInterval.
	shellProgressBytes    = 2 * 1024
	shellProgressInterval = 500 * time.Millisecond
)

// shellPipeGrace is how long output may keep flowing after the command was
// killed (a daemon that escaped the process group can hold the pipes open)
// before the pipes are closed under it. A variable only so a test can shorten
// it.
var shellPipeGrace = 5 * time.Second

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
		description: desc + shellUsage,
	}
}

// shellUsage is the part of the description both shells share.
const shellUsage = " Output is bounded: each stream keeps its head and tail, and when a workspace is " +
	"configured the full output is saved to a log file you can read. Use timeout_seconds for long " +
	"commands, and run_in_background for servers or watchers (when background jobs are enabled)."

// NewComputerUseTool builds the persistent computer_use shell over sb, with the
// agent workspace mounted (required, so artifacts persist on the host) and the
// ComputerUseLimits envelope. sb is optional: nil runs the commands directly on
// the host machine, where "persistent" needs no session container because the
// host filesystem already outlives every call. With a sandbox it reuses one
// session container per conversation, so installs and written files survive
// across calls — the Claude-Code-level "write code, install a tool, run it,
// produce a document" loop.
//
// networkAllow confines the session's egress to the listed hosts (and their
// subdomains) via the sandbox's filtering proxy; AnyHost grants open egress;
// empty runs offline. It has no effect on the host substrate, which cannot
// filter egress.
func NewComputerUseTool(sb agentcore.Sandbox, ws *Workspace, networkAllow ...string) *ShellTool {
	limits := ComputerUseLimits(networkAllow...)
	hosted := sb == nil
	network := "network egress limited to an allowlist"
	switch {
	case !limits.Network:
		network = "no network"
	case newEgressAllow(networkAllow).unrestricted:
		network = "network egress"
	}
	desc := "Run a shell command in a persistent Linux sandbox (" + network + ") with a " +
		"writable filesystem. State persists across calls in " +
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
		description: desc + shellUsage,
	}
}

func (t *ShellTool) Name() string { return t.name }

// ExternalContent marks command output as untrusted (agentcore.ExternalContentTool)
// when the command can reach the network: curl, package installs and git
// clones bring in text nobody in the run wrote. Without network the output
// derives from the workspace and the agent's own commands, like read_file, and
// is left unframed so the frame stays a signal rather than noise on every call.
func (t *ShellTool) ExternalContent() bool { return t.hosted || t.limits.Network }

// NetworkHosts reports the sandbox's egress allowlist (agentcore.NetworkTool).
// A command can contact any host egress permits, so a host-scoped credential
// reaches it only when every allowlisted host is granted; a shell on the host
// machine or with open egress never receives one.
func (t *ShellTool) NetworkHosts(string) ([]string, error) {
	return egressHosts(t.name, t.hosted, t.limits)
}

func (t *ShellTool) Schema() agentcore.ToolSchema {
	def, max := t.timeouts()
	timeout := "Kill the command after this many seconds."
	if def > 0 {
		timeout = fmt.Sprintf("Kill the command after this many seconds. Defaults to %.0f (background: %.0f); at most %.0f.", def, max, max)
	}
	return agentcore.ToolSchema{
		Name:        t.name,
		Description: t.description,
		Strict:      agentcore.ToolStrictEnabled,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{
					"type":        "string",
					"description": "The shell command to execute (run with /bin/sh -c).",
				},
				"timeout_seconds": map[string]any{"type": "integer", "minimum": 1, "description": timeout},
				"run_in_background": map[string]any{
					"type": "boolean",
					"description": "Start the command as a background job and return at once with its job id " +
						"(and, with a workspace, the live log path). Use for servers, watchers and long builds.",
				},
			},
			"required": []string{"command"},
		},
	}
}

// timeouts returns the default and maximum per-call timeout in seconds. A
// negative configured timeout means the host owns the deadline: no default and
// no maximum.
func (t *ShellTool) timeouts() (def, max float64) {
	switch c := t.limits.TimeoutSeconds; {
	case c < 0:
		return -1, -1
	case c > 0:
		def = c
	default:
		def = shellDefaultTimeoutS
	}
	return def, math.Max(def, shellMaxTimeoutS)
}

// Run is sequential-only (no ParallelTool): a shell command may mutate the
// session workdir, so concurrent runs are not opted into.
func (t *ShellTool) Run(ctx context.Context, args string) (string, error) {
	return t.RunStreaming(ctx, args, nil)
}

// RunStreaming runs the command and, while it runs, reports the newest output
// through emit (nil emits nothing). The returned string is the final result.
func (t *ShellTool) RunStreaming(ctx context.Context, args string, emit func(string)) (string, error) {
	var in struct {
		Command         string   `json:"command"`
		TimeoutSeconds  *float64 `json:"timeout_seconds"`
		RunInBackground bool     `json:"run_in_background"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("%s: invalid arguments: %w", t.name, err)
	}
	if strings.TrimSpace(in.Command) == "" {
		return "", fmt.Errorf("%s: empty command", t.name)
	}
	def, max := t.timeouts()
	timeout := def
	if in.RunInBackground {
		timeout = max
	}
	if in.TimeoutSeconds != nil {
		timeout = *in.TimeoutSeconds
		if timeout <= 0 || (max > 0 && timeout > max) {
			return "", fmt.Errorf("%s: timeout_seconds must be between 1 and %.0f", t.name, max)
		}
	}
	req := t.request(ctx, in.Command, timeout)
	if in.RunInBackground {
		return t.background(ctx, req, in.Command)
	}
	var log *shellLog
	if t.workspace != nil {
		log = &shellLog{ws: t.workspace, threshold: shellHeadBytes + shellTailBytes}
	}
	return t.execute(ctx, req, emit, log)
}

// request builds the sandbox execution for one command.
func (t *ShellTool) request(ctx context.Context, command string, timeout float64) agentcore.SandboxExec {
	limits := t.limits
	limits.TimeoutSeconds = timeout
	req := agentcore.SandboxExec{
		Argv:        []string{"/bin/sh", "-c", command},
		Constraints: limits,
	}
	// A persistent tool reuses one container per conversation so installs and
	// files survive between calls. The session id rides the context (set by the
	// runtime); absent it, the call degrades to an ephemeral run.
	if t.persistent {
		req.Session = agentcore.SandboxSessionFrom(ctx)
	}
	// Share the agent workspace with the file tools: bind-mount it read-write and
	// run from it, so write_file → run_shell (and back) is coherent. The bind
	// mount stays writable even with the read-only root; isolation (no host env,
	// no network) is unchanged.
	if t.workspace != nil {
		req.Mounts = []agentcore.SandboxMount{{Source: t.workspace.Root(), Target: shellWorkdir}}
		req.Workdir = shellWorkdir
	}
	return req
}

// background launches req as a run-owned background job. With a workspace the
// output streams to a log file from the first byte, so the model can read it
// while the command runs.
func (t *ShellTool) background(ctx context.Context, req agentcore.SandboxExec, command string) (string, error) {
	var log *shellLog
	if t.workspace != nil {
		log = &shellLog{ws: t.workspace}
		if err := log.open(); err != nil {
			return "", fmt.Errorf("%s: background log: %w", t.name, err)
		}
	}
	label := command
	if len(label) > 80 {
		label = clampUTF8(label, 77) + "..."
	}
	receipt, err := agentcore.LaunchBackground(ctx, t.name, label, func(jobCtx context.Context) (string, error) {
		return t.execute(jobCtx, req, nil, log)
	})
	if err != nil {
		log.remove()
		return "", fmt.Errorf("%s: run_in_background: %w", t.name, err)
	}
	out := "started in background: " + string(receipt)
	if log != nil {
		out += "\nlive output: " + log.rel + " (read it with read_file while the command runs)"
	}
	return out + "\nIts result is reported when it finishes; use the background job tools to check on or cancel it.", nil
}

func (t *ShellTool) execute(ctx context.Context, req agentcore.SandboxExec, emit func(string), log *shellLog) (string, error) {
	out, err := runBounded(ctx, t.sb, req, emit, log)
	if err != nil {
		return "", fmt.Errorf("%s: %w", t.name, err)
	}
	return out, nil
}

// runBounded runs req on sb, capturing bounded output (and the full output in
// log, when there is one), and renders the result for the model.
func runBounded(ctx context.Context, sb agentcore.Sandbox, req agentcore.SandboxExec, emit func(string), log *shellLog) (string, error) {
	out := newShellOutput(log)
	stop := out.report(emit)
	res, err := captureOutput(ctx, sb, req, out)
	stop()
	out.finish()
	if err != nil {
		return "", err
	}
	return out.format(res), nil
}

// captureOutput runs req and feeds its output into out. A backend that can stream
// (agentcore.ProcessSandbox) is read incrementally, so memory stays bounded by
// the head/tail windows however much the command prints; any other backend's
// buffered result is fed through the same windows afterwards.
func captureOutput(ctx context.Context, sb agentcore.Sandbox, req agentcore.SandboxExec, out *shellOutput) (agentcore.SandboxResult, error) {
	ps, ok := sb.(agentcore.ProcessSandbox)
	if !ok {
		res, err := sb.Exec(ctx, req)
		_, _ = out.stream(0).Write([]byte(res.Stdout))
		_, _ = out.stream(1).Write([]byte(res.Stderr))
		return res, err
	}
	proc, err := ps.Start(ctx, req)
	if err != nil {
		return agentcore.SandboxResult{}, err
	}
	_ = proc.Stdin().Close()
	copied := make(chan struct{})
	go func() {
		var wg sync.WaitGroup
		wg.Add(2)
		go func() { defer wg.Done(); _, _ = io.Copy(out.stream(0), proc.Stdout()) }()
		go func() { defer wg.Done(); _, _ = io.Copy(out.stream(1), proc.Stderr()) }()
		wg.Wait()
		close(copied)
	}()
	// The backend enforces the timeout and kills on cancellation; the watchdog
	// only guarantees this call returns even if an escaped descendant keeps the
	// pipes open past that kill.
	var deadline <-chan time.Time
	if s := req.Constraints.TimeoutSeconds; s > 0 {
		timer := time.NewTimer(time.Duration(s*float64(time.Second)) + shellPipeGrace)
		defer timer.Stop()
		deadline = timer.C
	}
	select {
	case <-copied:
	case <-ctx.Done():
	case <-deadline:
	}
	select {
	case <-copied:
	default:
		_ = proc.Kill()
		select {
		case <-copied:
		case <-time.After(shellPipeGrace):
			_ = proc.Stdout().Close()
			_ = proc.Stderr().Close()
			<-copied
		}
	}
	res, err := proc.Wait()
	if err == nil {
		err = ctx.Err() // cancelled by the caller: an error, not a result
	}
	return res, err
}
