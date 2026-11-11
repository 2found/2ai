package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/2found/2ai/agentcore"
)

// ToolBrowserUse is the higher-privilege, Claude-Code-level browser surface. It
// runs `agent-browser` CLI commands inside a persistent, network-enabled Chrome
// sandbox so the agent can drive a real browser across calls — open a page,
// snapshot its (LLM-optimized) accessibility tree, click by ref, type, and
// screenshot — with state surviving between invocations in one conversation.
const ToolBrowserUse = "browser_use"

const browserWorkdir = "/workspace"

// browserSessionSuffix namespaces the browser session container apart from the
// computer_use session of the same conversation: the two need different images
// (Chrome vs the doc toolchain), so they must be distinct containers even though
// they share a conversation id.
const browserSessionSuffix = "::browser"

// BrowserUseLimits is the envelope for the persistent browser sandbox:
// writable filesystem (Chrome's profile/cache + cloakbrowser's ~/.cloakbrowser
// binary), and generous memory/time so a page render finishes. Egress is
// granted only to networkAllow (pass AnyHost to browse the open web); empty
// runs the browser offline. Still hard-isolated by the backend (no host env,
// all caps dropped, no-new-privileges, resource caps).
func BrowserUseLimits(networkAllow ...string) agentcore.SandboxLimits {
	return agentcore.SandboxLimits{
		Network:        len(networkAllow) > 0,
		NetworkAllow:   networkAllow,
		WritableFS:     true,
		MemoryMB:       2048,
		CPUs:           2,
		PidsLimit:      512,
		TimeoutSeconds: 180,
	}
}

// BrowserTool drives a browser via the agent-browser CLI on the substrate it
// was built with. With an injected Sandbox it reuses a persistent,
// browser-scoped session container so the agent-browser daemon (and the page it
// controls) survive across calls — the property that makes multi-step browsing
// possible. The daemon self-reaps on idle (AGENT_BROWSER_IDLE_TIMEOUT_MS, set
// in the image) so no zombie browser pins resources between conversations.
// With no sandbox the same CLI is invoked on the host machine, where the daemon
// is already a long-lived host process and persists without a container.
type BrowserTool struct {
	sb        agentcore.Sandbox
	workspace *Workspace
	limits    agentcore.SandboxLimits
	// image is the Chrome-capable sandbox image (agent-browser + Chrome/cloak).
	// Empty falls back to the backend default — which generally lacks a browser,
	// so a deployment that grants browser_use should configure it. Ignored on the
	// host substrate, which has no images.
	image string
	// hosted records that no sandbox was injected, so the tool neither asks for a
	// session container nor claims isolation it is not providing.
	hosted bool
}

// NewBrowserTool builds the browser_use tool over sb, with the agent workspace
// mounted (so screenshots/exports persist on the host) and a Chrome-capable
// image. sb is optional: nil drives an `agent-browser` installed on the host
// machine instead, where the daemon persists across calls without a session
// container and the image is irrelevant. limits is the isolation envelope; pass
// BrowserUseLimits(hosts...) — or BrowserUseLimits(AnyHost) for the open web.
func NewBrowserTool(sb agentcore.Sandbox, workspace *Workspace, limits agentcore.SandboxLimits, image string) *BrowserTool {
	hosted := sb == nil
	if hosted {
		sb = NewHostSandbox()
	}
	return &BrowserTool{sb: sb, workspace: workspace, limits: limits, image: image, hosted: hosted}
}

func (t *BrowserTool) Name() string { return ToolBrowserUse }

// ExternalContent marks page text, snapshots and console output as untrusted
// (agentcore.ExternalContentTool).
func (t *BrowserTool) ExternalContent() bool { return true }

// NetworkHosts reports the sandbox's egress allowlist (agentcore.NetworkTool):
// a page can load or navigate anywhere its egress permits, so the allowlist,
// not the URL in one call, is what a credential would reach.
func (t *BrowserTool) NetworkHosts(string) ([]string, error) {
	return egressHosts(ToolBrowserUse, t.hosted, t.limits)
}

func (t *BrowserTool) Schema() agentcore.ToolSchema {
	// The artifact path differs by substrate: inside a sandbox the workspace is
	// bind-mounted at browserWorkdir, on the host it is simply the working
	// directory. Telling the model to write to /workspace on the host substrate
	// would send every screenshot to a path it cannot create.
	where, shot := "inside a persistent sandbox", browserWorkdir+"/page.png"
	if t.hosted {
		where, shot = "on this machine", "page.png"
	}
	desc := "Drive a real web browser via the `agent-browser` CLI " + where + ". Pass the CLI's arguments " +
		"as an array (no shell; the program name is implied). State persists across calls, so browse step by step: " +
		`["--session","s","open","<url>"], ["--session","s","snapshot","-i"] (interactive accessibility tree with refs), ` +
		`["--session","s","click","@e1"], ["--session","s","fill","@e2","text"], ["--session","s","screenshot","` + shot + `"]. ` +
		"Use a stable --session name. Commands run in the agent workspace; artifacts written there are saved. " +
		"Commands that install software, attach to other browsers, load extensions or launch arguments, or read " +
		"host files are refused. Returns exit code, stdout, stderr."
	return agentcore.ToolSchema{
		Name:        ToolBrowserUse,
		Description: desc,
		Strict:      agentcore.ToolStrictEnabled,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"args": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"minItems":    1,
					"description": "Arguments to `agent-browser`, one array element per argument.",
				},
			},
			"required": []string{"args"},
		},
	}
}

// browserCommands are the agent-browser commands browser_use may run: page
// navigation, interaction, inspection and capture. Everything else — install,
// upgrade, plugin, auth, connect, mcp, chat, dashboard, batch (which would run
// unvalidated nested commands), upload (reads arbitrary host files), clipboard
// (the host clipboard), stream (opens a port) — is refused.
var browserCommands = map[string]bool{
	"open": true, "read": true, "click": true, "dblclick": true, "type": true, "fill": true,
	"press": true, "keyboard": true, "hover": true, "focus": true, "check": true, "uncheck": true,
	"select": true, "drag": true, "download": true, "scroll": true, "scrollintoview": true,
	"wait": true, "screenshot": true, "pdf": true, "snapshot": true, "eval": true, "close": true,
	"back": true, "forward": true, "reload": true, "get": true, "is": true, "find": true,
	"mouse": true, "set": true, "cookies": true, "storage": true, "tab": true, "diff": true,
	"console": true, "errors": true, "highlight": true, "session": true, "skills": true,
}

// browserDeniedFlags change what the CLI launches or trusts: a custom browser
// binary, extensions, init scripts, raw Chrome arguments, another browser or
// profile to attach to, proxy/TLS trust, file:// access, remote providers, or
// where state is read from. Refused anywhere in the arguments, in both
// "--flag value" and "--flag=value" form.
var browserDeniedFlags = map[string]bool{
	"--executable-path": true, "--extension": true, "--init-script": true, "--enable": true,
	"--args": true, "--profile": true, "--restore": true, "--restore-save": true,
	"--restore-check-url": true, "--restore-check-text": true, "--restore-check-fn": true,
	"--session-name": true, "--state": true, "--auto-connect": true, "--cdp": true,
	"--proxy": true, "--proxy-bypass": true, "--ignore-https-errors": true, "--ca-cert": true,
	"--no-ca-cert": true, "--allow-file-access": true, "-p": true, "--provider": true,
	"--device": true, "--namespace": true, "--headed": true, "--screenshot-dir": true,
	"--credential-provider": true,
}

// validateBrowserArgs returns the command an argument list runs, or why it is
// refused. Before the command only --session <name> and --json may appear.
func validateBrowserArgs(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("args is empty")
	}
	command := ""
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if strings.ContainsRune(arg, 0) {
			return fmt.Errorf("argument %d contains a NUL byte", i)
		}
		flag, _, _ := strings.Cut(arg, "=")
		if browserDeniedFlags[flag] {
			return fmt.Errorf("option %s is not allowed", flag)
		}
		if command != "" {
			continue
		}
		switch {
		case arg == "--session":
			if i+1 >= len(args) || strings.HasPrefix(args[i+1], "-") {
				return fmt.Errorf("--session needs a name")
			}
			i++
		case strings.HasPrefix(arg, "--session="), arg == "--json":
		case strings.HasPrefix(arg, "-"):
			return fmt.Errorf("option %s must follow the command", arg)
		default:
			if !browserCommands[arg] {
				return fmt.Errorf("command %q is not allowed", arg)
			}
			command = arg
		}
	}
	if command == "" {
		return fmt.Errorf("no command given")
	}
	return nil
}

func (t *BrowserTool) Run(ctx context.Context, args string) (string, error) {
	if t.workspace == nil {
		return "", fmt.Errorf("browser_use: no workspace configured")
	}
	var in struct {
		Args []string `json:"args"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("browser_use: invalid arguments: %w", err)
	}
	if err := validateBrowserArgs(in.Args); err != nil {
		return "", fmt.Errorf("browser_use: %w", err)
	}
	req := agentcore.SandboxExec{
		Argv:        append([]string{"agent-browser"}, in.Args...),
		Image:       t.image,
		Constraints: t.limits,
		Mounts:      []agentcore.SandboxMount{{Source: t.workspace.Root(), Target: browserWorkdir}},
		Workdir:     browserWorkdir,
	}
	// Persist the browser across calls in one conversation by reusing a session
	// container — keyed to the conversation id but namespaced apart from the
	// computer_use session (different image). Absent a conversation id the call
	// degrades to an ephemeral container (single command still works; the daemon
	// just won't outlive the call).
	if sid := agentcore.SandboxSessionFrom(ctx); sid != "" {
		req.Session = sid + browserSessionSuffix
	}
	out, err := runBounded(ctx, t.sb, req, nil, &shellLog{ws: t.workspace, threshold: shellHeadBytes + shellTailBytes})
	if err != nil {
		return "", fmt.Errorf("browser_use: %w", err)
	}
	return out, nil
}
