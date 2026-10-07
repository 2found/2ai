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

// BrowserUseLimits is the envelope for the persistent browser sandbox: network
// on (to load pages and download the browser binary on first use), writable
// filesystem (Chrome's profile/cache + cloakbrowser's ~/.cloakbrowser binary),
// and generous memory/time so a page render finishes. Still hard-isolated by the
// backend (no host env, all caps dropped, no-new-privileges, resource caps).
func BrowserUseLimits() agentcore.SandboxLimits {
	return agentcore.SandboxLimits{
		Network:        true,
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
// BrowserUseLimits() for the network+writable browser profile.
func NewBrowserTool(sb agentcore.Sandbox, workspace *Workspace, limits agentcore.SandboxLimits, image string) *BrowserTool {
	hosted := sb == nil
	if hosted {
		sb = NewHostSandbox()
	}
	return &BrowserTool{sb: sb, workspace: workspace, limits: limits, image: image, hosted: hosted}
}

func (t *BrowserTool) Name() string { return ToolBrowserUse }

func (t *BrowserTool) Schema() agentcore.ToolSchema {
	// The artifact path differs by substrate: inside a sandbox the workspace is
	// bind-mounted at browserWorkdir, on the host it is simply the working
	// directory. Telling the model to write to /workspace on the host substrate
	// would send every screenshot to a path it cannot create.
	desc := "Drive a real web browser via the `agent-browser` CLI inside a " +
		"persistent, network-enabled sandbox. State persists across calls in the " +
		"same conversation, so browse step by step: `agent-browser --session s open <url>`, " +
		"`agent-browser --session s snapshot -i` (interactive accessibility tree with refs), " +
		"`agent-browser --session s click @e1`, `agent-browser --session s type @e2 \"text\"`, " +
		"`agent-browser --session s screenshot " + browserWorkdir + "/page.png`. Use a stable --session " +
		"name. Artifacts written under " + browserWorkdir + " are saved. Returns exit code, stdout, stderr."
	if t.hosted {
		desc = "Drive a real web browser via the `agent-browser` CLI on this machine. " +
			"State persists across calls, so browse step by step: " +
			"`agent-browser --session s open <url>`, " +
			"`agent-browser --session s snapshot -i` (interactive accessibility tree with refs), " +
			"`agent-browser --session s click @e1`, `agent-browser --session s type @e2 \"text\"`, " +
			"`agent-browser --session s screenshot page.png`. Use a stable --session name. " +
			"Commands run in the agent workspace directory; artifacts written there are saved " +
			"(use relative paths). Returns exit code, stdout, stderr."
	}
	return agentcore.ToolSchema{
		Name:        ToolBrowserUse,
		Description: desc,
		Strict:      agentcore.ToolStrictEnabled,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{
					"type":        "string",
					"description": "An `agent-browser` CLI command to run inside the browser sandbox.",
				},
			},
			"required": []string{"command"},
		},
	}
}

func (t *BrowserTool) Run(ctx context.Context, args string) (string, error) {
	if t.workspace == nil {
		return "", fmt.Errorf("browser_use: no workspace configured")
	}
	var in struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("browser_use: invalid arguments: %w", err)
	}
	if strings.TrimSpace(in.Command) == "" {
		return "", fmt.Errorf("browser_use: empty command")
	}

	exec := agentcore.SandboxExec{
		Argv:        []string{"/bin/sh", "-c", in.Command},
		Image:       t.image,
		Constraints: t.limits,
		Mounts: []agentcore.SandboxMount{{
			Source:   t.workspace.Root(),
			Target:   browserWorkdir,
			ReadOnly: false,
		}},
		Workdir: browserWorkdir,
	}
	// Persist the browser across calls in one conversation by reusing a session
	// container — keyed to the conversation id but namespaced apart from the
	// computer_use session (different image). Absent a conversation id the call
	// degrades to an ephemeral container (single command still works; the daemon
	// just won't outlive the call).
	if sid := agentcore.SandboxSessionFrom(ctx); sid != "" {
		exec.Session = sid + browserSessionSuffix
	}

	res, err := t.sb.Exec(ctx, exec)
	if err != nil {
		return "", fmt.Errorf("browser_use: %w", err)
	}
	return formatResult(res), nil
}
