package sandbox

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

// End-to-end test of the browser_use tool driven through the real agent loop,
// the way a production run reaches it. Same two-provider strategy as the
// computer_use suite:
//
//   - FAUX provider: deterministic proof that the loop routes a browser_use call
//     into a persistent Chrome session, the agent-browser CLI actually controls a
//     real browser (opens a page, snapshots its accessibility tree), state
//     persists across calls (open then snapshot reuse one container), and — the
//     zombie guard — tearing the session down removes the container so no Chrome
//     survives. Needs Docker + the browser image; no AI credentials.
//
//   - REAL provider: the one thing a script can't prove — that the agent, given a
//     plain task and the browser_use tool, decides on its own to drive the
//     browser and reports what it saw. Gated behind the operator OpenAI-compatible
//     endpoint env vars; skipped when unset.

// browserImageName mirrors the production knob AGENTRAY_SANDBOX_BROWSER_IMAGE and
// falls back to the name Dockerfile.browser builds.
func browserImageName() string {
	if v := strings.TrimSpace(os.Getenv("AGENTRAY_SANDBOX_BROWSER_IMAGE")); v != "" {
		return v
	}
	return "agentray-browser:latest"
}

// newBrowserSandbox returns a DockerSandbox and skips (not fails) when Docker is
// down or the browser image has not been built — the image is opt-in/heavy, so a
// developer without it gets a clean skip.
func newBrowserSandbox(t *testing.T) *DockerSandbox {
	t.Helper()
	sb := NewDockerSandbox()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if !sb.Available(ctx) {
		t.Skip("docker not available — skipping browser_use integration test")
	}
	if exec.CommandContext(ctx, "docker", "image", "inspect", browserImageName()).Run() != nil {
		t.Skipf("browser image %q not present — build it with "+
			"`docker build -f Dockerfile.browser -t %s .` (or set "+
			"AGENTRAY_SANDBOX_BROWSER_IMAGE) to run this test", browserImageName(), browserImageName())
	}
	return sb
}

// browserUseAgent wires a real agent loop around the browser_use tool over the
// given sandbox + workspace, granting the tool and pinning the browser image.
func browserUseAgent(t *testing.T, provider agentcore.LLMProvider, model string, sb agentcore.Sandbox, ws *Workspace, def agentcore.AgentDefinition) *agentcore.Agent {
	t.Helper()
	agent, err := agentcore.New(agentcore.Config{
		Provider:   provider,
		Model:      model,
		Tools:      agentcore.NewToolSet(NewBrowserTool(sb, ws, BrowserUseLimits(), browserImageName())),
		Policy:     agentcore.NewAllowList(ToolBrowserUse),
		Definition: def,
	})
	if err != nil {
		t.Fatalf("agentcore.New: %v", err)
	}
	return agent
}

// browserSessionContainerGone reports whether the session container backing the
// browser session id has been removed (the host-level no-zombie guarantee: when
// the container is gone, the Chrome it held is gone too).
func browserSessionContainerGone(t *testing.T, sb *DockerSandbox, sessionID string) bool {
	t.Helper()
	name := "agentray-ses-" + sessionNameRe.ReplaceAllString(sessionID, "-")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	// inspect returns non-zero (error) once the container no longer exists.
	return exec.CommandContext(ctx, "docker", "inspect", "-f", "{{.State.Running}}", name).Run() != nil
}

// --- faux provider: real browser control + persistence + no zombie -----------

// TestBrowserUseAgent_ControlsBrowser_Faux is the headline mechanism proof with
// the model scripted. Across two browser_use calls in one run: call 1 opens a
// page (launching Chrome in the persistent session), call 2 snapshots its
// accessibility tree — proving the agent-browser CLI genuinely drives a browser
// and that the session persisted the browser between calls. Then it tears the
// session down and asserts the container (and its Chrome) is gone.
func TestBrowserUseAgent_ControlsBrowser_Faux(t *testing.T) {
	sb := newBrowserSandbox(t)
	ws := newWorkspaceDir(t)

	faux := agentcore.NewFauxProvider(
		// Call 1: open a page — starts Chrome inside the persistent session.
		agentcore.AssistantToolCall("c1", ToolBrowserUse,
			`{"command":"agent-browser open https://example.com"}`),
		// Call 2: snapshot the page the FIRST call left open (proves persistence
		// across docker-exec calls into the same session container).
		agentcore.AssistantToolCall("c2", ToolBrowserUse,
			`{"command":"agent-browser snapshot"}`),
		agentcore.AssistantText("Browsed example.com."),
	)
	agent := browserUseAgent(t, faux, "faux-model", sb, ws, agentcore.AgentDefinition{})

	sessionID := "conv-faux-browser" + browserSessionSuffix
	ctx := sessionCtx(t, context.Background(), sb, sessionID)
	// Browsing can take a while on a cold container (Chrome launch); give it room.
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()

	res, err := agent.Prompt(ctx, "open example.com and snapshot it")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if allowed, blocked := toolCallsFor(res, ToolBrowserUse); allowed != 2 || blocked != 0 {
		t.Fatalf("expected 2 allowed browser_use calls, got allowed=%d blocked=%d; traces=%+v", allowed, blocked, res.Tools)
	}

	// The snapshot must contain content only a real rendered page yields — the
	// "Example Domain" heading example.com is known for.
	out := strings.Join(toolResults(res), "\n")
	if !strings.Contains(out, "Example Domain") {
		t.Fatalf("browser snapshot did not contain the page heading (browser not really driven?):\n%s", out)
	}

	// Zombie guard: tearing the session down removes the container, so no Chrome
	// process survives at the host. CloseSession is what the production Runner
	// calls when a conversation ends.
	if ss, ok := agentcore.Sandbox(sb).(agentcore.SessionSandbox); ok {
		if err := ss.CloseSession(sessionID); err != nil {
			t.Fatalf("CloseSession: %v", err)
		}
	}
	if !browserSessionContainerGone(t, sb, sessionID) {
		t.Fatal("browser session container still running after CloseSession — zombie browser")
	}
}

// --- real provider: genuine autonomous browsing ------------------------------

// TestBrowserUseAgent_RealProvider_DrivesBrowser gives the agent a plain task and
// the browser_use tool, then asserts it decided on its own to drive the browser
// and reported what the page said. Gated behind the operator OpenAI-compatible
// endpoint (AGENTRAY_TEST_OPENAI_BASE_URL / _API_KEY / _MODEL) and the browser
// image; any missing prerequisite yields a skip.
func TestBrowserUseAgent_RealProvider_DrivesBrowser(t *testing.T) {
	baseURL := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_BASE_URL"))
	apiKey := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_API_KEY"))
	model := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_MODEL"))
	if baseURL == "" || apiKey == "" || model == "" {
		t.Skip("set AGENTRAY_TEST_OPENAI_BASE_URL, AGENTRAY_TEST_OPENAI_API_KEY and " +
			"AGENTRAY_TEST_OPENAI_MODEL to run the real-provider browser_use test")
	}
	sb := newBrowserSandbox(t)
	ws := newWorkspaceDir(t)

	provider := ai.NewOpenAIProvider(apiKey, baseURL, ai.DefaultCompat())
	agent := browserUseAgent(t, provider, model, sb, ws, agentcore.AgentDefinition{
		Agents: "You can drive a real web browser with the browser_use tool, which runs " +
			"`agent-browser` CLI commands in a Linux sandbox. Use a stable session, e.g. " +
			"`agent-browser open <url>` then `agent-browser snapshot` to read the page's " +
			"accessibility tree. Actually use the tool to inspect pages — do not guess.",
	})

	sessionID := "conv-real-browser" + browserSessionSuffix
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	ctx = sessionCtx(t, ctx, sb, sessionID)

	res, err := agent.Prompt(ctx,
		"Open https://example.com in the browser and tell me the exact text of the page's main heading.")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	t.Logf("model final answer: %q", res.Final)

	if allowed, _ := toolCallsFor(res, ToolBrowserUse); allowed < 1 {
		t.Fatalf("the agent never called browser_use (did not act autonomously); traces=%+v", res.Tools)
	}
	if !strings.Contains(res.Final, "Example Domain") {
		t.Fatalf("the agent did not report the page heading from a real browse: %q", res.Final)
	}

	if ss, ok := agentcore.Sandbox(sb).(agentcore.SessionSandbox); ok {
		_ = ss.CloseSession(sessionID)
	}
	if !browserSessionContainerGone(t, sb, sessionID) {
		t.Fatal("browser session container still running after CloseSession — zombie browser")
	}
}
