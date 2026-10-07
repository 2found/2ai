package sandbox

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

// This file is the end-to-end test of the computer_use tool driven through the
// real agent loop (agentcore.Agent), the way a production run reaches it.
//
// Two provider strategies, matching what each behaviour actually needs:
//
//   - FAUX provider (no real AI, no tokens): used for everything that is
//     deterministic — the loop routes a computer_use tool call into the sandbox,
//     state persists across calls, an artifact lands in the host workspace, the
//     document toolchain works, and the policy gate blocks the tool when it is
//     not granted. The model's decisions are scripted, so these are reproducible
//     and run in CI; only Docker (and, for the document case, the computer-use
//     image) is required.
//
//   - REAL provider (needs genuine model reasoning): used only for the one thing
//     a script cannot prove — that the agent, given a natural-language task and
//     the computer_use tool, decides on its own to install/run code and produce
//     the requested document. Gated behind env vars (the OpenAI-compatible URL +
//     key the operator supplies); skipped when unset.

// --- helpers --------------------------------------------------------------

// cuImageName is the computer-use toolchain image (python + document stack).
// It mirrors the production config knob AGENTRAY_SANDBOX_COMPUTER_USE_IMAGE and
// falls back to the name the Dockerfile.computeruse build tags.
func cuImageName() string {
	if v := strings.TrimSpace(os.Getenv("AGENTRAY_SANDBOX_COMPUTER_USE_IMAGE")); v != "" {
		return v
	}
	return "agentray-computeruse:latest"
}

// newComputerUseSandbox returns a DockerSandbox whose persistent sessions use
// the computer-use toolchain image. It skips (not fails) when Docker is down or
// the image has not been built — the heavy image is opt-in, so a developer who
// hasn't built it still gets a clean skip rather than a red suite.
func newComputerUseSandbox(t *testing.T) *DockerSandbox {
	t.Helper()
	sb := NewDockerSandbox(WithComputerUseImage(cuImageName()))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if !sb.Available(ctx) {
		t.Skip("docker not available — skipping computer_use integration test")
	}
	if exec.CommandContext(ctx, "docker", "image", "inspect", cuImageName()).Run() != nil {
		t.Skipf("computer-use image %q not present — build it with "+
			"`docker build -f Dockerfile.computeruse -t %s .` (or set "+
			"AGENTRAY_SANDBOX_COMPUTER_USE_IMAGE) to run this test", cuImageName(), cuImageName())
	}
	return sb
}

// newWorkspaceDir builds a Workspace rooted in a throwaway temp dir, so files
// the agent writes are inspectable on the host and cleaned up automatically.
func newWorkspaceDir(t *testing.T) *Workspace {
	t.Helper()
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	// Container root has all capabilities dropped, including DAC override.
	// On Linux it cannot traverse/write a 0700 host-user fixture directory.
	// This disposable, secret-free artifact fixture explicitly grants access;
	// production workspace permissions remain host-owned.
	if err := os.Chmod(ws.Root(), 0o777); err != nil {
		t.Fatalf("grant container access to artifact fixture: %v", err)
	}
	return ws
}

// computerUseAgent wires a real agent loop around the computer_use tool over the
// given sandbox + workspace, granting the tool in the policy. provider is the
// only thing that varies between the faux and real tests.
func computerUseAgent(t *testing.T, provider *ai.FallbackProvider, model string, sb agentcore.Sandbox, ws *Workspace) *agentcore.Agent {
	t.Helper()
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: provider,
		Model:          model,
		Tools:          agentcore.NewToolSet(NewComputerUseTool(sb, ws)),
		Policy:         agentcore.NewAllowList(ToolComputerUse),
	})
	if err != nil {
		t.Fatalf("agentcore.New: %v", err)
	}
	return agent
}

// sessionCtx tags ctx with a persistent computer_use session and registers its
// teardown, exactly as the production Runner does around a run. Without this the
// tool degrades to ephemeral per-call containers and cross-call state is lost.
func sessionCtx(t *testing.T, ctx context.Context, sb agentcore.Sandbox, id string) context.Context {
	t.Helper()
	if ss, ok := sb.(agentcore.SessionSandbox); ok {
		t.Cleanup(func() { _ = ss.CloseSession(id) })
	}
	return agentcore.WithSandboxSession(ctx, id)
}

// toolCallsFor counts the allowed invocations of a named tool in a run result.
func toolCallsFor(res agentcore.RunResult, name string) (allowed, blocked int) {
	for _, tr := range res.Tools {
		if tr.Tool != name {
			continue
		}
		if tr.Allowed {
			allowed++
		} else {
			blocked++
		}
	}
	return allowed, blocked
}

// --- faux provider: deterministic loop / persistence / artifact -----------

// TestComputerUseAgent_PersistsStateAndWritesArtifact_Faux is the headline
// mechanism proof, with the model scripted so no real AI is needed. Across two
// computer_use calls in one run: call 1 writes state to the container's /tmp,
// call 2 reads that state back (proving the persistent session reuses one
// container) and writes a report file into the workspace (proving the artifact
// lands on the host filesystem). Uses the default sandbox image — no toolchain
// or network required.
func TestComputerUseAgent_PersistsStateAndWritesArtifact_Faux(t *testing.T) {
	sb := newTestSandbox(t)
	ws := newWorkspaceDir(t)

	faux := sandboxNativeFixture(
		// Turn 1: stash state inside the session container (not the workspace).
		agentcore.AssistantToolCall("c1", ToolComputerUse, `{"command":"echo step1-state > /tmp/agent_state"}`),
		// Turn 2: read the prior call's state and persist a real artifact to the
		// workspace (the working directory is the mounted host temp dir).
		agentcore.AssistantToolCall("c2", ToolComputerUse, `{"command":"cat /tmp/agent_state > report.txt && echo wrote report.txt"}`),
		agentcore.AssistantText("Saved the report."),
	)
	agent := computerUseAgent(t, faux, "faux-model", sb, ws)

	ctx := sessionCtx(t, context.Background(), sb, "conv-faux-persist")
	res, err := agent.Prompt(ctx, "stash some state then write a report file")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	if res.Final != "Saved the report." {
		t.Fatalf("unexpected final answer: %q", res.Final)
	}
	if allowed, blocked := toolCallsFor(res, ToolComputerUse); allowed != 2 || blocked != 0 {
		t.Fatalf("expected 2 allowed computer_use calls, got allowed=%d blocked=%d", allowed, blocked)
	}

	// The artifact must exist on the host workspace and carry the state written by
	// the FIRST call — proving both host persistence and cross-call container reuse.
	got, err := os.ReadFile(filepath.Join(ws.Root(), "report.txt"))
	if err != nil {
		t.Fatalf("artifact not written to workspace: %v", err)
	}
	if !strings.Contains(string(got), "step1-state") {
		t.Fatalf("artifact missing state from the first call (session did not persist): %q", got)
	}
}

// recordingSandbox is a stub Sandbox that fails if Exec is ever reached. It lets
// the permission test prove the gate blocks computer_use *before* any container
// runs, with no Docker dependency.
type recordingSandbox struct{ execCalled bool }

func (r *recordingSandbox) Exec(context.Context, agentcore.SandboxExec) (agentcore.SandboxResult, error) {
	r.execCalled = true
	return agentcore.SandboxResult{}, nil
}

// TestComputerUseAgent_BlockedWithoutGrant_Faux verifies the default-deny
// posture for the higher-privilege tool: when computer_use is NOT in the policy
// allow-list, a scripted call to it is blocked, never executes (the sandbox is
// never touched), and the block reason is fed back to the model. No real AI and
// no Docker are needed — the gate sits in front of execution.
func TestComputerUseAgent_BlockedWithoutGrant_Faux(t *testing.T) {
	stub := &recordingSandbox{}
	ws := newWorkspaceDir(t)

	faux := sandboxNativeFixture(
		agentcore.AssistantToolCall("c1", ToolComputerUse, `{"command":"echo i should never run"}`),
		agentcore.AssistantText("understood, I cannot use that tool"),
	)
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: faux,
		Model:          "faux-model",
		Tools:          agentcore.NewToolSet(NewComputerUseTool(stub, ws)),
		Policy:         agentcore.NewAllowList(), // computer_use deliberately NOT granted
	})
	if err != nil {
		t.Fatalf("agentcore.New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "run a shell command for me")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if stub.execCalled {
		t.Fatal("blocked computer_use must never reach the sandbox")
	}
	if allowed, blocked := toolCallsFor(res, ToolComputerUse); allowed != 0 || blocked != 1 {
		t.Fatalf("expected 1 blocked computer_use call, got allowed=%d blocked=%d", allowed, blocked)
	}
	var sawBlock bool
	for _, m := range res.Messages {
		// The native engine does not advertise ungranted tools; a scripted
		// attempt receives an unavailable-tool result before host dispatch.
		if m.Role == agentcore.RoleTool && strings.Contains(m.Content, "not found") {
			sawBlock = true
		}
	}
	if !sawBlock {
		t.Fatalf("block reason was not returned to the model: %+v", res.Messages)
	}
}

// TestComputerUseAgent_InstallAndGenerateDocument_Faux proves the full
// Claude-Code-level capability deterministically: across two scripted calls the
// agent installs a package from the network (call 1) and then runs Python that
// imports both the freshly-installed package and the preinstalled document stack
// to produce a real .xlsx in the workspace (call 2). Requires the computer-use
// image; skipped otherwise. No real AI — the capability, not the reasoning, is
// under test here.
func TestComputerUseAgent_InstallAndGenerateDocument_Faux(t *testing.T) {
	sb := newComputerUseSandbox(t)
	ws := newWorkspaceDir(t)

	faux := sandboxNativeFixture(
		// Call 1: install a tiny pure-python package from the network, proving the
		// session can `pip install` and that the install persists for the next call.
		agentcore.AssistantToolCall("c1", ToolComputerUse,
			`{"command":"pip install --quiet --no-input pyfiglet && python3 -c 'import pyfiglet; print(\"installed\", pyfiglet.__version__)'"}`),
		// Call 2: use the installed package AND the preinstalled openpyxl to write a
		// real spreadsheet into the workspace.
		agentcore.AssistantToolCall("c2", ToolComputerUse,
			`{"command":"python3 -c \"import pyfiglet, openpyxl; from openpyxl import Workbook; wb=Workbook(); ws=wb.active; ws.append(['Month','Revenue']); ws.append(['Jan',100]); ws.append(['Feb',120]); wb.save('sales.xlsx')\" && ls -l sales.xlsx"}`),
		agentcore.AssistantText("Created sales.xlsx with sample data."),
	)
	agent := computerUseAgent(t, faux, "faux-model", sb, ws)

	ctx := sessionCtx(t, context.Background(), sb, "conv-faux-doc")
	res, err := agent.Prompt(ctx, "install a tool then generate a spreadsheet")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	if allowed, _ := toolCallsFor(res, ToolComputerUse); allowed != 2 {
		t.Fatalf("expected 2 allowed computer_use calls, got %d; traces=%+v", allowed, res.Tools)
	}
	// The install must have succeeded (call 1's result is fed back to the model).
	if !strings.Contains(strings.Join(toolResults(res), "\n"), "installed") {
		t.Fatalf("pip install did not report success in tool output:\n%s", strings.Join(toolResults(res), "\n"))
	}
	assertValidXLSX(t, filepath.Join(ws.Root(), "sales.xlsx"))
}

// --- real provider: genuine autonomous reasoning --------------------------

// TestComputerUseAgent_RealProvider_GeneratesDocument is the only test that
// needs a real model: it gives the agent a plain natural-language task and the
// computer_use tool, then asserts the agent decided on its own to run code and
// produced the requested document. Gated behind the operator-supplied
// OpenAI-compatible endpoint:
//
//	AGENTRAY_TEST_OPENAI_BASE_URL  (e.g. https://api.openai.com/v1)
//	AGENTRAY_TEST_OPENAI_API_KEY
//	AGENTRAY_TEST_OPENAI_MODEL     (e.g. gpt-4o-mini)
//
// and the computer-use image. Any missing prerequisite yields a skip, so the
// suite stays green without credentials.
func TestComputerUseAgent_RealProvider_GeneratesDocument(t *testing.T) {
	baseURL := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_BASE_URL"))
	apiKey := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_API_KEY"))
	model := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_MODEL"))
	if baseURL == "" || apiKey == "" || model == "" {
		t.Skip("set AGENTRAY_TEST_OPENAI_BASE_URL, AGENTRAY_TEST_OPENAI_API_KEY and " +
			"AGENTRAY_TEST_OPENAI_MODEL to run the real-provider computer_use test")
	}
	sb := newComputerUseSandbox(t)
	ws := newWorkspaceDir(t)

	rawModel, _ := json.Marshal(map[string]any{"id": model, "api": "openai-completions", "provider": "openai", "baseUrl": baseURL, "contextWindow": 128000})
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: rawModel, Stream: func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		controls := map[string]any{}
		for k, v := range options {
			controls[k] = v
		}
		controls["apiKey"] = apiKey
		return (ai.NativeProvider{}).Stream(ctx, model, view, controls)
	}}}}
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: provider,
		Model:          model,
		Tools:          agentcore.NewToolSet(NewComputerUseTool(sb, ws)),
		Policy:         agentcore.NewAllowList(ToolComputerUse),
		Definition: agentcore.AgentDefinition{
			Agents: "You can run shell commands and code via the computer_use tool in a " +
				"Linux sandbox. The sandbox already has python3 with openpyxl, python-docx, " +
				"and pypdf installed. The working directory is the workspace; files you " +
				"create there are saved. When asked to produce a file, actually create it " +
				"with the tool — do not just describe how.",
		},
	})
	if err != nil {
		t.Fatalf("agentcore.New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	ctx = sessionCtx(t, ctx, sb, "conv-real-doc")

	res, err := agent.Prompt(ctx,
		"Create an Excel file named sales.xlsx in the current working directory. "+
			"It should have a header row with columns Month and Revenue, followed by "+
			"three rows of sample monthly data. After creating it, reply with just the filename.")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	t.Logf("model final answer: %q", res.Final)

	if allowed, _ := toolCallsFor(res, ToolComputerUse); allowed < 1 {
		t.Fatalf("the agent never called computer_use (did not act autonomously); traces=%+v", res.Tools)
	}
	assertValidXLSX(t, filepath.Join(ws.Root(), "sales.xlsx"))
}

// --- shared assertions ----------------------------------------------------

// toolResults collects the tool-result message contents from a run, so a test
// can assert on what a tool reported back to the model.
func toolResults(res agentcore.RunResult) []string {
	var out []string
	for _, m := range res.Messages {
		if m.Role == agentcore.RoleTool {
			out = append(out, m.Content)
		}
	}
	return out
}

// assertValidXLSX checks the file exists and looks like a real .xlsx — an Office
// Open XML file is a ZIP, so it must begin with the "PK" magic bytes and be
// non-trivial in size. This catches an agent that wrote a placeholder or an
// error message instead of a genuine spreadsheet.
func assertValidXLSX(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("expected %s to exist: %v", filepath.Base(path), err)
	}
	if len(data) < 256 {
		t.Fatalf("%s is implausibly small (%d bytes) — not a real spreadsheet", filepath.Base(path), len(data))
	}
	if len(data) < 2 || data[0] != 'P' || data[1] != 'K' {
		t.Fatalf("%s is not a valid xlsx (missing ZIP/PK magic header)", filepath.Base(path))
	}
}
