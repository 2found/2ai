package integration

// Real-provider integration tests for the capabilities that genuinely require a
// model to reason — the ones a scripted faux provider cannot prove. They are the
// Claude-Code-level bar applied end to end: the model decides, on its own, to
// call a tool, fetch the web, follow steering, keep a plan across a compacting
// session, and pull a skill on demand.
//
// All are gated behind the operator-supplied OpenAI-compatible endpoint and skip
// when it is absent, so the suite stays green without credentials:
//
//	AGENTRAY_TEST_OPENAI_BASE_URL   e.g. http://localhost:20128/v1
//	AGENTRAY_TEST_OPENAI_API_KEY
//	AGENTRAY_TEST_OPENAI_MODEL      e.g. plus
//
// The deterministic mechanics of each capability (compaction replacing a span,
// the steering queue draining before a turn, the todo pin surviving compaction,
// progressive skill disclosure, the permission gate, trace records) are proven
// separately and reproducibly by the faux unit tests in this package
// (compaction_test, prompt_test, todo_test, loop_test). These tests confirm a real model actually exercises
// them.

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/todo"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/sandbox"
)

// realProvider builds the operator's OpenAI-compatible provider, or skips.
func realProvider(t *testing.T) (agentcore.LLMProvider, string) {
	t.Helper()
	base := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_BASE_URL"))
	key := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_API_KEY"))
	model := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_MODEL"))
	if base == "" || key == "" || model == "" {
		t.Skip("set AGENTRAY_TEST_OPENAI_BASE_URL, AGENTRAY_TEST_OPENAI_API_KEY, AGENTRAY_TEST_OPENAI_MODEL to run real-provider tests")
	}
	return ai.NewOpenAIProvider(key, base, ai.DefaultCompat()), model
}

// recordingProvider wraps a real provider and records every ChatRequest, so a
// test can assert on exactly what the model was shown each turn (the steering
// message arrived, the pinned plan was present after compaction, …) without
// depending on the model's wording.
type recordingProvider struct {
	inner agentcore.LLMProvider
	mu    sync.Mutex
	reqs  []agentcore.ChatRequest
}

func (r *recordingProvider) record(req agentcore.ChatRequest) {
	r.mu.Lock()
	r.reqs = append(r.reqs, req)
	r.mu.Unlock()
}
func (r *recordingProvider) requests() []agentcore.ChatRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]agentcore.ChatRequest(nil), r.reqs...)
}
func (r *recordingProvider) Name() string        { return r.inner.Name() }
func (r *recordingProvider) SupportsTools() bool { return r.inner.SupportsTools() }
func (r *recordingProvider) Chat(ctx context.Context, req agentcore.ChatRequest) (agentcore.ChatResponse, error) {
	r.record(req)
	return r.inner.Chat(ctx, req)
}
func (r *recordingProvider) Stream(ctx context.Context, req agentcore.ChatRequest) (<-chan agentcore.ChatDelta, error) {
	r.record(req)
	return r.inner.Stream(ctx, req)
}

// recordTool is a permitted no-op tool that records each invocation's args, used
// to keep a multi-turn loop alive and observe what the model did per turn.
type recordTool struct {
	name  string
	reply string
	mu    sync.Mutex
	args  []string
}

func (c *recordTool) Name() string { return c.name }
func (c *recordTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:        c.name,
		Description: "Record a step. Call once per turn while working.",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"item": map[string]any{"type": "string", "description": "what was done"}},
		},
	}
}
func (c *recordTool) Run(_ context.Context, args string) (string, error) {
	c.mu.Lock()
	c.args = append(c.args, args)
	c.mu.Unlock()
	if c.reply != "" {
		return c.reply, nil
	}
	return "recorded", nil
}
func (c *recordTool) calls() int { c.mu.Lock(); defer c.mu.Unlock(); return len(c.args) }

func toolWasCalled(res agentcore.RunResult, name string) bool {
	for _, tr := range res.Tools {
		if tr.Tool == name && tr.Allowed {
			return true
		}
	}
	return false
}

// maxTokensProvider raises the per-request output ceiling for tasks that emit a
// lot of text (a full HTML page), since the loop leaves max_tokens unset and the
// gateway's small default otherwise truncates the turn with stop_reason "length".
type maxTokensProvider struct {
	inner agentcore.LLMProvider
	max   int
}

func (m *maxTokensProvider) Name() string        { return m.inner.Name() }
func (m *maxTokensProvider) SupportsTools() bool { return m.inner.SupportsTools() }
func (m *maxTokensProvider) Chat(ctx context.Context, req agentcore.ChatRequest) (agentcore.ChatResponse, error) {
	if req.MaxTokens == 0 {
		req.MaxTokens = m.max
	}
	return m.inner.Chat(ctx, req)
}
func (m *maxTokensProvider) Stream(ctx context.Context, req agentcore.ChatRequest) (<-chan agentcore.ChatDelta, error) {
	if req.MaxTokens == 0 {
		req.MaxTokens = m.max
	}
	return m.inner.Stream(ctx, req)
}

// flakyWriter is a realistic file-writer tool whose first failFirst calls return
// a transient error, then succeed and capture the content. It exercises the
// tool-error resilience path with a real model: an erroring tool must not stop
// the agent — it should retry and ultimately deliver the work.
type flakyWriter struct {
	mu        sync.Mutex
	failFirst int
	attempts  int
	errors    int
	files     map[string]string
}

func (w *flakyWriter) Name() string { return "write_file" }
func (w *flakyWriter) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:        "write_file",
		Description: "Write a file to disk. May transiently fail; retry on error.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":    map[string]any{"type": "string", "description": "file path"},
				"content": map[string]any{"type": "string", "description": "full file contents"},
			},
			"required": []any{"path", "content"},
		},
	}
}
func (w *flakyWriter) Run(_ context.Context, args string) (string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.attempts++
	if w.attempts <= w.failFirst {
		w.errors++
		return "", errors.New("storage temporarily unavailable (E_BUSY); please retry")
	}
	var in struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", err
	}
	if w.files == nil {
		w.files = map[string]string{}
	}
	w.files[in.Path] = in.Content
	return "wrote " + in.Path + " (" + itoa(len(in.Content)) + " bytes)", nil
}
func (w *flakyWriter) best() (string, string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	var bp, bc string
	for p, c := range w.files {
		if len(c) > len(bc) {
			bp, bc = p, c
		}
	}
	return bp, bc
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// TestReal_CreativeLandingPage confirms end-to-end agent behavior on an open,
// creative task while the only tool it has keeps failing transiently. It is the
// real-model proof of this session's resilience work: a tool error must not stop
// the run — the model retries through the failures and still delivers the page.
//
// Assertions are deliberately model-agnostic: we don't grade taste, we confirm
// (1) the agent actually hit tool errors and kept going, (2) it ultimately wrote
// a substantial, self-contained HTML landing page for the named product. The
// generated page is written to AGENTRAY_TEST_ARTIFACT_DIR (when set) so a human
// can eyeball the creativity.
func TestReal_CreativeLandingPage(t *testing.T) {
	rawProvider, model := realProvider(t)
	// A full HTML page is large; give the turn enough output headroom so it isn't
	// truncated mid-write with stop_reason "length".
	provider := &maxTokensProvider{inner: rawProvider, max: 16000}

	// Fail the first two write attempts: transient errors the agent must ride out
	// without the circuit breaker (threshold 3) ever tripping the tool.
	writer := &flakyWriter{failFirst: 2}

	limits := agentcore.DefaultLimits()
	limits.MaxTurns = 14
	limits.MaxToolCalls = 30

	agent, err := agentcore.New(agentcore.Config{
		Provider: provider,
		Model:    model,
		Limits:   &limits,
		Tools:    agentcore.NewToolSet(writer),
		Policy:   agentcore.NewAllowList("write_file"),
		Definition: agentcore.AgentDefinition{
			Agents: "You are a world-class creative web designer. Produce the page ONLY by calling " +
				"write_file(path, content); never put HTML in your text reply. content must be a single, " +
				"self-contained HTML document (inline CSS, no external assets). Keep any reasoning extremely " +
				"brief — go straight to the write_file call. The write_file tool may fail transiently and " +
				"return an error — if it does, simply call it again with the same content until it succeeds. " +
				"Do not give up after a tool error. Once the file is written, reply with a one-line confirmation.",
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	const prompt = "Create a VERY VERY creative, visually striking landing page for a fictional product called " +
		"\"Nimbus\" — a cloud-powered note-taking app. Single-file HTML with bold, original inline CSS: a hero " +
		"headline, a short pitch, a few feature highlights, and a call-to-action button. Surprise me."
	res, err := agent.Prompt(ctx, prompt)
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	// Behavior log: what the agent actually did.
	writer.mu.Lock()
	attempts, failed := writer.attempts, writer.errors
	writer.mu.Unlock()
	t.Logf("behavior: turns=%d stop=%q write_attempts=%d transient_failures=%d final=%q",
		res.Turns, res.StopReason, attempts, failed, snippet(res.Final, 160))

	// (1) The agent genuinely encountered tool errors and did not stop on them.
	var sawToolError bool
	for _, tr := range res.Tools {
		if tr.Tool == "write_file" && tr.Error != "" {
			sawToolError = true
		}
	}
	if !sawToolError {
		t.Fatalf("test did not exercise the error path (no write_file error recorded); traces=%+v", res.Tools)
	}
	if failed == 0 || attempts <= failed {
		t.Fatalf("expected the agent to retry past %d transient failures, attempts=%d", failed, attempts)
	}

	// (2) Despite the failures, it delivered: a substantial self-contained HTML
	// landing page for the product landed in a file.
	path, page := writer.best()
	if page == "" {
		t.Fatalf("agent never successfully wrote the page despite retrying; gave up after errors")
	}
	low := strings.ToLower(page)
	if !strings.Contains(low, "<html") || !strings.Contains(low, "</html>") {
		t.Fatalf("written file is not a self-contained HTML document: %q", snippet(page, 200))
	}
	if !strings.Contains(low, "nimbus") {
		t.Fatalf("the page is not about the requested product 'Nimbus': %q", snippet(page, 200))
	}
	if len(page) < 500 {
		t.Fatalf("the page is too thin to be a real landing page (%d bytes)", len(page))
	}
	// Loose creativity signal: a striking page carries some styling + structure.
	if !strings.Contains(low, "<style") && !strings.Contains(low, "style=") {
		t.Fatalf("no inline CSS — not the bold styled page requested: %q", snippet(page, 200))
	}

	// Persist the artifact for human inspection when a dir is provided.
	if dir := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_ARTIFACT_DIR")); dir != "" {
		name := filepath.Base(path)
		if name == "" || name == "." || name == "/" {
			name = "nimbus.html"
		}
		out := filepath.Join(dir, name)
		if err := os.WriteFile(out, []byte(page), 0o644); err != nil {
			t.Logf("could not save artifact: %v", err)
		} else {
			t.Logf("saved generated landing page (%d bytes) to %s", len(page), out)
		}
	}
}

// snippet returns at most n runes of s on a single line, for compact logs.
func snippet(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

// TestReal_ToolCall_And_WebFetch proves tool calling + web fetch: given the
// http_request tool allowlisted to example.com and a plain instruction, the
// model decides to call it and reports content fetched from the live page. The
// outbound request is real, so run with network access.
func TestReal_ToolCall_And_WebFetch(t *testing.T) {
	provider, model := realProvider(t)
	web := sandbox.NewHTTPRequestTool(nil, sandbox.WithHTTPAllowHosts([]string{"example.com"}))

	agent, err := agentcore.New(agentcore.Config{
		Provider: provider,
		Model:    model,
		Tools:    agentcore.NewToolSet(web),
		Policy:   agentcore.NewAllowList(sandbox.ToolHTTPRequest),
		Definition: agentcore.AgentDefinition{
			Agents: "You can fetch web pages with the http_request tool (only allowlisted hosts). " +
				"When asked about a web page, actually fetch it before answering.",
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	res, err := agent.Prompt(ctx, "Fetch https://example.com and tell me the exact text of the page's main heading.")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	t.Logf("final: %q", res.Final)

	if !toolWasCalled(res, sandbox.ToolHTTPRequest) {
		t.Fatalf("the model never called http_request; traces=%+v", res.Tools)
	}
	if !strings.Contains(res.Final, "Example Domain") {
		t.Fatalf("answer did not reflect the fetched page (expected 'Example Domain'): %q", res.Final)
	}
}

// TestReal_SteeringMidRun proves mid-session steering changes the model's
// behavior: a fact injected via the steering queue before the second turn (the
// session having been extended by a follow-up) is used in the answer. It
// combines the follow-up restart and the steering inject — both observed in the
// recorded turn-2 request and reflected in the model's output.
func TestReal_SteeringMidRun(t *testing.T) {
	provider, model := realProvider(t)
	rec := &recordingProvider{inner: provider}

	var steered, followed bool
	agent, err := agentcore.New(agentcore.Config{
		Provider: rec,
		Model:    model,
		GetSteeringMessages: func(context.Context) []agentcore.Message {
			if steered {
				return nil
			}
			steered = true
			return []agentcore.Message{{Role: agentcore.RoleUser, Content: "New information: my favorite color is teal."}}
		},
		GetFollowUpMessages: func(context.Context) []agentcore.Message {
			if followed {
				return nil
			}
			followed = true
			return []agentcore.Message{{Role: agentcore.RoleUser, Content: "Now tell me my favorite color."}}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	res, err := agent.Prompt(ctx,
		"Do you know my favorite color? If you don't know it yet, reply with exactly the word UNKNOWN.")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	t.Logf("final: %q (turns=%d)", res.Final, res.Turns)

	// Mechanism: the steering message must have entered a later request.
	var injected bool
	for _, req := range rec.requests() {
		for _, m := range req.Messages {
			if strings.Contains(m.Content, "favorite color is teal") {
				injected = true
			}
		}
	}
	if !injected {
		t.Fatal("steering message was never injected into a turn's request")
	}
	// Behavior: the steered fact shaped the final answer.
	if !strings.Contains(strings.ToLower(res.Final), "teal") {
		t.Fatalf("steering did not influence the answer (expected 'teal'): %q", res.Final)
	}
}

// TestReal_TodoPlanSurvivesLongSession proves the todo/plan persists across a
// long, auto-compacting session: the model writes a plan with update_plan and
// works step-by-step over many turns under a deliberately tiny context budget
// that forces repeated compaction. The assertion is the property the user cares
// about — the plan and the original goal are still pinned into the LAST request
// the model saw, after compaction has run.
func TestReal_TodoPlanSurvivesLongSession(t *testing.T) {
	provider, model := realProvider(t)
	rec := &recordingProvider{inner: provider}

	plan := todo.NewStore()
	step := &recordTool{name: "do_step", reply: "step done"}

	limits := agentcore.DefaultLimits()
	limits.MaxTurns = 10
	limits.MaxContextTokens = 1200 // tiny: forces compaction within a few turns

	agent, err := agentcore.New(agentcore.Config{
		Provider: rec,
		Model:    model,
		Limits:   &limits,
		Tools:    agentcore.NewToolSet(todo.NewTool(plan), step),
		Policy:   agentcore.NewAllowList(todo.ToolName, "do_step"),
		Hooks:    agentcore.Hooks{PiContext: []agentcore.PiContextHook{todo.PiContextHook(plan)}},
		Definition: agentcore.AgentDefinition{
			Agents: "Work methodically and ALWAYS use the tools. Your VERY FIRST action MUST be a single " +
				"update_plan call recording a four-step plan (steps: gather, analyze, draft, review) with " +
				"the first step in_progress and the rest pending — do this before saying anything. Then " +
				"complete the steps one per turn: call do_step for the current step, then call update_plan " +
				"to mark it completed and move the next to in_progress. When all four are completed, reply " +
				"with exactly: ALL STEPS DONE.",
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	const goal = "Produce a short market report by following your four-step plan."
	res, err := agent.Prompt(ctx, goal)
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	t.Logf("final: %q (turns=%d, steps=%d)", res.Final, res.Turns, step.calls())

	if !toolWasCalled(res, todo.ToolName) {
		t.Fatalf("the model never wrote a plan with update_plan; traces=%+v", res.Tools)
	}
	// The model must have actually recorded a plan in the store.
	items := plan.List()
	if len(items) == 0 {
		t.Fatal("update_plan ran but no plan landed in the store")
	}
	// It must have been a genuine multi-turn working session (so the pin is
	// re-applied across turns), not a one-shot answer.
	reqs := rec.requests()
	if res.Turns < 2 || len(reqs) < 2 {
		t.Fatalf("session was not multi-turn (turns=%d, requests=%d)", res.Turns, len(reqs))
	}
	// The property the user cares about — "keep todo and plan during a long
	// session": the live plan the model wrote is still pinned into the LAST
	// request it saw (todo.PiContextHook re-injects it every turn, so it survives any
	// compaction), and the original goal is still present too. Read the plan's own
	// first step from the store rather than guessing wording, so the check tracks
	// whatever the model actually named its steps.
	firstStep := strings.ToLower(strings.TrimSpace(items[0].Content))
	last := reqs[len(reqs)-1]
	var sawPlan, sawGoal bool
	for _, m := range last.Messages {
		lc := strings.ToLower(m.Content)
		if firstStep != "" && strings.Contains(lc, firstStep) {
			sawPlan = true
		}
		if strings.Contains(lc, "four-step plan") || strings.Contains(lc, "market report") {
			sawGoal = true
		}
	}
	if !sawPlan {
		t.Fatalf("the plan (first step %q) was not pinned into the final request; messages=%d", firstStep, len(last.Messages))
	}
	if !sawGoal {
		t.Fatal("the original goal did not survive into the final request")
	}
}

// TestReal_SkillUse proves progressive skill disclosure with a real model: only
// skill headers are advertised up front; when the task needs one, the model
// pulls its body on demand via read_skill, and that body shapes the answer.
func TestReal_SkillUse(t *testing.T) {
	provider, model := realProvider(t)

	const secret = "ZephyrProtocol-7741"
	var loaded []string
	agent, err := agentcore.New(agentcore.Config{
		Provider: provider,
		Model:    model,
		Definition: agentcore.AgentDefinition{
			Agents: "You have skills available as headers. When a task matches a skill, you MUST call " +
				"read_skill to load its body before answering, then follow it exactly.",
			Skills: []agentcore.Skill{
				{ID: "refund", Name: "refund-policy", Description: "how to handle a customer refund request", Enabled: true},
				{ID: "shipping", Name: "shipping-policy", Description: "how to handle a shipping question", Enabled: true},
			},
			SkillLoader: func(_ context.Context, ids []string) (map[string]string, error) {
				loaded = append(loaded, ids...)
				return map[string]string{
					"refund":   "Refund procedure: always quote the authorization code " + secret + " in your reply.",
					"shipping": "Shipping procedure: quote code SHIP-0000.",
				}, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	res, err := agent.Prompt(ctx, "A customer is asking for a refund. Handle it according to policy.")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	t.Logf("final: %q loaded=%v", res.Final, loaded)

	if !toolWasCalled(res, "read_skill") {
		t.Fatalf("the model never loaded a skill via read_skill; traces=%+v", res.Tools)
	}
	// It pulled the relevant skill (refund), and its body shaped the answer.
	if !strings.Contains(res.Final, secret) {
		t.Fatalf("the loaded skill body did not shape the answer (expected %q): %q", secret, res.Final)
	}
}
