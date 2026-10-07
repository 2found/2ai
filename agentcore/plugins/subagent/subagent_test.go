package subagent_test

import (
	"context"
	"github.com/2found/2ai/ai"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/subagent"
)

// subagentAgent wires a minimal agent with delegation enabled: the echo tool is
// the only host tool, spawn_subagent + echo are policy-permitted, and the faux
// provider replays the given script (shared by parent and child runs, in call
// order).
func subagentAgent(t *testing.T, settings *subagent.Plugin, script ...ai.Message) (*agentcore.Agent, *nativeScript) {
	t.Helper()
	provider := newNativeScript(script...)
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: provider.Provider,
		Model:          "faux-1",
		Tools:          agentcore.NewToolSet(&echoTool{name: "echo"}),
		Policy:         agentcore.NewAllowList("echo", subagent.ToolSpawnSubagent),
		Extensions:     []agentcore.ExtensionFactory{settings},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return agent, provider
}

// TestSubagentRunsIsolatedChildAndReturnsFinal proves the core delegation loop:
// the parent spawns a child, the child does its own tool work in an isolated
// history, and only the child's final answer returns to the parent.
func TestSubagentRunsIsolatedChildAndReturnsFinal(t *testing.T) {
	agent, provider := subagentAgent(t, &subagent.Plugin{},
		// Parent turn 1: delegate.
		AssistantToolCall("c1", subagent.ToolSpawnSubagent, `{"task":"echo the word banana and report what came back"}`),
		// Child turn 1: use a tool inside the child run.
		AssistantToolCall("c2", "echo", `{"text":"banana"}`),
		// Child turn 2: child's final answer.
		nativeAnswer("the echo returned: banana"),
		// Parent turn 2: parent's final answer, quoting the child's result.
		nativeAnswer("child reported: banana"),
	)

	res, err := agent.Prompt(context.Background(), "delegate an echo test")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "child reported: banana" {
		t.Fatalf("final = %q", res.Final)
	}
	// The spawn tool's result fed to the parent is the child's final answer.
	var spawnResult string
	for _, m := range res.Messages {
		if m.Role == agentcore.RoleTool && m.Name == subagent.ToolSpawnSubagent {
			spawnResult = m.Content
		}
	}
	if spawnResult != "the echo returned: banana" {
		t.Fatalf("spawn result = %q", spawnResult)
	}
	// Isolation: the child's intermediate tool traffic (the echo call/result)
	// must not appear in the parent transcript.
	for _, m := range res.Messages {
		if m.Role == agentcore.RoleTool && m.Name == "echo" {
			t.Fatalf("child tool result leaked into parent transcript: %q", m.Content)
		}
	}
	// The child run saw a fresh history: its first request contains only the
	// delegated task, not the parent's user prompt.
	childReq := provider.Recorded[1]
	for _, m := range childReq.Messages {
		if strings.Contains(m.Content, "delegate an echo test") {
			t.Fatalf("parent conversation leaked into child context: %q", m.Content)
		}
	}
}

// TestSubagentDepthCap proves a child at MaxDepth is not offered spawn_subagent:
// a grandchild spawn attempt fails as an unknown tool instead of recursing.
func TestSubagentDepthCap(t *testing.T) {
	agent, _ := subagentAgent(t, &subagent.Plugin{}, // MaxDepth defaults to 1
		AssistantToolCall("c1", subagent.ToolSpawnSubagent, `{"task":"try to delegate again"}`),
		// Child tries to spawn a grandchild — the tool is not registered at depth 1.
		AssistantToolCall("c2", subagent.ToolSpawnSubagent, `{"task":"grandchild"}`),
		nativeAnswer("could not delegate further"),
		nativeAnswer("done"),
	)

	res, err := agent.Prompt(context.Background(), "nest")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "done" {
		t.Fatalf("final = %q", res.Final)
	}
	// The grandchild attempt surfaced as an unknown-tool error inside the child
	// (visible only via the trace-free child transcript — assert indirectly: the
	// parent's spawn result is the child's recovery answer, not a recursion).
	var spawnResult string
	for _, m := range res.Messages {
		if m.Role == agentcore.RoleTool && m.Name == subagent.ToolSpawnSubagent {
			spawnResult = m.Content
		}
	}
	if spawnResult != "could not delegate further" {
		t.Fatalf("spawn result = %q", spawnResult)
	}
}

// TestSubagentPerRunBudget proves the per-run spawn cap: once exhausted, a
// further spawn fails fast with a budget error and consumes no provider call.
func TestSubagentPerRunBudget(t *testing.T) {
	agent, _ := subagentAgent(t, &subagent.Plugin{MaxPerRun: 1},
		AssistantToolCall("c1", subagent.ToolSpawnSubagent, `{"task":"first"}`),
		nativeAnswer("first child answer"),
		AssistantToolCall("c2", subagent.ToolSpawnSubagent, `{"task":"second"}`),
		// No child script needed: the second spawn is refused before any call.
		nativeAnswer("wrapped up without the second child"),
	)

	res, err := agent.Prompt(context.Background(), "fan out")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	var results []string
	for _, m := range res.Messages {
		if m.Role == agentcore.RoleTool && m.Name == subagent.ToolSpawnSubagent {
			results = append(results, m.Content)
		}
	}
	if len(results) != 2 {
		t.Fatalf("want 2 spawn results, got %d", len(results))
	}
	if results[0] != "first child answer" {
		t.Fatalf("first spawn = %q", results[0])
	}
	if !strings.Contains(results[1], "budget exhausted") {
		t.Fatalf("second spawn should be budget-blocked, got %q", results[1])
	}
}

// TestSubagentOutputCapAndUsageFolding proves the child's answer is middle-
// truncated to MaxOutputBytes and the child's token usage folds into the
// parent run's accounting.
func TestSubagentOutputCapAndUsageFolding(t *testing.T) {
	long := strings.Repeat("A", 400) + "TAIL-SIGNAL"
	childResp := nativeAnswer(long)
	childResp.Usage = &ai.Usage{Input: 100, Output: 40, Cost: ai.UsageCost{Total: 0.5}}
	agent, _ := subagentAgent(t, &subagent.Plugin{MaxOutputBytes: 256},
		AssistantToolCall("c1", subagent.ToolSpawnSubagent, `{"task":"produce a long answer"}`),
		childResp,
		nativeAnswer("done"),
	)

	res, err := agent.Prompt(context.Background(), "cap test")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	var spawnResult string
	for _, m := range res.Messages {
		if m.Role == agentcore.RoleTool && m.Name == subagent.ToolSpawnSubagent {
			spawnResult = m.Content
		}
	}
	if len(spawnResult) > 256 {
		t.Fatalf("spawn result exceeds cap: %d bytes", len(spawnResult))
	}
	if !strings.Contains(spawnResult, "bytes truncated") {
		t.Fatalf("expected truncation marker, got %q", spawnResult)
	}
	if !strings.HasSuffix(spawnResult, "TAIL-SIGNAL") {
		t.Fatalf("tail should be preserved, got %q", spawnResult)
	}
	if res.Usage.InputTokens < 100 || res.Usage.OutputTokens < 40 || res.Usage.CostUSD < 0.5 {
		t.Fatalf("child usage not folded into parent: %+v", res.Usage)
	}
}

// TestSubagentRoutesToNamedDelegate proves cross-agent delegation: a spawn
// with agent set routes the task to the granted delegate's Run closure (with
// the delegation depth incremented on ctx), the delegate's answer becomes the
// tool result, and its usage folds into the parent run. An unknown name is
// refused with the roster hint.
func TestSubagentRoutesToNamedDelegate(t *testing.T) {
	var gotTask string
	var gotDepth int
	provider := newNativeScript(
		AssistantToolCall("c1", subagent.ToolSpawnSubagent, `{"task":"summarize chapter 3","agent":"Writer"}`),
		AssistantToolCall("c2", subagent.ToolSpawnSubagent, `{"task":"x","agent":"Nobody"}`),
		nativeAnswer("done"),
	)
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: provider.Provider,
		Model:          "faux-1",
		Tools:          agentcore.NewToolSet(),
		Policy:         agentcore.NewAllowList(subagent.ToolSpawnSubagent),
		Extensions: []agentcore.ExtensionFactory{subagent.To(subagent.Delegate{
			Name:        "Writer",
			Description: "drafts prose",
			Run: func(ctx context.Context, task string, sink agentcore.StreamSink) (string, agentcore.Usage, error) {
				gotTask = task
				gotDepth = agentcore.DelegationDepth(ctx)
				return "chapter 3 summary", agentcore.Usage{InputTokens: 7, OutputTokens: 3, CostUSD: 0.01}, nil
			},
		})},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "delegate to the writer")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if gotTask != "summarize chapter 3" {
		t.Fatalf("delegate task = %q", gotTask)
	}
	if gotDepth != 1 {
		t.Fatalf("delegate depth = %d, want 1", gotDepth)
	}
	var results []string
	for _, m := range res.Messages {
		if m.Role == agentcore.RoleTool && m.Name == subagent.ToolSpawnSubagent {
			results = append(results, m.Content)
		}
	}
	if len(results) != 2 {
		t.Fatalf("want 2 spawn results, got %d: %v", len(results), results)
	}
	if results[0] != "chapter 3 summary" {
		t.Fatalf("delegate result = %q", results[0])
	}
	if !strings.Contains(results[1], "unknown agent") {
		t.Fatalf("unknown delegate should be refused, got %q", results[1])
	}
	if res.Usage.InputTokens < 7 || res.Usage.CostUSD < 0.01 {
		t.Fatalf("delegate usage not folded: %+v", res.Usage)
	}
	// The roster is advertised in the schema so the model can pick a teammate.
	var schemaDesc string
	for _, req := range provider.Recorded {
		for _, s := range req.Tools {
			if s.Name == subagent.ToolSpawnSubagent {
				if p, ok := s.Parameters["properties"].(map[string]any); ok {
					if ag, ok := p["agent"].(map[string]any); ok {
						schemaDesc, _ = ag["description"].(string)
					}
				}
			}
		}
	}
	if !strings.Contains(schemaDesc, "Writer — drafts prose") {
		t.Fatalf("roster missing from schema: %q", schemaDesc)
	}
}

// TestSubagentDisabledWithoutConfig proves an agent without Config.Subagents
// never advertises or accepts spawn_subagent even when policy would permit it.
func TestSubagentDisabledWithoutConfig(t *testing.T) {
	provider := newNativeScript(
		AssistantToolCall("c1", subagent.ToolSpawnSubagent, `{"task":"x"}`),
		nativeAnswer("ok"),
	)
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: provider.Provider,
		Model:          "faux-1",
		Tools:          agentcore.NewToolSet(&echoTool{name: "echo"}),
		Policy:         agentcore.NewAllowList("echo", subagent.ToolSpawnSubagent),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "try")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	for _, req := range provider.Recorded {
		for _, s := range req.Tools {
			if s.Name == subagent.ToolSpawnSubagent {
				t.Fatalf("spawn_subagent advertised without Config.Subagents")
			}
		}
	}
	var toolMsg string
	for _, m := range res.Messages {
		if m.Role == agentcore.RoleTool && m.Name == subagent.ToolSpawnSubagent {
			toolMsg = m.Content
		}
	}
	if !strings.Contains(toolMsg, "spawn_subagent not found") {
		t.Fatalf("expected unknown-tool result, got %q", toolMsg)
	}
}
