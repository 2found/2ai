package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

type evalBridgeProbeTool struct {
	calls atomic.Int32
	mu    sync.Mutex
	args  []string
}

func (p *evalBridgeProbeTool) Name() string { return "bridge_probe" }

func (p *evalBridgeProbeTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:        p.Name(),
		Description: "return a tagged value",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"value": map[string]any{"type": "string"},
			},
			"required": []string{"value"},
		},
	}
}

func (p *evalBridgeProbeTool) Run(_ context.Context, args string) (string, error) {
	p.calls.Add(1)
	p.mu.Lock()
	p.args = append(p.args, args)
	p.mu.Unlock()
	var in struct {
		Value string `json:"value"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", err
	}
	return "probe:" + in.Value, nil
}

func evalBridgeAgent(t *testing.T, language, code string, allowed []string, limits *agentcore.Limits) (*evalBridgeProbeTool, agentcore.RunResult) {
	t.Helper()
	var eval *EvalTool
	if language == "javascript" {
		eval, _, _ = testJavaScriptEvalTool(t, EvalConfig{})
	} else {
		eval, _, _ = testEvalTool(t, EvalConfig{})
	}
	probe := &evalBridgeProbeTool{}
	args, err := json.Marshal(map[string]any{"language": language, "code": code})
	if err != nil {
		t.Fatal(err)
	}
	provider := sandboxNativeFixture(
		agentcore.AssistantToolCall("eval-call", ToolEval, string(args)),
		agentcore.AssistantText("done"),
	)
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: provider,
		Model:          "faux",
		Tools:          agentcore.NewToolSet(eval, probe),
		Policy:         agentcore.NewAllowList(allowed...),
		Limits:         limits,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(evalContext("bridge-"+language+"-"+strings.ReplaceAll(code, " ", "_")), "use eval")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	return probe, res
}

func toolMessageFor(t *testing.T, result agentcore.RunResult, name string) agentcore.Message {
	t.Helper()
	for _, message := range result.Messages {
		if message.Role == agentcore.RoleTool && message.Name == name {
			return message
		}
	}
	t.Fatalf("no %s tool message in %+v", name, result.Messages)
	return agentcore.Message{}
}

func TestEvalHostToolBridgeJavaScriptAndPython(t *testing.T) {
	for _, test := range []struct {
		language string
		code     string
		want     string
	}{
		{"javascript", `await tool.bridge_probe({value: "javascript"})`, "probe:javascript"},
		{"python", `tool("bridge_probe", {"value": "python"})`, "probe:python"},
	} {
		t.Run(test.language, func(t *testing.T) {
			probe, result := evalBridgeAgent(t, test.language, test.code, []string{ToolEval, probeToolName}, nil)
			if probe.calls.Load() != 1 {
				t.Fatalf("probe calls = %d", probe.calls.Load())
			}
			if message := toolMessageFor(t, result, ToolEval); !strings.Contains(message.Content, test.want) {
				t.Fatalf("eval result = %q, want %q", message.Content, test.want)
			}
			if len(result.Tools) != 2 || result.Tools[0].Tool != ToolEval || result.Tools[1].Tool != probeToolName {
				t.Fatalf("traces = %+v, want outer eval then nested probe", result.Tools)
			}
			if !result.Tools[1].Allowed || !strings.HasPrefix(result.Tools[1].CallID, "eval-call/bridge-") {
				t.Fatalf("nested trace = %+v", result.Tools[0])
			}
		})
	}
}

const probeToolName = "bridge_probe"

func TestEvalHostToolBridgeCannotBypassPolicyOrSchema(t *testing.T) {
	for _, test := range []struct {
		name    string
		code    string
		allowed []string
		want    string
	}{
		{
			name:    "policy",
			code:    `await (async () => { try { await tool.bridge_probe({value: "denied"}) } catch (error) { return error.message } })()`,
			allowed: []string{ToolEval},
			want:    "blocked",
		},
		{
			name:    "schema",
			code:    `await (async () => { try { await tool.bridge_probe({}) } catch (error) { return error.message } })()`,
			allowed: []string{ToolEval, probeToolName},
			want:    "missing required",
		},
		{
			name:    "recursion",
			code:    `await (async () => { try { await tool.eval({language: "javascript", code: "1"}) } catch (error) { return error.message } })()`,
			allowed: []string{ToolEval, probeToolName},
			want:    "recursive nested tool call",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			probe, result := evalBridgeAgent(t, "javascript", test.code, test.allowed, nil)
			if probe.calls.Load() != 0 {
				t.Fatalf("blocked probe executed %d times", probe.calls.Load())
			}
			message := toolMessageFor(t, result, ToolEval)
			if !strings.Contains(message.Content, test.want) {
				t.Fatalf("eval result = %q, want %q", message.Content, test.want)
			}
			if len(result.Tools) != 2 || !result.Tools[0].Allowed || result.Tools[1].Allowed {
				t.Fatalf("traces = %+v, want allowed eval and blocked nested trace", result.Tools)
			}
		})
	}
}

func TestEvalHostToolBridgeSharesRunToolBudget(t *testing.T) {
	limits := agentcore.DefaultLimits()
	limits.MaxToolCalls = 2
	code := `
const first = await tool.bridge_probe({value: "first"});
let second;
try { second = await tool.bridge_probe({value: "second"}); }
catch (error) { second = error.message; }
({first, second})`
	probe, result := evalBridgeAgent(t, "javascript", code, []string{ToolEval, probeToolName}, &limits)
	if probe.calls.Load() != 1 {
		t.Fatalf("probe calls = %d, want one execution within eval's remaining slot", probe.calls.Load())
	}
	message := toolMessageFor(t, result, ToolEval)
	if !strings.Contains(message.Content, "probe:first") || !strings.Contains(message.Content, "tool-call budget exhausted") {
		t.Fatalf("eval result = %q", message.Content)
	}
	if len(result.Tools) != 3 {
		t.Fatalf("traces = %+v, want allowed probe, blocked probe, eval", result.Tools)
	}
	if !result.Tools[0].Allowed || !result.Tools[1].Allowed || result.Tools[2].Allowed {
		t.Fatalf("trace permissions = %+v", result.Tools)
	}
}

func TestEvalHostToolBridgeEnforcesPerCellCap(t *testing.T) {
	eval, _, _ := testJavaScriptEvalTool(t, EvalConfig{MaxBridgeCalls: 1})
	probe := &evalBridgeProbeTool{}
	code := `
const first = await tool.bridge_probe({value: "first"});
let second;
try { second = await tool.bridge_probe({value: "second"}); }
catch (error) { second = error.message; }
({first, second})`
	args, _ := json.Marshal(map[string]any{"language": "javascript", "code": code})
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: sandboxNativeFixture(
			agentcore.AssistantToolCall("eval-call", ToolEval, string(args)),
			agentcore.AssistantText("done"),
		),
		Model: "faux", Tools: agentcore.NewToolSet(eval, probe),
		Policy: agentcore.NewAllowList(ToolEval, probeToolName),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	result, err := agent.Prompt(evalContext("bridge-cell-cap"), "use eval")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if probe.calls.Load() != 1 {
		t.Fatalf("probe calls = %d", probe.calls.Load())
	}
	if message := toolMessageFor(t, result, ToolEval); !strings.Contains(message.Content, "per cell") {
		t.Fatalf("eval result = %q", message.Content)
	}
	if len(result.Tools) != 3 || !result.Tools[1].Allowed || result.Tools[2].Allowed || result.Tools[2].Reason == "" {
		t.Fatalf("cap traces = %+v", result.Tools)
	}
}

func TestEvalHostToolBridgeRunsHooksAndPairsStreamEvents(t *testing.T) {
	eval, _, _ := testJavaScriptEvalTool(t, EvalConfig{})
	probe := &evalBridgeProbeTool{}
	var beforeMu sync.Mutex
	var before []string
	hooks := agentcore.Hooks{
		Before: []agentcore.BeforeToolCall{func(_ context.Context, call agentcore.ToolCall) agentcore.Decision {
			beforeMu.Lock()
			before = append(before, call.Name)
			beforeMu.Unlock()
			return agentcore.Allowed()
		}},
		After: []agentcore.AfterToolCall{func(_ context.Context, call agentcore.ToolCall, result string, runErr error) (string, bool) {
			if runErr == nil {
				result += "|after:" + call.Name
			}
			return result, false
		}},
	}
	args, _ := json.Marshal(map[string]any{
		"language": "javascript",
		"code":     `await tool.bridge_probe({value: "hooked"})`,
	})
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: sandboxNativeFixture(
			agentcore.AssistantToolCall("eval-call", ToolEval, string(args)),
			agentcore.AssistantText("done"),
		),
		Model: "faux", Tools: agentcore.NewToolSet(eval, probe),
		Policy: agentcore.NewAllowList(ToolEval, probeToolName), Hooks: hooks,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	starts := map[string]int{}
	ends := map[string]int{}
	result, err := agent.PromptStream(evalContext("bridge-hooks"), "use eval", func(event agentcore.StreamEvent) {
		if event.Tool == nil {
			return
		}
		switch event.Type {
		case agentcore.StreamToolExecStart:
			starts[event.Tool.CallID]++
		case agentcore.StreamToolExecEnd:
			ends[event.Tool.CallID]++
		}
	})
	if err != nil {
		t.Fatalf("PromptStream: %v", err)
	}
	message := toolMessageFor(t, result, ToolEval)
	if !strings.Contains(message.Content, "probe:hooked|after:bridge_probe") || !strings.Contains(message.Content, "after:eval") {
		t.Fatalf("hooked eval result = %q", message.Content)
	}
	beforeMu.Lock()
	gotBefore := append([]string(nil), before...)
	beforeMu.Unlock()
	if strings.Join(gotBefore, ",") != "eval,bridge_probe" {
		t.Fatalf("before hooks = %v", gotBefore)
	}
	for _, trace := range result.Tools {
		if starts[trace.CallID] != 1 || ends[trace.CallID] != 1 {
			t.Fatalf("unpaired stream events for %q: starts=%d ends=%d", trace.CallID, starts[trace.CallID], ends[trace.CallID])
		}
	}
}

type evalBridgeSlowTool struct{ started chan struct{} }

func (*evalBridgeSlowTool) Name() string { return "bridge_slow" }
func (*evalBridgeSlowTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{Name: "bridge_slow", Parameters: map[string]any{"type": "object"}}
}
func (s *evalBridgeSlowTool) Run(ctx context.Context, _ string) (string, error) {
	close(s.started)
	<-ctx.Done()
	return "", ctx.Err()
}

type evalBridgeConcurrencyTool struct {
	name     string
	parallel bool
	active   atomic.Int32
	maximum  atomic.Int32
}

func (t *evalBridgeConcurrencyTool) Name() string { return t.name }
func (t *evalBridgeConcurrencyTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{Name: t.name, Parameters: map[string]any{"type": "object"}}
}
func (t *evalBridgeConcurrencyTool) Parallel() bool { return t.parallel }
func (t *evalBridgeConcurrencyTool) Run(ctx context.Context, _ string) (string, error) {
	active := t.active.Add(1)
	defer t.active.Add(-1)
	for {
		maximum := t.maximum.Load()
		if active <= maximum || t.maximum.CompareAndSwap(maximum, active) {
			break
		}
	}
	select {
	case <-time.After(75 * time.Millisecond):
		return "ok", nil
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func TestEvalHostToolBridgeHonorsParallelToolContract(t *testing.T) {
	for _, parallel := range []bool{false, true} {
		name := "bridge_serial"
		wantMax := int32(1)
		if parallel {
			name = "bridge_parallel"
			wantMax = 2
		}
		t.Run(name, func(t *testing.T) {
			eval, _, _ := testJavaScriptEvalTool(t, EvalConfig{})
			probe := &evalBridgeConcurrencyTool{name: name, parallel: parallel}
			code := fmt.Sprintf(`await Promise.all([tool.%s({}), tool.%s({})])`, name, name)
			args, _ := json.Marshal(map[string]any{"language": "javascript", "code": code})
			agent, err := agentcore.New(agentcore.Config{
				NativeProvider: sandboxNativeFixture(
					agentcore.AssistantToolCall("eval-call", ToolEval, string(args)),
					agentcore.AssistantText("done"),
				),
				Model: "faux", Tools: agentcore.NewToolSet(eval, probe),
				Policy: agentcore.NewAllowList(ToolEval, name),
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			result, err := agent.Prompt(evalContext("bridge-concurrency-"+name), "use eval")
			if err != nil {
				t.Fatalf("Prompt: %v", err)
			}
			if probe.maximum.Load() != wantMax {
				t.Fatalf("maximum concurrency = %d, want %d; traces=%+v", probe.maximum.Load(), wantMax, result.Tools)
			}
			if len(result.Tools) != 3 || result.Tools[1].Tool != name || result.Tools[2].Tool != name {
				t.Fatalf("traces = %+v", result.Tools)
			}
		})
	}
}

func TestEvalHostToolBridgeCancellationSettlesNestedTrace(t *testing.T) {
	eval, _, _ := testJavaScriptEvalTool(t, EvalConfig{TimeoutSeconds: 2})
	slow := &evalBridgeSlowTool{started: make(chan struct{})}
	args, _ := json.Marshal(map[string]any{
		"language":        "javascript",
		"timeout_seconds": 1,
		"code":            `await tool.bridge_slow({})`,
	})
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: sandboxNativeFixture(
			agentcore.AssistantToolCall("eval-call", ToolEval, string(args)),
			agentcore.AssistantText("recovered from timeout"),
		),
		Model: "faux", Tools: agentcore.NewToolSet(eval, slow),
		Policy: agentcore.NewAllowList(ToolEval, slow.Name()),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	startedAt := time.Now()
	result, err := agent.Prompt(evalContext("bridge-cancellation"), "time out")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if time.Since(startedAt) > 3*time.Second {
		t.Fatalf("cancellation took too long: %s", time.Since(startedAt))
	}
	select {
	case <-slow.started:
	default:
		t.Fatal("slow bridged tool never started")
	}
	if len(result.Tools) != 2 || result.Tools[1].Tool != slow.Name() || result.Tools[1].Error == "" || result.Tools[0].Tool != ToolEval {
		t.Fatalf("timeout traces = %+v, want eval and settled nested failure", result.Tools)
	}
}

func TestEvalHostToolBridgeUnavailableForDirectToolConstruction(t *testing.T) {
	for _, test := range []struct {
		language string
		code     string
	}{
		{"javascript", `await tool.bridge_probe({value: "x"})`},
		{"python", `tool("bridge_probe", {"value": "x"})`},
	} {
		t.Run(test.language, func(t *testing.T) {
			var tool *EvalTool
			if test.language == "javascript" {
				tool, _, _ = testJavaScriptEvalTool(t, EvalConfig{})
			} else {
				tool, _, _ = testEvalTool(t, EvalConfig{})
			}
			args := fmt.Sprintf(`{"language":%q,"code":%q}`, test.language, test.code)
			_, err := tool.Run(evalContext("direct-bridge-"+test.language), args)
			if err == nil || !strings.Contains(err.Error(), "unavailable outside a live agent run") {
				t.Fatalf("direct bridge error = %v", err)
			}
		})
	}
}
