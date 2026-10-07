package subagent_test

import (
	"context"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/subagent"
)

// spawnTool reaches the model-facing tool the way the loop does: begin the
// plugin for a run, then take the tool it contributes.
func spawnTool(t *testing.T, durable bool, runner subagent.ForkRunner) interface {
	agentcore.Tool
	RetrySafeCall(agentcore.ToolCall) bool
} {
	t.Helper()
	ext, err := (subagent.Plugin{RunFork: runner}).BeginRun(context.Background(), agentcore.RunInfo{Durable: durable})
	if err != nil || ext == nil {
		t.Fatalf("BeginRun = %v, %v", ext, err)
	}
	tools := ext.(interface{ Tools() []agentcore.Tool }).Tools()
	if len(tools) != 1 {
		t.Fatalf("expected exactly one contributed tool, got %d", len(tools))
	}
	tool, ok := tools[0].(interface {
		agentcore.Tool
		RetrySafeCall(agentcore.ToolCall) bool
	})
	if !ok {
		t.Fatalf("%T must declare per-call retry safety", tools[0])
	}
	return tool
}

// TestSpawnRetrySafetyIsCallConditional pins the spawn_subagent retry contract:
// only self-forks are retry-safe (their child session ID is deterministic in
// the call, so a replay reattaches), while delegate-routed spawns — which have
// no reattach wiring — and malformed calls are not.
func TestSpawnRetrySafetyIsCallConditional(t *testing.T) {
	runner := func(context.Context, *agentcore.Agent, subagent.ForkRequest, agentcore.StreamSink) (agentcore.RunResult, error) {
		return agentcore.RunResult{}, nil
	}
	tool := spawnTool(t, true, runner)
	for args, want := range map[string]bool{
		`{"task":"t"}`:                   true,
		`{"task":"t","agent":"self"}`:    true,
		`{"task":"t","agent":"Self"}`:    true,
		`{"task":"t","agent":"analyst"}`: false,
		`not json`:                       false,
		`{"task":""}`:                    false,
		`{"task":"t","async":true}`:      false,
	} {
		if got := tool.RetrySafeCall(agentcore.ToolCall{Name: subagent.ToolSpawnSubagent, Arguments: args}); got != want {
			t.Errorf("RetrySafeCall(%s) = %v, want %v", args, got, want)
		}
	}

	for _, unsafe := range []interface{ RetrySafeCall(agentcore.ToolCall) bool }{spawnTool(t, false, runner), spawnTool(t, true, nil)} {
		if unsafe.RetrySafeCall(agentcore.ToolCall{Name: subagent.ToolSpawnSubagent, Arguments: `{"task":"t"}`}) {
			t.Fatal("spawn without durable native host advertised replay safety")
		}
	}
}
