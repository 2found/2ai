package goal_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/goal"
	"github.com/2found/2ai/ai"
)

func lifecycleAgent(t *testing.T, p goal.Plugin, replies ...ai.Message) *agentcore.Agent {
	t.Helper()
	p.Goal = "do the task"
	a, err := agentcore.New(agentcore.Config{NativeProvider: nativeProvider(ai.ScriptedStream(replies...)), Model: "test", Goal: p.Goal, Extensions: []agentcore.ExtensionFactory{p}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}
func stateOf(t *testing.T, r agentcore.RunResult) goal.State {
	t.Helper()
	var checkpoint struct {
		Plugins map[string]struct {
			Lifecycle goal.State `json:"lifecycle"`
		}
	}
	if err := json.Unmarshal(r.NativeState, &checkpoint); err != nil {
		t.Fatal(err)
	}
	return checkpoint.Plugins["goal"].Lifecycle
}
func TestLifecyclePauseResumeAcrossFreshAgents(t *testing.T) {
	ctx := context.Background()
	p := goal.Plugin{Lifecycle: true}
	paused, err := lifecycleAgent(t, p).RunNative(ctx, agentcore.NativeRun{ControlOnly: true, Commands: map[string]json.RawMessage{"goal": json.RawMessage(`{"action":"pause"}`)}})
	if err != nil || paused.Turns != 0 || stateOf(t, paused).Status != goal.Paused {
		t.Fatalf("pause: %s %v", paused.StopReason, err)
	}
	still, err := lifecycleAgent(t, p).RunNative(ctx, agentcore.NativeRun{State: paused.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "ordinary input cannot resume"}}})
	if err != nil || still.Turns != 0 || stateOf(t, still).Status != goal.Paused {
		t.Fatalf("paused state lost: %v", err)
	}
	result, err := lifecycleAgent(t, p, nativeAnswer("done\nSTATUS: DONE")).RunNative(ctx, agentcore.NativeRun{State: still.NativeState, Commands: map[string]json.RawMessage{"goal": json.RawMessage(`{"action":"resume"}`)}, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "continue"}}})
	if err != nil || result.Turns != 1 || stateOf(t, result).Status != goal.Complete {
		t.Fatalf("resume/complete: %s %+v %v", result.StopReason, stateOf(t, result), err)
	}
	if _, err := lifecycleAgent(t, p).RunNative(ctx, agentcore.NativeRun{State: result.NativeState, ControlOnly: true, Commands: map[string]json.RawMessage{"goal": json.RawMessage(`{"action":"resume"}`)}}); err == nil {
		t.Fatal("completed goal resumed")
	}
}
func TestLifecycleBudgetCannotBeResetByResume(t *testing.T) {
	p := goal.Plugin{Lifecycle: true, TokenBudget: 2}
	message := nativeAnswer("still working")
	message.Usage = &ai.Usage{Input: 2, Output: 1}
	first, err := lifecycleAgent(t, p, message).RunNative(context.Background(), agentcore.NativeRun{})
	if err != nil || first.Turns != 1 || stateOf(t, first).Status != goal.BudgetLimited || stateOf(t, first).TokensUsed != 3 {
		t.Fatalf("budget: %+v %v", stateOf(t, first), err)
	}
	if _, err := lifecycleAgent(t, p).RunNative(context.Background(), agentcore.NativeRun{State: first.NativeState, ControlOnly: true, Commands: map[string]json.RawMessage{"goal": json.RawMessage(`{"action":"resume"}`)}}); err == nil {
		t.Fatal("exhausted cap bypassed")
	}
	resumed, err := lifecycleAgent(t, p).RunNative(context.Background(), agentcore.NativeRun{State: first.NativeState, ControlOnly: true, Commands: map[string]json.RawMessage{"goal": json.RawMessage(`{"action":"resume","token_budget":20}`)}})
	if err != nil || stateOf(t, resumed).TokensUsed != 3 || stateOf(t, resumed).Status != goal.Active {
		t.Fatalf("new grant reset usage: %v", err)
	}
}
func TestLifecycleWallTimeAndInvalidCheckpoint(t *testing.T) {
	p := goal.Plugin{Lifecycle: true, TimeBudget: time.Millisecond}
	ext, err := p.BeginRun(context.Background(), agentcore.RunInfo{Goal: "work"})
	if err != nil {
		t.Fatal(err)
	}
	controller := ext.(agentcore.RunController)
	controller.ControlRun(context.Background(), agentcore.StepInfo{})
	time.Sleep(3 * time.Millisecond)
	reason, err := controller.ControlRun(context.Background(), agentcore.StepInfo{})
	if err != nil || reason != "goal_budget_limited" {
		t.Fatalf("wall time: %s %v", reason, err)
	}
	if err := ext.(agentcore.NativeStateContributor).RestoreNativeState(json.RawMessage(`{"lifecycle":{"status":"active","tokens_used":-1}}`)); err == nil {
		t.Fatal("negative counter accepted")
	}
}

type pausingTool struct{ store *goal.Store }

func (pausingTool) Name() string { return "pause_fixture" }
func (pausingTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{Name: "pause_fixture", Parameters: map[string]any{"type": "object"}}
}
func (p pausingTool) Run(ctx context.Context, _ string) (string, error) {
	_, err := p.store.ApplyCommand(ctx, goal.Command{Action: "pause", Reason: "host requested pause"})
	return "settled", err
}
func TestLifecycleLivePauseWaitsForSettledToolThenStops(t *testing.T) {
	store := goal.NewStore("work")
	a, err := agentcore.New(agentcore.Config{NativeProvider: nativeProvider(ai.ScriptedStream(nativeCall("pause", "pause_fixture", `{}`))), Model: "test", Goal: "work", Tools: agentcore.NewToolSet(pausingTool{store}), Policy: agentcore.NewAllowList("pause_fixture"), Extensions: []agentcore.ExtensionFactory{goal.Plugin{Lifecycle: true, Store: store}}})
	if err != nil {
		t.Fatal(err)
	}
	r, err := a.RunNative(context.Background(), agentcore.NativeRun{})
	if err != nil || r.Turns != 1 || r.StopReason != "goal_paused" || len(r.Tools) != 1 || r.Tools[0].Error != "" {
		t.Fatalf("pause boundary: %s turns=%d err=%v", r.StopReason, r.Turns, err)
	}
	if !strings.Contains(string(r.NativeState), "settled") {
		t.Fatal("pause lost settled tool receipt")
	}
}
