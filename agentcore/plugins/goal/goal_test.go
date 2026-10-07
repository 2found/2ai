package goal_test

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/2found/2ai/ai"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/goal"
)

// TestGoalGateHoldsUntilDone verifies a finish without a sentinel re-opens the
// run with the keep-going nudge, and a later STATUS: DONE answer is accepted.
func TestGoalGateHoldsUntilDone(t *testing.T) {
	faux := newNativeScript(
		nativeAnswer("made some progress"),
		nativeAnswer("still working on it"),
		nativeAnswer("all fixed.\nSTATUS: DONE"),
	)
	agent, err := agentcore.New(gated(agentcore.Config{
		NativeProvider: faux.Provider,
		Model:          "test",
		Goal:           "all tests pass",
	}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "fix the build")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if !strings.Contains(res.Final, "STATUS: DONE") {
		t.Fatalf("final = %q, want the STATUS: DONE answer", res.Final)
	}
	if res.StopReason != "stop" {
		t.Fatalf("stop reason = %q, want normal stop", res.StopReason)
	}
	if len(faux.Recorded) != 3 {
		t.Fatalf("provider calls = %d, want 3 (two nudges)", len(faux.Recorded))
	}
	// The contract must ride in the system prompt of every request.
	sys := faux.Recorded[0].Messages[0]
	if sys.Role != agentcore.RoleSystem || !strings.Contains(sys.Content, "all tests pass") || !strings.Contains(sys.Content, goal.Done) {
		t.Fatalf("goal contract missing from system prompt: %q", sys.Content)
	}
	// Each re-opened request must carry the nudge as a user message.
	var nudges int
	for _, req := range faux.Recorded {
		for _, m := range req.Messages {
			if m.Role == agentcore.RoleUser && strings.Contains(m.Content, "[goal gate]") {
				nudges++
				break
			}
		}
	}
	if nudges != 2 {
		t.Fatalf("nudge present in %d requests, want 2", nudges)
	}
}

// TestGoalGateAcceptsBlocked verifies the honest-blocker escape hatch: a
// STATUS: BLOCKED answer ends the run on the first finish.
func TestGoalGateAcceptsBlocked(t *testing.T) {
	faux := newNativeScript(
		nativeAnswer("Need prod credentials only you have.\nSTATUS: BLOCKED"),
	)
	agent, err := agentcore.New(gated(agentcore.Config{NativeProvider: faux.Provider, Model: "test", Goal: "deploy to prod"}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "ship it")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if len(faux.Recorded) != 1 {
		t.Fatalf("provider calls = %d, want 1 (no nudge)", len(faux.Recorded))
	}
	if res.StopReason != "stop" {
		t.Fatalf("stop reason = %q, want normal stop", res.StopReason)
	}
}

// TestGoalGateCaseInsensitive verifies sentinel matching tolerates casing.
func TestGoalGateCaseInsensitive(t *testing.T) {
	faux := newNativeScript(nativeAnswer("Everything works now. Status: done"))
	agent, err := agentcore.New(gated(agentcore.Config{NativeProvider: faux.Provider, Model: "test", Goal: "make it work"}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if len(faux.Recorded) != 1 {
		t.Fatalf("provider calls = %d, want 1 — lower-case sentinel must count", len(faux.Recorded))
	}
}

// TestGoalGateSentinelMustCloseAnswer verifies a mid-prose mention of the
// sentinel does not close the gate: only the answer's last non-empty line
// counts, per the contract's "end your final answer with the line".
func TestGoalGateSentinelMustCloseAnswer(t *testing.T) {
	faux := newNativeScript(
		nativeAnswer("I cannot yet write STATUS: DONE — the tests still fail.\nNext I will retry the flaky suite."),
		nativeAnswer("all green now.\nSTATUS: DONE"),
	)
	agent, err := agentcore.New(gated(agentcore.Config{NativeProvider: faux.Provider, Model: "test", Goal: "all tests pass"}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "fix the build")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if len(faux.Recorded) != 2 {
		t.Fatalf("provider calls = %d, want 2 — a mid-prose sentinel mention must nudge", len(faux.Recorded))
	}
	if !strings.Contains(res.Final, "all green") {
		t.Fatalf("final = %q, want the post-nudge answer", res.Final)
	}
}

// A failed run's native checkpoint restores the goal even when the consumer
// cannot re-supply Config.Goal. The recovered gate must still reject a finish
// without the completion sentinel.
func TestGoalGateSurvivesResume(t *testing.T) {
	ctx := context.Background()
	retry := agentcore.RetryPolicy{MaxAttempts: 1}
	first, err := agentcore.New(gated(agentcore.Config{NativeProvider: nativeProvider(func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		return nil, errors.New("provider unavailable")
	}), Model: "test", Goal: "the report is published", Retry: &retry}))
	if err != nil {
		t.Fatal(err)
	}
	failed, err := first.Prompt(ctx, "publish the report")
	if err == nil || len(failed.NativeState) == 0 {
		t.Fatalf("expected checkpointed failure: %v", err)
	}
	faux := newNativeScript(nativeAnswer("resumed, still working"), nativeAnswer("published.\nSTATUS: DONE"))
	agent, err := agentcore.New(gated(agentcore.Config{NativeProvider: faux.Provider, Model: "test"}))
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.RunNative(ctx, agentcore.NativeRun{State: failed.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "resume"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(faux.Recorded) != 2 {
		t.Fatalf("recovered goal did not nudge: %d requests", len(faux.Recorded))
	}
	var saved struct{ Goal string }
	if err := json.Unmarshal(res.NativeState, &saved); err != nil {
		t.Fatal(err)
	}
	if saved.Goal != "the report is published" || !strings.Contains(res.Final, goal.Done) {
		t.Fatalf("goal lost on resume: %+v %q", saved, res.Final)
	}
}

// TestGoalGateStallBreaker verifies a model repeating the same sentinel-free
// answer after a nudge stops the run as goal_stalled instead of burning turns.
func TestGoalGateStallBreaker(t *testing.T) {
	faux := newNativeScript(
		nativeAnswer("I cannot make further progress."),
		nativeAnswer("I cannot make further progress."),
	)
	agent, err := agentcore.New(gated(agentcore.Config{NativeProvider: faux.Provider, Model: "test", Goal: "impossible thing"}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "try")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.StopReason != "goal_stalled" {
		t.Fatalf("stop reason = %q, want goal_stalled", res.StopReason)
	}
	if len(faux.Recorded) != 2 {
		t.Fatalf("provider calls = %d, want 2 (one nudge, then stall accept)", len(faux.Recorded))
	}
	if res.Final != "I cannot make further progress." {
		t.Fatalf("final = %q, want the repeated answer", res.Final)
	}
}

// TestGoalGateSkippedOnBudgetWrapUp verifies the budget ceiling always wins:
// a wrap-up turn ends the run even without a sentinel.
func TestGoalGateSkippedOnBudgetWrapUp(t *testing.T) {
	faux := newNativeScript(nativeAnswer("partial progress summary"))
	agent, err := agentcore.New(gated(agentcore.Config{
		NativeProvider: faux.Provider,
		Model:          "test",
		Goal:           "never satisfied",
		BudgetGate:     func(context.Context, agentcore.Usage) bool { return true },
	}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.StopReason != "budget_exhausted" {
		t.Fatalf("stop reason = %q, want budget_exhausted", res.StopReason)
	}
	if len(faux.Recorded) != 1 {
		t.Fatalf("provider calls = %d, want 1 — no goal nudge on wrap-up", len(faux.Recorded))
	}
}

// TestGoalGateBoundedByMaxTurns verifies an ever-changing sentinel-free answer
// still terminates at the MaxTurns budget.
func TestGoalGateBoundedByMaxTurns(t *testing.T) {
	faux := newNativeScript(
		nativeAnswer("attempt one"),
		nativeAnswer("attempt two"),
		nativeAnswer("attempt three"),
		nativeAnswer("attempt four"),
	)
	limits := agentcore.DefaultLimits()
	limits.MaxTurns = 3
	agent, err := agentcore.New(gated(agentcore.Config{NativeProvider: faux.Provider, Model: "test", Goal: "unreachable", Limits: &limits}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.StopReason != "max_turns" {
		t.Fatalf("stop reason = %q, want max_turns", res.StopReason)
	}
	if len(faux.Recorded) != 4 {
		t.Fatalf("provider calls = %d, want 3 work turns plus one tool-free wrap-up", len(faux.Recorded))
	}
}

// Corrective input must survive checkpoint serialization so the next run sees
// the same native transcript, including the gate's earlier nudge.
func TestGoalGateNudgePersistedDurably(t *testing.T) {
	faux := newNativeScript(nativeAnswer("not there yet"), nativeAnswer("done now.\nSTATUS: DONE"))
	agent, err := agentcore.New(gated(agentcore.Config{NativeProvider: faux.Provider, Model: "test", Goal: "finish the report"}))
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), "start")
	if err != nil {
		t.Fatal(err)
	}
	resumed := newNativeScript(nativeAnswer("still done.\nSTATUS: DONE"))
	agent, err = agentcore.New(gated(agentcore.Config{NativeProvider: resumed.Provider, Model: "test"}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RunNative(context.Background(), agentcore.NativeRun{State: result.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "continue"}}}); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, m := range resumed.Recorded[0].Messages {
		if m.Role == agentcore.RoleUser && strings.Contains(m.Content, "[goal gate]") {
			found = true
		}
	}
	if !found {
		t.Fatal("checkpoint lost the goal nudge")
	}
}

// TestGoalGateRunsBeforeFinishGuard verifies ordering: an unmet goal nudges
// without consulting the finish guard; the guard sees only sentinel-bearing
// finishes.
func TestGoalGateRunsBeforeFinishGuard(t *testing.T) {
	faux := newNativeScript(
		nativeAnswer("no sentinel yet"),
		nativeAnswer("finished.\nSTATUS: DONE"),
	)
	var guardSaw []string
	agent, err := agentcore.New(gated(agentcore.Config{
		NativeProvider: faux.Provider,
		Model:          "test",
		Goal:           "do the thing",
		Extensions:     []agentcore.ExtensionFactory{recordingStopper{seen: &guardSaw}},
	}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if len(guardSaw) != 1 || !strings.Contains(guardSaw[0], "STATUS: DONE") {
		t.Fatalf("finish guard consulted with %v, want only the sentinel-bearing final", guardSaw)
	}
}

// recordingStopper is a StopInterceptor that accepts every finish and records
// the answer it was shown. It stands in for a verify-on-stop plugin without
// this package importing one — the goal gate must run BEFORE any extension
// sees the answer, and that ordering is what this checks.
type recordingStopper struct{ seen *[]string }

func (recordingStopper) Name() string { return "recording_stopper" }

func (s recordingStopper) BeginRun(context.Context, agentcore.RunInfo) (agentcore.Extension, error) {
	return s, nil
}

func (s recordingStopper) TurnStopping(_ context.Context, info agentcore.StopInfo) agentcore.StopDecision {
	*s.seen = append(*s.seen, info.Final)
	return agentcore.StopDecision{}
}

// gated installs the goal gate for a config that declares a goal. agentcore's
// Config records the condition durably; ENFORCING it is this plugin's job, so a
// composition asks for both. preset.New wires the pair automatically.
func gated(cfg agentcore.Config) agentcore.Config {
	// PREPENDED, because stop interceptors are consulted in registration order
	// and the first to re-open the run wins: the gate must see the answer before
	// any verify-on-stop extension does. preset.New encodes the same ordering.
	//
	// Installed even when cfg.Goal is empty — a resumed run recovers its goal
	// from its native checkpoint, and an extension that was never registered cannot
	// enforce it.
	cfg.Extensions = append([]agentcore.ExtensionFactory{goal.Until(cfg.Goal)}, cfg.Extensions...)
	return cfg
}

func TestPublicTextAndStreamTailHideOnlyClosingProtocol(t *testing.T) {
	if got := goal.PublicText("answer\n**STATUS: DONE**\n"); got != "answer" {
		t.Fatal(got)
	}
	if got := goal.StripStatusLine(" words tiếp theo\nSTATUS: BLOCKED"); got != " words tiếp theo" {
		t.Fatal(got)
	}
	text := "I cannot claim STATUS: DONE yet.\nMore work remains."
	if goal.PublicText(text) != text {
		t.Fatal("ordinary prose was removed")
	}
}
