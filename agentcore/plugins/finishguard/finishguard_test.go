package finishguard_test

import (
	"context"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/finishguard"
)

// TestFinishGuardNudgeReopensRun verifies a guard rejecting the first finish
// injects its nudge as a user message, the loop continues, and the guard's
// evidence snapshot carries the answer that triggered it.
func TestFinishGuardNudgeReopensRun(t *testing.T) {
	faux := newNativeScript(
		nativeAnswer("unverified claim"),
		nativeAnswer("verified answer"),
	)
	var states []finishguard.State
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: faux.Provider,
		Model:          "test",
		Extensions: []agentcore.ExtensionFactory{finishguard.Of(func(_ context.Context, s finishguard.State) string {
			states = append(states, s)
			if s.Nudges == 0 {
				return "NUDGE: verify before finishing"
			}
			return ""
		})},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "start")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "verified answer" {
		t.Fatalf("final = %q, want the post-nudge answer", res.Final)
	}
	if len(states) != 2 {
		t.Fatalf("guard consulted %d times, want 2", len(states))
	}
	if states[0].Final != "unverified claim" || states[0].Nudges != 0 {
		t.Fatalf("first consult state wrong: %+v", states[0])
	}
	if states[1].Nudges != 1 {
		t.Fatalf("second consult should see Nudges=1: %+v", states[1])
	}
	// The nudge must appear in the second provider request as a user message.
	if len(faux.Recorded) != 2 {
		t.Fatalf("expected 2 provider calls, got %d", len(faux.Recorded))
	}
	var seen bool
	for _, m := range faux.Recorded[1].Messages {
		if m.Role == agentcore.RoleUser && strings.Contains(m.Content, "NUDGE: verify before finishing") {
			seen = true
		}
	}
	if !seen {
		t.Fatalf("nudge not present in reopened request: %+v", faux.Recorded[1].Messages)
	}
}

// TestFinishGuardCapped verifies an always-rejecting guard is cut off at
// finishguard.DefaultMaxNudges so the run still terminates with the model's answer.
func TestFinishGuardCapped(t *testing.T) {
	faux := newNativeScript(
		nativeAnswer("try 1"),
		nativeAnswer("try 2"),
		nativeAnswer("try 3"),
	)
	consults := 0
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: faux.Provider,
		Model:          "test",
		Extensions: []agentcore.ExtensionFactory{finishguard.Of(func(context.Context, finishguard.State) string {
			consults++
			return "never satisfied"
		})},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "start")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if consults != finishguard.DefaultMaxNudges {
		t.Fatalf("guard consulted %d times, want %d (cap)", consults, finishguard.DefaultMaxNudges)
	}
	if res.Final != "try 3" {
		t.Fatalf("final = %q, want the answer after the capped nudges", res.Final)
	}
	if res.StopReason != "stop" {
		t.Fatalf("stop reason = %q, want normal stop", res.StopReason)
	}
}

// TestFinishGuardSkippedOnBudgetWrapUp verifies the guard never fires on a
// budget-exhausted wrap-up turn: the ceiling is hit, so nothing may re-open the
// run.
func TestFinishGuardSkippedOnBudgetWrapUp(t *testing.T) {
	faux := newNativeScript(nativeAnswer("wrap-up summary"))
	consulted := false
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: faux.Provider,
		Model:          "test",
		BudgetGate:     func(context.Context, agentcore.Usage) bool { return true },
		Extensions: []agentcore.ExtensionFactory{
			finishguard.Of(func(context.Context, finishguard.State) string { consulted = true; return "reopen" }),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "start")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.StopReason != "budget_exhausted" {
		t.Fatalf("stop reason = %q, want budget_exhausted", res.StopReason)
	}
	if consulted {
		t.Fatal("finish guard must not be consulted on a budget wrap-up turn")
	}
}

// TestFinishGuardPanicAcceptsFinish verifies a panicking guard is hardened: the
// finish is accepted rather than crashing the run goroutine.
func TestFinishGuardPanicAcceptsFinish(t *testing.T) {
	faux := newNativeScript(nativeAnswer("answer"))
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: faux.Provider,
		Model:          "test",
		Extensions: []agentcore.ExtensionFactory{
			finishguard.Of(func(context.Context, finishguard.State) string { panic("reckless guard") }),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "start")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "answer" {
		t.Fatalf("final = %q, want the model's answer despite guard panic", res.Final)
	}
}

// TestFinishGuardNudgePersistedDurably verifies the injected nudge lands in the
// native checkpoint, so a resume presents the same corrective input.
func TestFinishGuardNudgePersistedDurably(t *testing.T) {
	faux := newNativeScript(
		nativeAnswer("first"),
		nativeAnswer("second"),
	)
	nudged := false
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: faux.Provider,
		Model:          "test",
		Extensions: []agentcore.ExtensionFactory{finishguard.Of(func(_ context.Context, s finishguard.State) string {
			if nudged {
				return ""
			}
			nudged = true
			return "GUARD-NUDGE"
		})},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	result, err := agent.Prompt(context.Background(), "start")
	if err != nil {
		t.Fatal(err)
	}
	resumed := newNativeScript(nativeAnswer("continued"))
	agent, err = agentcore.New(agentcore.Config{NativeProvider: resumed.Provider, Model: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RunNative(context.Background(), agentcore.NativeRun{State: result.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "continue"}}}); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, m := range resumed.Recorded[0].Messages {
		if m.Role == agentcore.RoleUser && m.Content == "GUARD-NUDGE" {
			found = true
		}
	}
	if !found {
		t.Fatal("native checkpoint lost guard nudge")
	}
}
