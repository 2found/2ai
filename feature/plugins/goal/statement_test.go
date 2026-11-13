package goal_test

import (
	"context"
	"testing"

	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/feature/plugins/goal"
)

// The condition is stated by appending a system message, and only when the
// statement the model can see as in force is not the current condition.
func TestGoalIsStatedOnlyWhenNotInForce(t *testing.T) {
	ctx := context.Background()
	const a, b = "all tests pass", "all tests pass and the flaky one is quarantined"
	stmt := func(g string) string { return goal.StatementPrefix + " This run finishes when:\n" + g }
	stated := func(g string) agentcore.Message {
		return agentcore.Message{Role: agentcore.RoleSystem, Content: stmt(g)}
	}
	st := goal.NewStore(a)
	ext, err := goal.Plugin{Store: st, Revisable: true}.BeginRun(ctx, agentcore.RunInfo{Goal: a})
	if err != nil {
		t.Fatal(err)
	}
	step, observer := ext.(agentcore.StepInterceptor), ext.(agentcore.RunObserver)
	view := func(phase agentcore.ObservePhase, msgs ...agentcore.Message) {
		observer.ObserveMessages(ctx, phase, 1, msgs)
	}
	next := func() string {
		got := step.BeforeStep(ctx, agentcore.StepInfo{}).AdditionalContexts
		switch len(got) {
		case 0:
			return ""
		case 1:
			// System role: compaction keeps system messages verbatim, and a late
			// one is rendered in place, never folded into the cached head.
			if got[0].Role != agentcore.RoleSystem {
				t.Fatalf("statement role = %s, want system", got[0].Role)
			}
			return got[0].Content
		}
		t.Fatalf("stated %d messages", len(got))
		return ""
	}
	// The protocol names the marker; it is not a statement.
	system := agentcore.Message{Role: agentcore.RoleSystem, Content: ext.(agentcore.PromptContributor).SystemPrompt()}

	view(agentcore.PhaseRestore, system)
	if got := next(); got != stmt(a) {
		t.Fatalf("first step stated %q", got)
	}
	if got := next(); got != "" {
		t.Fatalf("statement repeated before a new view: %q", got)
	}
	view(agentcore.PhaseRequest, system, agentcore.Message{Role: agentcore.RoleUser, Content: "go"}, stated(a))
	if got := next(); got != "" {
		t.Fatalf("visible condition restated: %q", got)
	}

	// A revision is stated by the next step, even though the update_goal
	// result already told the model: the result is not a durable statement.
	if !st.Update(b, "the flaky test is not ours") {
		t.Fatal("revision rejected")
	}
	view(agentcore.PhaseRequest, system, stated(a), agentcore.Message{Role: agentcore.RoleTool, Content: "Goal updated. The run now finishes when:\n" + b})
	if got := next(); got != stmt(b) {
		t.Fatalf("revised condition stated as %q", got)
	}
	if got := next(); got != "" {
		t.Fatalf("revision stated twice: %q", got)
	}

	// Revising back: a's earlier statement is superseded by b's, so a is
	// stated again. Quoting a statement (user, tool, model) is not stating it.
	if !st.Update(a, "the quarantine was rejected") {
		t.Fatal("revision rejected")
	}
	view(agentcore.PhaseRequest, system, stated(a), stated(b),
		agentcore.Message{Role: agentcore.RoleUser, Content: stmt(a)},
		agentcore.Message{Role: agentcore.RoleTool, Content: stmt(a)},
		agentcore.Message{Role: agentcore.RoleAssistant, Content: stmt(a)},
	)
	if got := next(); got != stmt(a) {
		t.Fatalf("superseded or quoted statement counted as in force: %q", got)
	}

	// A condition that is a prefix of a longer stated one is not in force.
	view(agentcore.PhaseRequest, system, stated(b))
	if got := next(); got != stmt(a) {
		t.Fatalf("prefix of a longer statement counted as in force: %q", got)
	}

	// Compaction keeps system messages, so a statement survives a summary...
	view(agentcore.PhaseRebase, system, stated(a), agentcore.Message{Role: agentcore.RoleUser, Content: "[Earlier work summary]"})
	if got := next(); got != "" {
		t.Fatalf("condition kept by compaction restated: %q", got)
	}
	// ...and a view that lost it anyway (a host that drops them) restates it.
	view(agentcore.PhaseRebase, system, agentcore.Message{Role: agentcore.RoleUser, Content: "[Earlier work summary]"})
	if got := next(); got != stmt(a) {
		t.Fatalf("lost condition not restated: %q", got)
	}
}
