package advisor

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
)

func TestPeriodicIntervalCooldownCheckpointAndDedupe(t *testing.T) {
	calls := 0
	p := Plugin{IntervalTurns: 2, MaxPeriodicReviews: 4, CooldownTurns: 3, Reviewer: func(_ context.Context, r Review) ([]Note, error) {
		calls++
		if calls == 1 {
			return nil, errors.New("quota")
		}
		return []Note{{Text: "Verify the deployment result before advancing.", Severity: SeverityBlocker}}, nil
	}}
	ext, _ := p.BeginRun(context.Background(), agentcore.RunInfo{})
	a := ext.(*advisorRun)
	for turn := 1; turn <= 5; turn++ {
		if d := a.BeforeStep(context.Background(), agentcore.StepInfo{Turn: turn}); len(d.AdditionalContexts) != 0 {
			t.Fatal("cooldown did not suppress review")
		}
	}
	if calls != 1 {
		t.Fatal(calls)
	}
	d := a.BeforeStep(context.Background(), agentcore.StepInfo{Turn: 6})
	if len(d.AdditionalContexts) != 1 || calls != 2 {
		t.Fatalf("missing review %d", calls)
	}
	raw, _ := a.NativeState()
	restored, _ := p.BeginRun(context.Background(), agentcore.RunInfo{})
	b := restored.(*advisorRun)
	if err := b.RestoreNativeState(raw); err != nil {
		t.Fatal(err)
	}
	b.BeforeStep(context.Background(), agentcore.StepInfo{Turn: 1})
	if d := b.BeforeStep(context.Background(), agentcore.StepInfo{Turn: 3}); len(d.AdditionalContexts) != 0 || calls != 3 {
		t.Fatal("dedupe or schedule lost on resume")
	}
	for turn := 4; turn < 50; turn++ {
		b.BeforeStep(context.Background(), agentcore.StepInfo{Turn: turn})
	}
	if calls != 4 {
		t.Fatalf("unbounded periodic reviews: %d", calls)
	}
}
func TestMultiplePeriodicReviewersAndFreshEvidence(t *testing.T) {
	counts := []int{0, 0}
	p := Plugin{Reviewers: []ReviewerConfig{
		{Name: "correctness", IntervalTurns: 1, Reviewer: func(_ context.Context, r Review) ([]Note, error) {
			counts[0]++
			if len(r.Messages) == 0 || !strings.Contains(r.Messages[len(r.Messages)-1].Content, "tool evidence") {
				t.Error("review missed last completed tool")
			}
			return []Note{{Text: "Check the tool evidence.", Severity: SeverityConcern}}, nil
		}},
		{Name: "scope", IntervalTurns: 2, Reviewer: func(context.Context, Review) ([]Note, error) { counts[1]++; return nil, nil }},
	}}
	ext, err := p.BeginRun(context.Background(), agentcore.RunInfo{})
	if err != nil {
		t.Fatal(err)
	}
	g := ext.(*advisorGroup)
	g.ObserveMessages(context.Background(), agentcore.PhaseRequest, 1, []agentcore.Message{{Role: agentcore.RoleUser, Content: "work"}})
	g.ObserveMessages(context.Background(), agentcore.PhaseAppend, 1, []agentcore.Message{{Role: agentcore.RoleTool, Content: "tool evidence"}})
	g.BeforeStep(context.Background(), agentcore.StepInfo{Turn: 2})
	g.BeforeStep(context.Background(), agentcore.StepInfo{Turn: 3})
	if counts[0] != 2 || counts[1] != 1 {
		t.Fatal(counts)
	}
	raw, _ := g.NativeState()
	next, _ := p.BeginRun(context.Background(), agentcore.RunInfo{})
	if err := next.(agentcore.NativeStateContributor).RestoreNativeState(raw); err != nil {
		t.Fatal(err)
	}
}
