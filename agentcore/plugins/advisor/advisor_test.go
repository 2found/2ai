package advisor_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/advisor"
	"github.com/2found/2ai/ai"
)

type nativeRequests struct{ Recorded []ai.TranscriptContext }

func nativeAnswer(text string) ai.Message {
	return ai.Message{Role: "assistant", Model: "test", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: text})}
}

func provider(stream ai.StreamFn) *ai.FallbackProvider {
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: stream}}}
}

// build exercises review policy through the real native engine and records
// the final provider view, including any corrective host input.
func build(t *testing.T, p advisor.Plugin, replies ...ai.Message) (*agentcore.Agent, *nativeRequests) {
	t.Helper()
	recorded := &nativeRequests{}
	script := ai.ScriptedStream(replies...)
	stream := func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		recorded.Recorded = append(recorded.Recorded, view)
		return script(ctx, model, view, options)
	}
	agent, err := agentcore.New(agentcore.Config{NativeProvider: provider(stream), Model: "test", Extensions: []agentcore.ExtensionFactory{p}})
	if err != nil {
		t.Fatal(err)
	}
	return agent, recorded
}

func TestConcernReopensRunAndAgentResolvesIt(t *testing.T) {
	agent, faux := build(t, advisor.Plugin{
		Reviewer: func(_ context.Context, r advisor.Review) ([]advisor.Note, error) {
			if r.Round == 0 {
				return []advisor.Note{{Text: "the revenue figure sums a filtered and an unfiltered query", Severity: advisor.SeverityConcern}}, nil
			}
			return nil, nil
		},
	},
		nativeAnswer("revenue was 4.2M"),
		nativeAnswer("revenue was 3.1M (corrected: the two queries used different filters)"),
	)

	res, err := agent.Prompt(context.Background(), "what was revenue")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if !strings.Contains(res.Final, "3.1M") {
		t.Fatalf("final = %q, want the answer the agent gave after resolving the note", res.Final)
	}
	if len(faux.Recorded) != 2 {
		t.Fatalf("provider calls = %d, want 2 (the answer, then the resolution turn)", len(faux.Recorded))
	}
	var injected string
	for _, m := range faux.Recorded[1].Messages() {
		if m.Role == "user" && m.Content.Text != nil && strings.Contains(*m.Content.Text, "<advisory") {
			injected = *m.Content.Text
		}
	}
	if injected == "" {
		t.Fatalf("no advisory reached the reopened turn: %+v", faux.Recorded[1].Messages())
	}
	for _, want := range []string{
		`severity="concern"`,
		"weigh, don&#39;t blindly obey",
		"filtered and an unfiltered query",
		"COMPLETE answer",
	} {
		if !strings.Contains(injected, want) {
			t.Errorf("injection missing %q:\n%s", want, injected)
		}
	}
}

func TestNitDoesNotReopenTheRunButIsReported(t *testing.T) {
	var reported []advisor.Note
	agent, faux := build(t, advisor.Plugin{
		Reviewer: func(context.Context, advisor.Review) ([]advisor.Note, error) {
			return []advisor.Note{{Text: "the second query could reuse the first CTE", Severity: advisor.SeverityNit}}, nil
		},
		OnNotes: func(_ context.Context, n []advisor.Note, _ bool) { reported = append(reported, n...) },
	}, nativeAnswer("done"))

	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "done" {
		t.Fatalf("final = %q, want the first answer — a nit must not buy a turn", res.Final)
	}
	if len(faux.Recorded) != 1 {
		t.Fatalf("provider calls = %d, want 1", len(faux.Recorded))
	}
	if len(reported) != 1 || reported[0].Severity != advisor.SeverityNit {
		t.Fatalf("OnNotes got %+v, want the one nit", reported)
	}
}

// A review is injected whole. So the delivered flag OnNotes reports is a
// property of the review, not of each note's severity — and a host that infers
// it from severity writes down that the agent never saw a note it read.
func TestANitRidingWithABlockerIsReportedAsDelivered(t *testing.T) {
	type record struct {
		notes     []advisor.Note
		delivered bool
	}
	var got []record
	round := 0
	agent, _ := build(t, advisor.Plugin{
		Reviewer: func(context.Context, advisor.Review) ([]advisor.Note, error) {
			round++
			if round > 1 {
				return nil, nil
			}
			return []advisor.Note{
				{Text: "the total double-counts refunded orders", Severity: advisor.SeverityBlocker},
				{Text: "the second query could reuse the first CTE", Severity: advisor.SeverityNit},
			}, nil
		},
		OnNotes: func(_ context.Context, n []advisor.Note, delivered bool) {
			got = append(got, record{notes: n, delivered: delivered})
		},
	}, nativeAnswer("first"), nativeAnswer("second"))

	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("OnNotes called %d times, want 1", len(got))
	}
	if !got[0].delivered {
		t.Error("a review containing a blocker re-opened the run, so every note in it reached the agent — delivered must say so")
	}
	if len(got[0].notes) != 2 {
		t.Fatalf("OnNotes got %d notes, want both — the nit rode along in the same injection", len(got[0].notes))
	}
}

func TestSilentReviewerAcceptsTheFinish(t *testing.T) {
	calls := 0
	agent, faux := build(t, advisor.Of(func(context.Context, advisor.Review) ([]advisor.Note, error) {
		calls++
		return nil, nil
	}), nativeAnswer("answer"))

	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "answer" || len(faux.Recorded) != 1 {
		t.Fatalf("final = %q over %d calls, want the untouched answer in one call", res.Final, len(faux.Recorded))
	}
	if calls != 1 {
		t.Fatalf("reviewer consulted %d times, want 1", calls)
	}
}

func TestReviewerErrorAcceptsTheFinish(t *testing.T) {
	agent, faux := build(t, advisor.Of(func(context.Context, advisor.Review) ([]advisor.Note, error) {
		return []advisor.Note{{Text: "ignored", Severity: advisor.SeverityBlocker}}, errors.New("provider down")
	}), nativeAnswer("answer"))

	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "answer" {
		t.Fatalf("final = %q — a failed reviewer must leave the run exactly as it was", res.Final)
	}
	if len(faux.Recorded) != 1 {
		t.Fatalf("provider calls = %d, want 1", len(faux.Recorded))
	}
}

func TestRoundsAreCapped(t *testing.T) {
	calls := 0
	agent, _ := build(t, advisor.Plugin{
		Reviewer: func(_ context.Context, r advisor.Review) ([]advisor.Note, error) {
			calls++
			// A new objection every round, so only the cap can stop this.
			return []advisor.Note{{Text: "still wrong, round " + string(rune('a'+r.Round)), Severity: advisor.SeverityBlocker}}, nil
		},
	},
		nativeAnswer("try 1"),
		nativeAnswer("try 2"),
		nativeAnswer("try 3"),
	)

	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if calls != advisor.DefaultMaxRounds {
		t.Fatalf("reviewer consulted %d times, want the cap of %d", calls, advisor.DefaultMaxRounds)
	}
	if res.Final != "try 3" {
		t.Fatalf("final = %q, want the answer after the capped rounds", res.Final)
	}
	if res.StopReason != "stop" {
		t.Fatalf("stop reason = %q — an exhausted advisor must not change how the run ended", res.StopReason)
	}
}

func TestRepeatedNoteIsDroppedSoARoundCannotSpin(t *testing.T) {
	agent, faux := build(t, advisor.Plugin{
		Reviewer: func(context.Context, advisor.Review) ([]advisor.Note, error) {
			// The same objection, worded identically, every round.
			return []advisor.Note{{Text: "verify the totals", Severity: advisor.SeverityConcern}}, nil
		},
	},
		nativeAnswer("try 1"),
		nativeAnswer("try 2"),
	)

	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "try 2" {
		t.Fatalf("final = %q, want the second answer", res.Final)
	}
	// Round 0 delivers it; round 1 re-raises the same text at the same severity
	// and the guard drops it, so the run ends without a third turn.
	if len(faux.Recorded) != 2 {
		t.Fatalf("provider calls = %d, want 2 — a verbatim repeat must not buy another turn", len(faux.Recorded))
	}
}

func TestEscalationGetsASecondTurn(t *testing.T) {
	agent, faux := build(t, advisor.Plugin{
		Reviewer: func(_ context.Context, r advisor.Review) ([]advisor.Note, error) {
			sev := advisor.SeverityConcern
			if r.Round > 0 {
				sev = advisor.SeverityBlocker
			}
			return []advisor.Note{{Text: "the totals do not add up", Severity: sev}}, nil
		},
	},
		nativeAnswer("try 1"),
		nativeAnswer("try 2"),
		nativeAnswer("try 3"),
	)

	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if len(faux.Recorded) != 3 {
		t.Fatalf("provider calls = %d, want 3 — an escalation from concern to blocker is not a repeat", len(faux.Recorded))
	}
}

func TestReviewerSeesTheConversationAndPriorNotes(t *testing.T) {
	var round1 advisor.Review
	agent, _ := build(t, advisor.Plugin{
		Reviewer: func(_ context.Context, r advisor.Review) ([]advisor.Note, error) {
			if r.Round == 0 {
				return []advisor.Note{{Text: "check the join", Severity: advisor.SeverityConcern}}, nil
			}
			round1 = r
			return nil, nil
		},
	},
		nativeAnswer("first"),
		nativeAnswer("second"),
	)

	if _, err := agent.Prompt(context.Background(), "the task"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if round1.Round != 1 {
		t.Fatalf("second consultation saw Round=%d, want 1", round1.Round)
	}
	if len(round1.Delivered) != 1 || round1.Delivered[0].Text != "check the join" {
		t.Fatalf("Delivered = %+v, want the note from round 0", round1.Delivered)
	}
	if round1.Final != "second" {
		t.Fatalf("Final = %q, want the answer produced after the note", round1.Final)
	}
	var sawTask bool
	for _, m := range round1.Messages {
		if m.Role == agentcore.RoleUser && strings.Contains(m.Content, "the task") {
			sawTask = true
		}
	}
	if !sawTask {
		t.Fatalf("reviewer never saw the conversation: %+v", round1.Messages)
	}
}

func TestNilReviewerIsInert(t *testing.T) {
	agent, faux := build(t, advisor.Plugin{}, nativeAnswer("answer"))
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "answer" || len(faux.Recorded) != 1 {
		t.Fatalf("a disabled advisor changed the run: %q over %d calls", res.Final, len(faux.Recorded))
	}
}

func TestBudgetWrapUpIsNotReviewed(t *testing.T) {
	consulted := false
	stream := ai.ScriptedStream(nativeAnswer("wrap-up"))
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: provider(stream),
		Model:          "test",
		BudgetGate:     func(context.Context, agentcore.Usage) bool { return true },
		Extensions: []agentcore.ExtensionFactory{advisor.Of(func(context.Context, advisor.Review) ([]advisor.Note, error) {
			consulted = true
			return []advisor.Note{{Text: "reopen", Severity: advisor.SeverityBlocker}}, nil
		})},
	})
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
	if consulted {
		t.Fatal("the advisor must not be consulted on a stop the run did not choose")
	}
}
