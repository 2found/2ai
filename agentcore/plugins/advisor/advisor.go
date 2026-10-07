// Package advisor reviews completed work and, optionally, work in progress at
// turn boundaries. Reviewer callbacks contribute bounded, deduplicated notes;
// material concerns reopen a finish or enter the next request. Each reviewer
// owns independent review caps, cooldown and checkpointed delivery history.
// Native bindings account AI usage to the owning run. No background scheduler
// or in-flight turn interruption is introduced.
package advisor

import (
	"context"
	"time"

	"github.com/2found/2ai/agentcore"
)

// DefaultMaxRounds caps reviewer consultations per run. Two is the same
// allowance finishguard gives, for the same reason: a reviewer that is never
// satisfied must not be able to loop the run against MaxTurns.
const DefaultMaxRounds = 2

// DefaultMaxNotesPerReview bounds how many notes one review may deliver. It is
// breadth control, not rate limiting — a reviewer handed a whole run will
// happily list nine things, and an injection that long stops being advice and
// becomes a second task.
const DefaultMaxNotesPerReview = 3

// Severity is how strongly one note should be weighed. It also decides
// delivery: a nit is recorded and the finish is accepted, while a concern or a
// blocker re-opens the run.
type Severity string

const (
	// SeverityNit is cleanup, simplification, a low-risk edge case. It never
	// re-opens the run: at a finish there is no next step boundary for an aside
	// to ride, so spending a whole turn on a nit costs more than the nit is
	// worth. It is reported to the host instead.
	SeverityNit Severity = "nit"
	// SeverityConcern is material risk: a likely-wrong direction, a missing
	// constraint, a figure that does not follow from the evidence. Re-opens the
	// run so the agent can weigh it.
	SeverityConcern Severity = "concern"
	// SeverityBlocker is work that would be wrong to hand over: a claim of
	// completion over sampled scope, an answer never exercised against what was
	// asked. Re-opens the run.
	SeverityBlocker Severity = "blocker"
)

// Interrupting reports whether a note at this severity re-opens the run.
func (s Severity) Interrupting() bool {
	return s == SeverityConcern || s == SeverityBlocker
}

// Note is one piece of advice from the reviewer.
type Note struct {
	// Text is the advice itself: concrete, terse, actionable.
	Text string `json:"text"`
	// Severity decides delivery. An unrecognized or empty value is treated as
	// a nit, so a reviewer that omits the field cannot accidentally interrupt.
	Severity Severity `json:"severity,omitempty"`
}

// Review is the evidence handed to a Reviewer.
type Review struct {
	// Final is the answer the run would return.
	Final string
	// Turns is the number of reasoning turns consumed, including this one.
	Turns int
	// Tools is the run's tool trace — what was called, what was blocked, what
	// errored. Read-only; it aliases the run's live slice.
	Tools []agentcore.ToolTrace
	// Messages is the conversation the final answer came out of, captured at
	// the last provider request. It does NOT include the final assistant
	// message itself, which is Final.
	Messages []agentcore.Message
	// Round is 0 on the first review of a run and increments per re-opening,
	// so a reviewer can tell "look at this work" from "look at whether the
	// agent dealt with what you already said".
	Round int
	// Delivered is every note this run has already put in front of the agent,
	// oldest first. On a second round it is what the reviewer checks the new
	// answer against; repeating one of these verbatim is dropped by the
	// emission guard, so a reviewer that still objects must escalate.
	Delivered []Note
}

// Reviewer reads a finished run and returns the notes worth making. Returning
// no notes accepts the finish, and silence is the expected outcome of a run
// that went fine.
//
// An error accepts the finish too. A reviewer is a second opinion, not a
// dependency: a provider outage, a timeout, or an unparseable response must
// leave the agent's answer exactly as it would have been with no advisor
// configured, never wedge or fail the run.
type Reviewer func(ctx context.Context, r Review) ([]Note, error)

// Plugin installs the advisor. A nil Reviewer registers nothing, so a
// composition that wires the plugin for a disabled agent is inert rather than
// broken — which is what makes "advisor: off" a config value rather than a
// different composition.
type Plugin struct {
	// Reviewer is consulted at each normal finish.
	Reviewer Reviewer
	// Reviewers adds independently scheduled reviewers (at most four).
	Reviewers []ReviewerConfig
	// IntervalTurns enables periodic review after this many completed turns.
	IntervalTurns      int
	MaxPeriodicReviews int
	CooldownTurns      int
	Timeout            time.Duration
	// MaxRounds bounds consultations per run. 0 uses DefaultMaxRounds. The cap
	// lives here rather than in the loop because it is this capability's
	// property: the loop only knows that SOMETHING asked to continue.
	MaxRounds int
	// MaxNotesPerReview bounds notes delivered per review. 0 uses
	// DefaultMaxNotesPerReview.
	MaxNotesPerReview int
	// OnNotes, when set, receives every note the guard accepted — including
	// the nits the agent never sees — so the host can record what the reviewer
	// said. delivered reports whether these notes were actually put in front of
	// the agent: a review is injected whole or not at all, so a nit riding
	// alongside a blocker IS delivered, and severity alone cannot tell you that.
	OnNotes func(ctx context.Context, notes []Note, delivered bool)
}

// Of wraps a reviewer with the default bounds.
func Of(r Reviewer) Plugin { return Plugin{Reviewer: r} }

// Name identifies the plugin and the extension it installs.
func (Plugin) Name() string { return "advisor" }

// Register adds the plugin as a run extension. A nil reviewer declines.
func (p Plugin) Register(r *agentcore.Registry) error {
	if p.Reviewer == nil && len(p.Reviewers) == 0 {
		return nil
	}
	r.AddExtension(p)
	return nil
}

// BeginRun starts one run's advisor state: its round budget, its emission
// guard, and the message snapshot the reviewer will read.
func (p Plugin) BeginRun(ctx context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	if len(p.Reviewers) > 0 {
		return p.beginGroup(ctx, info)
	}
	if p.Reviewer == nil {
		return nil, nil
	}
	rounds := p.MaxRounds
	if rounds <= 0 {
		rounds = DefaultMaxRounds
	}
	perReview := p.MaxNotesPerReview
	if perReview <= 0 {
		perReview = DefaultMaxNotesPerReview
	}
	return &advisorRun{
		reviewer:    p.Reviewer,
		interval:    p.IntervalTurns,
		maxPeriodic: defaultPositive(p.MaxPeriodicReviews, 8),
		cooldown:    defaultPositive(p.CooldownTurns, 3),
		timeout:     p.Timeout,
		rounds:      rounds,
		onNotes:     p.OnNotes,
		guard:       NewEmissionGuard(perReview),
	}, nil
}

// advisorRun is one run's advisor state. Per-run state is why this is a
// factory rather than a bare function: the dedupe history and the delivered
// list are what make a second round "did you deal with it" rather than a
// repeat of the first.
type advisorRun struct {
	interval, maxPeriodic, cooldown                                int
	timeout                                                        time.Duration
	periodic, finished, ticks, lastTurn, lastReview, cooldownUntil int
	reviewer                                                       Reviewer
	rounds                                                         int
	onNotes                                                        func(context.Context, []Note, bool)
	guard                                                          *EmissionGuard
	messages                                                       []agentcore.Message
	delivered                                                      []Note
}

// Name identifies the extension in composition diagnostics.
func (*advisorRun) Name() string { return "advisor" }

// ObserveMessages keeps the conversation the reviewer will read.
//
// This is why the plugin needs no kernel change: StopInfo carries the answer
// and the tool trace but not the history, and the history is most of what
// there is to review. PhaseRequest is the snapshot the model was actually
// asked to answer from, so at a finish the last one held is exactly the
// context that produced the answer.
//
// A rebase (compaction, a context edit) replaces the history the earlier notes
// were about, so the dedupe history is dropped with it — a reviewer looking at
// a rewritten transcript must be free to re-raise what it raised before.
func (a *advisorRun) ObserveMessages(_ context.Context, phase agentcore.ObservePhase, turn int, msgs []agentcore.Message) {
	switch phase {
	case agentcore.PhaseExternalInput:
		if turn == 0 && len(msgs) > 0 && a.finished > 0 {
			a.periodic = 0
			a.finished = 0
			a.ticks = 0
			a.lastTurn = 0
			a.lastReview = 0
			a.cooldownUntil = 0
			a.delivered = nil
			a.guard.Reset()
		}
	case agentcore.PhaseAppend:
		a.messages = boundedMessages(append(a.messages, msgs...))
	case agentcore.PhaseRequest:
		a.messages = boundedMessages(msgs)
	case agentcore.PhaseRebase:
		a.messages = boundedMessages(msgs)
		a.guard.Reset()
	}
}

// TurnStopping consults the reviewer on a finish the run chose.
//
// The cap is enforced against info.Attempt — a count the loop keeps, so a
// reviewer that never runs out of objections is still bounded. A reviewer
// error, like a panicking guard, accepts the finish: the advisor is a second
// opinion and must never be the reason a run cannot end.
func (a *advisorRun) TurnStopping(ctx context.Context, info agentcore.StopInfo) agentcore.StopDecision {
	if info.Attempt >= a.rounds || a.finished >= a.rounds {
		return agentcore.StopDecision{}
	}
	a.finished++
	return a.review(ctx, Review{
		Final:     info.Final,
		Turns:     info.Turns,
		Tools:     info.Tools,
		Messages:  a.messages,
		Round:     a.finished - 1,
		Delivered: a.delivered,
	})
}

func (a *advisorRun) review(ctx context.Context, review Review) agentcore.StopDecision {
	a.guard.BeginReview()
	timeout := a.timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	review.Final = agentcore.TruncateMiddle(review.Final, 8000)
	review.Tools = boundedTools(review.Tools)
	review.Messages = boundedMessages(review.Messages)
	review.Delivered = append([]Note(nil), review.Delivered...)
	raw, err := a.reviewer(ctx, review)
	if err != nil {
		a.cooldownUntil = a.ticks + a.cooldown
	}
	if err != nil || len(raw) == 0 {
		return agentcore.StopDecision{}
	}

	var accepted []Note
	interrupting := false
	for _, n := range raw {
		note, ok := a.guard.Accept(n)
		if !ok {
			continue
		}
		accepted = append(accepted, note)
		if note.Severity.Interrupting() {
			interrupting = true
		}
	}
	if len(accepted) == 0 {
		return agentcore.StopDecision{}
	}
	// interrupting decides the whole review, not each note: the injection below
	// carries every accepted note, so a nit beside a blocker reaches the agent
	// too. Report that, rather than letting the host infer delivery from
	// severity and record a note the agent read as one it never saw.
	if a.onNotes != nil {
		a.onNotes(ctx, accepted, interrupting)
	}
	// Nits alone do not buy a turn. They are already recorded through OnNotes,
	// which is the whole delivery a nit gets.
	if !interrupting {
		return agentcore.StopDecision{}
	}
	a.delivered = append(a.delivered, accepted...)
	if len(a.delivered) > 32 {
		a.delivered = append([]Note(nil), a.delivered[len(a.delivered)-32:]...)
	}
	return agentcore.StopDecision{
		Continue: true,
		Inject:   []agentcore.Message{{Role: agentcore.RoleUser, Content: FormatInjection(accepted)}},
		Note:     "reviewing the answer with the advisor",
	}
}
