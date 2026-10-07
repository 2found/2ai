// Package goal installs the run-level completion contract: a goal-gated run may
// only stop when it says so.
//
// The condition is stated in the system prompt up front, and the model must end
// its final answer with STATUS: DONE once the goal is met, or STATUS: BLOCKED
// plus the reason when only the user can unblock it. A finish with neither
// sentinel re-opens the run with a keep-going nudge.
//
// Unlike the finishguard plugin (a capped verify nudge), the gate is UNCAPPED by
// design: it holds the line until the model declares done or blocked. It cannot
// wedge a run — MaxTurns, MaxToolCalls, and the budget gate still bound the
// loop, a budget wrap-up never consults a stop interceptor at all, and the two
// stall breakers below stop a model that has nothing new to say.
//
// # Stalling, and why one breaker is not enough
//
// Being uncapped is only affordable if "the model has nothing left to give" is
// detected cheaply. The original breaker compared the finish to the previous one
// verbatim, which is a good proxy for a strong model — asked the same question
// against a near-identical context, it tends to emit the same tokens — and a
// useless one for a weaker model, which paraphrases instead. Measured: a model
// that answers every nudge with the same claim in different words trips the
// verbatim breaker never, and burns the entire turn budget re-asserting that it
// is finished. So the second breaker asks the behavioral question rather than
// the lexical one: did the model DO anything between one nudge and the next? A
// run of finishes with no tool call in between is a stall whatever the wording,
// and unlike text comparison it does not care how the model phrases itself.
//
// The nudge escalates for the same reason. A model that ignored an instruction
// is unlikely to obey a byte-identical repeat of it, so the second and later
// nudges state the contract far more mechanically than the first.
//
// The goal STRING is durable state the loop owns: it writes the EntryGoal record
// and recovers it on resume, then hands it to this plugin as RunInfo.Goal. So a
// crashed run comes back gated even when the resuming caller could not re-supply
// the condition, without this package ever writing to the log.
//
// # Composition order
//
// The loop consults stop interceptors in registration order and the first one to
// continue wins, so this plugin must be registered BEFORE finishguard: an unmet
// goal makes any verify pass on that answer moot.
package goal

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/2found/2ai/agentcore"
)

// Done and Blocked are the completion sentinels the gate looks for in the final
// answer's closing line (case-insensitive, so "Status: done" also counts).
const (
	Done    = "STATUS: DONE"
	Blocked = "STATUS: BLOCKED"
)

// StopReasonStalled is recorded when a stall breaker accepts a finish because
// the model has stopped making progress toward the goal — it either repeated its
// previous answer verbatim, or kept declaring itself finished without doing any
// work between one nudge and the next.
const StopReasonStalled = "goal_stalled"

// maxIdleNudges is how many consecutive finishes with NO tool call in between
// the gate will nudge before calling the run stalled.
//
// It is a small number on purpose. Every one of these is a wasted model call:
// the run has already been told, in the system prompt and then in a nudge, how
// to declare itself done. Three attempts is enough slack for a model that
// misreads the contract once and corrects, and cheap enough that the failure
// costs a handful of calls rather than the whole turn budget. A model that IS
// working keeps calling tools and is never counted here, so this bounds only the
// case where nudging has demonstrably stopped buying anything.
const maxIdleNudges = 3

// Plugin installs the goal gate. An empty Goal leaves the run ungated, so a
// composition can wire the plugin unconditionally and let configuration decide.
type Plugin struct {
	Goal string
	// Lifecycle enables checkpointed pause/resume, accounting and host controls.
	Lifecycle      bool
	NewTaskOnInput bool
	TokenBudget    int
	TimeBudget     time.Duration
	// Revisable offers the model the update_goal tool, letting it restate what
	// finishing means when the work shows the condition to be wrong.
	//
	// Off by default, and that default is the honest one: it hands the model the
	// key to its own gate, so a composition takes it deliberately rather than
	// inheriting it. See updateGoalTool for what the trade actually is.
	Revisable bool
	// Store, when set, is the condition's live home — share it to read the
	// revision trail after the run. Left nil, the plugin builds its own per run.
	Store *Store
}

// Until gates the run on a condition.
func Until(condition string) Plugin { return Plugin{Goal: condition} }

// UntilRevisable gates the run on a condition the AGENT may restate, with the
// reason recorded, when the work shows that condition to be wrong. The returned
// Store is the audit trail: Revisions() holds every condition the run has held,
// in order, each with the justification the model gave for changing it.
func UntilRevisable(condition string) (Plugin, *Store) {
	st := NewStore(condition)
	return Plugin{Goal: condition, Revisable: true, Store: st}, st
}

// Name identifies the plugin and the extension it installs.
func (Plugin) Name() string { return "goal" }

// Register claims the goal seam (the loop persists the condition) and adds the
// gate itself as a run extension. Both halves are one call because a recorded
// goal nobody enforces, or an enforced goal nobody recorded, is a bug either
// way — the first resumes ungated, the second gates a run whose condition is
// invisible to the log.
func (p Plugin) Register(r *agentcore.Registry) error {
	if p.Goal != "" {
		if err := r.SetGoal(p.Goal); err != nil {
			return err
		}
	}
	// The extension is added even with no configured goal, because a RESUMED
	// run gets its condition from the durable log rather than from this field —
	// the caller that restarts a crashed run usually cannot re-supply it. An
	// extension installed for a run that turns out to be ungated declines at
	// BeginRun and costs nothing; one that was never installed cannot gate a
	// recovered goal at all, which is the case gating matters most.
	r.AddExtension(p)
	return nil
}

// BeginRun arms the gate for this run. It declines an ungated run — the run has
// no condition to hold it to, so there is nothing to consult and no contract to
// put in the prompt.
//
// The condition comes from RunInfo, not from p.Goal: on a resume the loop
// recovers it from the durable log, and a plugin that trusted its own field
// would come back ungated exactly when gating matters most.
func (p Plugin) BeginRun(_ context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	if info.Goal == "" {
		return nil, nil
	}
	// The store is seeded from RunInfo, not from p.Goal, for the same reason the
	// gate is: on a resume the loop has already folded the log's revisions down
	// to the condition currently in force, and a run that came back holding the
	// ORIGINAL condition would re-arm an objective it had explicitly moved past.
	store := p.Store
	if info.Depth > 0 {
		store = nil
	}
	switch {
	case store == nil:
		store = NewStore(info.Goal)
	case store.Goal() != info.Goal:
		// A caller-supplied store still owns the trail; bring it to the recovered
		// condition rather than silently running on a private copy. adopt and not
		// Update, because this is not a revision — the loop read this condition
		// OUT of the log. Marking it pending would have the resumed run's first
		// turn drain it and announce a goal change that never happened.
		store.adopt(info.Goal, "recovered from the durable log")
	}
	g := &gateRun{store: store, revisable: p.Revisable, lifecycle: p.Lifecycle, newTaskOnInput: p.NewTaskOnInput}
	if p.Lifecycle {
		if p.TokenBudget < 0 || p.TimeBudget < 0 {
			return nil, fmt.Errorf("goal: negative budget")
		}
		store.mu.Lock()
		if store.state == nil {
			duration := p.TimeBudget.Milliseconds()
			if p.TimeBudget > 0 {
				duration = max(int64(1), duration)
			}
			store.state = &State{Status: Active, TokenBudget: p.TokenBudget, TimeBudgetMS: duration}
		}
		store.mu.Unlock()
	}
	return g, nil
}

// gateRun is one run's gate state. It is why this is a factory rather than a
// bare function: both stall breakers compare against the previous nudged finish,
// which is per-run.
type gateRun struct {
	// store holds the condition. It is read on every use rather than copied at
	// BeginRun, because the condition can change mid-run (update_goal) and a gate
	// enforcing a stale copy is worse than no gate: it holds the model to a
	// contract nothing else in the run still states.
	store          *Store
	newTaskOnInput bool
	lifecycle      bool
	accounted      int
	lastTick       time.Time
	revisable      bool
	nudges         int
	lastFinal      string
	// lastTools is the size of the run's tool trace at the previous nudge, and
	// idle counts how many nudges in a row have failed to move it. Tool calls are
	// the only progress the gate can see from here: a finish is by definition a
	// turn that called nothing, so a chain of them with a flat trace means the
	// model answered every nudge with prose alone.
	lastTools int
	idle      int
}

// Name identifies the extension in composition diagnostics.
func (*gateRun) Name() string { return "goal" }

// goal is the condition currently in force.
func (g *gateRun) goalText() string { return g.store.Goal() }

// SystemPrompt states the completion protocol before the model's first turn.
// Stating it once as a fixed prefix costs less than repeating it per turn, and
// it is in front of the model before the first answer rather than after it.
func (g *gateRun) SystemPrompt() string {
	return "## Goal gate\nThis run has a completion goal:\n" + strings.TrimSpace(g.goalText()) +
		"\nDo not stop until it is satisfied. End your final answer with the line \"" + Done +
		"\" once the goal is met, or \"" + Blocked +
		"\" plus the reason if only the user can unblock you. A finish without either line re-opens the run."
}

// TurnStopping holds the run to its goal.
//
// The stall breakers fire first, before the run is re-opened. Verbatim repeat is
// checked as an immediate give-up because it is unambiguous — the model was
// asked to continue and produced the same bytes. No-progress needs a few
// attempts before it means anything, since the first nudge is exactly the case
// where a model that simply forgot the sentinel gets to add it.
func (g *gateRun) TurnStopping(_ context.Context, info agentcore.StopInfo) agentcore.StopDecision {
	if satisfied(info.Final) {
		return agentcore.StopDecision{}
	}
	if g.nudges > 0 && info.Final == g.lastFinal {
		return agentcore.StopDecision{StopReason: StopReasonStalled}
	}
	// Progress resets the count rather than decrementing it: a model that goes
	// back to work has answered the nudge, and the tally of what it did before
	// that should not be held against it later in a long run.
	if g.nudges > 0 && len(info.Tools) == g.lastTools {
		g.idle++
		if g.idle >= maxIdleNudges {
			return agentcore.StopDecision{StopReason: StopReasonStalled}
		}
	} else {
		g.idle = 0
	}
	// Built before the counter moves: nudge() escalates on the SECOND and later
	// nudges, and incrementing first would hand the very first finish the
	// "your last answer was rejected" wording for an answer nothing had rejected
	// yet.
	msg := g.nudge()
	g.nudges++
	g.lastFinal = info.Final
	g.lastTools = len(info.Tools)
	return agentcore.StopDecision{
		Continue: true,
		Inject:   []agentcore.Message{{Role: agentcore.RoleUser, Content: msg}},
		Note:     "Goal not met — continuing.",
	}
}

// nudge is the synthetic user message injected when the model tries to finish
// without declaring the goal met or blocked.
//
// The first one explains the contract. Later ones stop explaining and start
// dictating: the model has now ignored the instruction at least twice, so
// repeating the same prose is the one thing known not to work. The escalated
// form names the failure, gives the literal line to emit, and rules out the
// specific near-miss that keeps a compliant-looking answer from counting — a
// sign-off after the sentinel, which is the most common way a weaker model
// misses a contract it is otherwise trying to follow.
func (g *gateRun) nudge() string {
	if g.nudges == 0 {
		return "[goal gate] The run's goal is not met yet:\n" + strings.TrimSpace(g.goalText()) +
			"\nKeep working toward it. When it is fully satisfied, end your answer with the line \"" + Done +
			"\". If you are genuinely blocked by something only the user can resolve, explain the blocker and end with \"" + Blocked + "\"."
	}
	return "[goal gate] Your last answer was rejected because it did not end with the required status line. " +
		"This is a formatting requirement, not a judgment about the work.\n\nThe goal is:\n" + strings.TrimSpace(g.goalText()) +
		"\n\nIf the goal is met, repeat your answer and make the LAST line of it exactly:\n" + Done +
		"\nIf you are blocked by something only the user can resolve, state the blocker and make the LAST line exactly:\n" + Blocked +
		"\n\nWrite nothing at all after that line — no sign-off, no follow-up question. " +
		"If the goal is not met, ignore the above and keep working."
}

// satisfied reports whether the final answer declares the goal met or the run
// honestly blocked. Only the LAST non-empty line is checked — the contract says
// "end your final answer with the line", and an unanchored match would let an
// answer that merely mentions the sentinel mid-prose ("I cannot yet write
// STATUS: DONE — the tests still fail") falsely close the gate. The line match
// is a contains, not a prefix, so markdown decoration around the sentinel
// ("**STATUS: DONE**") still counts.
func satisfied(final string) bool {
	lines := strings.Split(final, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		up := strings.ToUpper(line)
		return strings.Contains(up, Done) || strings.Contains(up, Blocked)
	}
	return false
}

// PublicText removes a standalone closing protocol line from customer prose.
// A status mentioned within an explanation is retained verbatim.
func PublicText(text string) string { return strings.TrimSpace(StripStatusLine(text)) }

// StripStatusLine also works on a buffered stream tail: it preserves every
// preceding byte, so hosts can hide the completion protocol without joining
// words or dropping whitespace between preview chunks.
func StripStatusLine(text string) string {
	trimmed := strings.TrimRight(text, " \t\r\n")
	index := strings.LastIndex(trimmed, "\n")
	normalized := strings.Trim(strings.ToUpper(strings.TrimSpace(trimmed[index+1:])), "*_` ")
	if normalized == Done || normalized == Blocked {
		if index < 0 {
			return ""
		}
		return trimmed[:index]
	}
	return text
}
