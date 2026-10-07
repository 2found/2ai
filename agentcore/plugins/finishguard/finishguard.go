// Package finishguard installs verify-on-stop: a bounded second look before a
// normal finish is accepted.
//
// The guard never runs checks itself. It turns passive run evidence — the tool
// trace, the answer text — into one synthetic follow-up at the exact moment the
// model tries to finish, which is the only moment where "you did not actually
// check that" is both cheap to say and still actionable.
//
// It is consulted only on a NORMAL finish (the model produced a text answer
// with no tool calls). A budget wrap-up, a tool-budget stop, a MaxTurns stop, an
// abort, or a terminal tool never re-opens the run: those are stops the run did
// not choose, and nudging them would wedge it.
//
// Generalized from hermes-agent's turn-end verification guard.
package finishguard

import (
	"context"

	"github.com/2found/2ai/agentcore"
)

// DefaultMaxNudges caps injections per run so a guard that is never satisfied
// cannot loop the run against MaxTurns (hermes max_attempts).
const DefaultMaxNudges = 2

// State is the evidence snapshot handed to a Guard: the answer the model just
// produced, the run shape so far, and how many nudges this guard has already
// spent. Tools aliases the run's live trace slice — read-only.
type State struct {
	// Final is the assistant text the run would return.
	Final string
	// Turns is the number of reasoning turns consumed, including this one.
	Turns int
	// Tools is the run's tool trace so far (allowed and blocked calls).
	Tools []agentcore.ToolTrace
	// Nudges counts guard messages already injected this run (0 on the first
	// consultation), so a guard can weaken or drop its demand on the retry.
	Nudges int
}

// Guard is consulted when the model produces a final answer and the run would
// end normally. Returning a non-empty string injects it as a synthetic user
// message — persisted to the durable log like a steer — and the loop continues
// instead of returning, giving the model a bounded chance to verify or repair
// before finishing. Returning "" accepts the finish.
type Guard func(ctx context.Context, state State) string

// Plugin installs the guard. A nil Guard accepts every finish, so a composition
// that wires the plugin but has nothing to check is inert rather than broken.
type Plugin struct {
	Guard Guard
	// MaxNudges bounds injections per run. 0 uses DefaultMaxNudges. The cap
	// lives here rather than in the loop because it is this capability's
	// property: the loop only knows that SOMETHING asked to continue.
	MaxNudges int
}

// Of wraps a guard function with the default cap.
func Of(g Guard) Plugin { return Plugin{Guard: g} }

// Name identifies the plugin and the extension it installs.
func (Plugin) Name() string { return "finish_guard" }

// Register adds the plugin as a run extension.
func (p Plugin) Register(r *agentcore.Registry) error {
	if p.Guard == nil {
		return nil
	}
	r.AddExtension(p)
	return nil
}

// BeginRun starts this run's nudge budget. A plugin with no guard declines.
func (p Plugin) BeginRun(context.Context, agentcore.RunInfo) (agentcore.Extension, error) {
	if p.Guard == nil {
		return nil, nil
	}
	max := p.MaxNudges
	if max <= 0 {
		max = DefaultMaxNudges
	}
	return &guardRun{guard: p.Guard, max: max}, nil
}

// guardRun is one run's guard state: the per-run nudge count is why this is a
// factory rather than a bare function.
type guardRun struct {
	guard Guard
	max   int
}

// Name identifies the extension in composition diagnostics.
func (*guardRun) Name() string { return "finish_guard" }

// TurnStopping consults the guard on a finish the run chose.
//
// The loop supplies Attempt (how many times THIS extension has already
// re-opened the run), so the cap is enforced against a count the extension
// cannot get wrong — a guard that forgets to bound itself is still bounded.
// Returning Continue with nothing to inject would re-run the same turn against
// an unchanged conversation, so the loop treats that as accepting the finish;
// this returns the zero decision instead, which says the same thing plainly.
func (g *guardRun) TurnStopping(ctx context.Context, info agentcore.StopInfo) agentcore.StopDecision {
	if info.Attempt >= g.max {
		return agentcore.StopDecision{}
	}
	nudge := g.guard(ctx, State{
		Final:  info.Final,
		Turns:  info.Turns,
		Tools:  info.Tools,
		Nudges: info.Attempt,
	})
	if nudge == "" {
		return agentcore.StopDecision{}
	}
	return agentcore.StopDecision{
		Continue: true,
		Inject:   []agentcore.Message{{Role: agentcore.RoleUser, Content: nudge}},
		Note:     "verifying before finishing",
	}
}
