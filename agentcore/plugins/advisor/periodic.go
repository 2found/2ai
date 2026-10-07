package advisor

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/2found/2ai/agentcore"
)

// ReviewerConfig names an independent review policy. No goroutines or second
// scheduler: periodic reviews run at the core's existing turn boundary.
type ReviewerConfig struct {
	Name               string
	Reviewer           Reviewer
	IntervalTurns      int
	MaxRounds          int
	MaxPeriodicReviews int
	CooldownTurns      int
}

func defaultPositive(n, d int) int {
	if n > 0 {
		return n
	}
	return d
}
func boundedMessages(msgs []agentcore.Message) []agentcore.Message {
	out := append([]agentcore.Message(nil), msgs[max(0, len(msgs)-32):]...)
	budget := 32000
	for i := len(out) - 1; i >= 0; i-- {
		// Reviewers see a detached text projection; native signatures stay in core.
		content := out[i].Content
		for _, call := range out[i].ToolCalls[:min(8, len(out[i].ToolCalls))] {
			content += "\nTool call " + agentcore.TruncateBytes(call.Name, 64) + ": " + agentcore.TruncateMiddle(call.Arguments, 1024)
		}
		out[i] = agentcore.Message{Role: out[i].Role, Name: agentcore.TruncateBytes(out[i].Name, 64), ToolCallID: agentcore.TruncateBytes(out[i].ToolCallID, 128), Content: agentcore.TruncateMiddle(content, min(4000, budget))}
		budget -= len(out[i].Content)
		if budget <= 0 {
			return out[i:]
		}
	}
	return out
}
func boundedTools(tools []agentcore.ToolTrace) []agentcore.ToolTrace {
	out := append([]agentcore.ToolTrace(nil), tools[max(0, len(tools)-32):]...)
	for i := range out {
		out[i].Args = agentcore.TruncateMiddle(out[i].Args, 1000)
		out[i].Error = agentcore.TruncateMiddle(out[i].Error, 1000)
	}
	return out
}
func (a *advisorRun) BeforeStep(ctx context.Context, info agentcore.StepInfo) agentcore.StepDecision {
	a.ticks += max(0, info.Turn-1-a.lastTurn)
	a.lastTurn = max(a.lastTurn, info.Turn-1)
	if a.interval <= 0 || a.periodic >= a.maxPeriodic || a.ticks-a.lastReview < a.interval || a.ticks < a.cooldownUntil {
		return agentcore.StepDecision{}
	}
	a.periodic++
	a.lastReview = a.ticks
	d := a.review(ctx, Review{Turns: a.ticks, Messages: a.messages, Round: a.periodic - 1, Delivered: append([]Note(nil), a.delivered...)})
	return agentcore.StepDecision{AdditionalContexts: d.Inject}
}

type advisorState struct {
	Periodic      int            `json:"periodic"`
	Finished      int            `json:"finished"`
	Ticks         int            `json:"ticks"`
	LastReview    int            `json:"last_review"`
	CooldownUntil int            `json:"cooldown_until"`
	Delivered     []Note         `json:"delivered"`
	Seen          map[string]int `json:"seen"`
	Order         []string       `json:"order"`
}

func (a *advisorRun) NativeState() (json.RawMessage, error) {
	return json.Marshal(advisorState{a.periodic, a.finished, a.ticks, a.lastReview, a.cooldownUntil, a.delivered, a.guard.seen, a.guard.order})
}
func (a *advisorRun) RestoreNativeState(raw json.RawMessage) error {
	var s advisorState
	if err := json.Unmarshal(raw, &s); err != nil {
		return err
	}
	if s.Periodic < 0 || s.Finished < 0 || s.Ticks < 0 || s.LastReview < 0 || s.LastReview > s.Ticks || s.CooldownUntil < 0 || len(s.Delivered) > 32 || len(s.Seen) > noteCapacity || len(s.Order) > noteCapacity {
		return fmt.Errorf("advisor: invalid checkpoint")
	}
	for _, n := range s.Delivered {
		if len([]rune(n.Text)) > MaxNoteRunes {
			return fmt.Errorf("advisor: oversized note")
		}
	}
	a.periodic = s.Periodic
	a.finished = s.Finished
	a.ticks = s.Ticks
	a.lastReview = s.LastReview
	a.cooldownUntil = s.CooldownUntil
	a.delivered = s.Delivered
	a.guard.Reset()
	if s.Seen != nil {
		a.guard.seen = s.Seen
		a.guard.order = s.Order
	}
	return nil
}

type advisorGroup struct {
	names []string
	runs  []*advisorRun
}

func (*advisorGroup) Name() string { return "advisor" }
func (p Plugin) beginGroup(ctx context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	if len(p.Reviewers) > 4 || p.Reviewer != nil {
		return nil, fmt.Errorf("advisor: configure either Reviewer or 1..4 named Reviewers")
	}
	group := &advisorGroup{}
	seen := map[string]bool{}
	for _, cfg := range p.Reviewers {
		if cfg.Name == "" || len(cfg.Name) > 64 || seen[cfg.Name] || cfg.Reviewer == nil {
			return nil, fmt.Errorf("advisor: unique name and reviewer required")
		}
		seen[cfg.Name] = true
		ext, err := (Plugin{Reviewer: cfg.Reviewer, IntervalTurns: cfg.IntervalTurns, MaxRounds: cfg.MaxRounds, MaxPeriodicReviews: cfg.MaxPeriodicReviews, CooldownTurns: cfg.CooldownTurns, Timeout: p.Timeout, MaxNotesPerReview: p.MaxNotesPerReview, OnNotes: p.OnNotes}).BeginRun(ctx, info)
		if err != nil {
			return nil, err
		}
		group.names = append(group.names, cfg.Name)
		group.runs = append(group.runs, ext.(*advisorRun))
	}
	return group, nil
}
func (g *advisorGroup) ObserveMessages(ctx context.Context, p agentcore.ObservePhase, t int, m []agentcore.Message) {
	for _, a := range g.runs {
		a.ObserveMessages(ctx, p, t, m)
	}
}
func (g *advisorGroup) BeforeStep(ctx context.Context, i agentcore.StepInfo) agentcore.StepDecision {
	var d agentcore.StepDecision
	for _, a := range g.runs {
		d.AdditionalContexts = append(d.AdditionalContexts, a.BeforeStep(ctx, i).AdditionalContexts...)
	}
	return d
}
func (g *advisorGroup) TurnStopping(ctx context.Context, i agentcore.StopInfo) agentcore.StopDecision {
	var d agentcore.StopDecision
	for _, a := range g.runs {
		i.Attempt = 0
		n := a.TurnStopping(ctx, i)
		d.Inject = append(d.Inject, n.Inject...)
		d.Continue = d.Continue || n.Continue
	}
	return d
}
func (g *advisorGroup) NativeState() (json.RawMessage, error) {
	s := map[string]json.RawMessage{}
	for i, a := range g.runs {
		raw, err := a.NativeState()
		if err != nil {
			return nil, err
		}
		s[g.names[i]] = raw
	}
	return json.Marshal(s)
}
func (g *advisorGroup) RestoreNativeState(raw json.RawMessage) error {
	var s map[string]json.RawMessage
	if err := json.Unmarshal(raw, &s); err != nil {
		return err
	}
	for i, a := range g.runs {
		if r := s[g.names[i]]; len(r) > 0 {
			if err := a.RestoreNativeState(r); err != nil {
				return err
			}
		}
	}
	return nil
}

// FinalizeRun counts the last settled turn even when no next request is made.
func (a *advisorRun) FinalizeRun(_ context.Context, r agentcore.RunResult, _ error) error {
	a.ticks += max(0, r.Turns-a.lastTurn)
	a.lastTurn = max(a.lastTurn, r.Turns)
	return nil
}
func (g *advisorGroup) FinalizeRun(ctx context.Context, r agentcore.RunResult, err error) error {
	for _, a := range g.runs {
		a.FinalizeRun(ctx, r, err)
	}
	return nil
}
