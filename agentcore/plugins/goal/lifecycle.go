package goal

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/2found/2ai/agentcore"
)

type Status string

const (
	Active        Status = "active"
	Paused        Status = "paused"
	BudgetLimited Status = "budget_limited"
	Complete      Status = "complete"
	Dropped       Status = "dropped"
)

// State is retained with the transcript, never reconstructed from a summary.
// ElapsedMS counts active execution only; time spent paused/offline is excluded.
type State struct {
	Objective    string `json:"objective"`
	Status       Status `json:"status"`
	TokenBudget  int    `json:"token_budget,omitempty"`
	TokensUsed   int    `json:"tokens_used"`
	TimeBudgetMS int64  `json:"time_budget_ms,omitempty"`
	ElapsedMS    int64  `json:"elapsed_ms"`
	Reason       string `json:"reason,omitempty"`
}

// Command is a host-authorized lifecycle operation. Budgets are absolute totals,
// not additional grants. Resume never resets usage or bypasses an exhausted cap.
// The model can inspect state; pause/resume/drop belong to the caller.
type Command struct {
	Action       string `json:"action"`
	Reason       string `json:"reason,omitempty"`
	TokenBudget  *int   `json:"token_budget,omitempty"`
	TimeBudgetMS *int64 `json:"time_budget_ms,omitempty"`
}

func (s *Store) State() State {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == nil {
		return State{}
	}
	state := *s.state
	state.Objective = s.goal
	return state
}
func limited(s State) bool {
	return (s.TokenBudget > 0 && s.TokensUsed >= s.TokenBudget) || (s.TimeBudgetMS > 0 && s.ElapsedMS >= s.TimeBudgetMS)
}
func (g *gateRun) account(usage agentcore.Usage) {
	if !g.lifecycle {
		return
	}
	g.store.mu.Lock()
	defer g.store.mu.Unlock()
	s := g.store.state
	tokens := usage.InputTokens + usage.OutputTokens + usage.CacheReadTokens + usage.CacheWriteTokens
	s.TokensUsed += max(0, tokens-g.accounted)
	g.accounted = max(g.accounted, tokens)
	now := time.Now()
	if !g.lastTick.IsZero() {
		s.ElapsedMS += max(int64(0), now.Sub(g.lastTick).Milliseconds())
	}
	g.lastTick = time.Time{}
	if s.Status == Active {
		g.lastTick = now
	}
	if s.Status == Active && limited(*s) {
		s.Status = BudgetLimited
		s.Reason = "goal budget exhausted"
	}
}
func (g *gateRun) ControlRun(_ context.Context, info agentcore.StepInfo) (string, error) {
	if !g.lifecycle {
		return "", nil
	}
	g.account(info.Usage)
	switch g.store.State().Status {
	case Active:
		return "", nil
	case Paused:
		return "goal_paused", nil
	case BudgetLimited:
		return "goal_budget_limited", nil
	case Complete:
		return "goal_complete", nil
	case Dropped:
		return "goal_dropped", nil
	default:
		return "", fmt.Errorf("goal: invalid lifecycle state")
	}
}
func (g *gateRun) HandleRunCommand(ctx context.Context, raw json.RawMessage) (json.RawMessage, error) {
	if !g.lifecycle {
		return nil, fmt.Errorf("goal lifecycle is disabled")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var c Command
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	state, err := g.store.ApplyCommand(ctx, c)
	if err != nil {
		return nil, err
	}
	if c.Action != "get" {
		g.lastTick = time.Time{}
	}
	return json.Marshal(state)
}

// ApplyCommand lets a host pause a live run at its next settled boundary.
// It is safe to call concurrently; resume across processes uses NativeRun.Commands.
func (s *Store) ApplyCommand(ctx context.Context, c Command) (State, error) {
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == nil {
		return State{}, fmt.Errorf("goal lifecycle is not initialized")
	}
	next := *s.state
	if len(c.Reason) > 2000 {
		return State{}, fmt.Errorf("goal: reason exceeds 2000 bytes")
	}
	if c.TokenBudget != nil {
		if *c.TokenBudget < 0 {
			return State{}, fmt.Errorf("goal: negative token budget")
		}
		next.TokenBudget = *c.TokenBudget
	}
	if c.TimeBudgetMS != nil {
		if *c.TimeBudgetMS < 0 {
			return State{}, fmt.Errorf("goal: negative time budget")
		}
		next.TimeBudgetMS = *c.TimeBudgetMS
	}
	switch c.Action {
	case "get":
		if c.TokenBudget != nil || c.TimeBudgetMS != nil {
			return State{}, fmt.Errorf("goal: get cannot change budgets")
		}
	case "pause":
		if next.Status != Active && next.Status != Paused {
			return State{}, fmt.Errorf("goal: cannot pause %s", next.Status)
		}
		next.Status = Paused
	case "resume":
		if next.Status != Active && next.Status != Paused && next.Status != BudgetLimited {
			return State{}, fmt.Errorf("goal: cannot resume %s", next.Status)
		}
		if limited(next) {
			return State{}, fmt.Errorf("goal: raise exhausted budget before resuming")
		}
		next.Status = Active
	case "drop":
		if next.Status == Complete {
			return State{}, fmt.Errorf("goal: cannot drop completed goal")
		}
		next.Status = Dropped
	case "create":
		// Start another task under the host's unchanged completion contract.
		if next.Status != Complete && next.Status != Dropped {
			return State{}, fmt.Errorf("goal: unfinished goal already exists")
		}
		next.TokensUsed = 0
		next.ElapsedMS = 0
		next.Status = Active
	default:
		return State{}, fmt.Errorf("goal: unknown action %q", c.Action)
	}
	if c.Action != "get" {
		next.Reason = strings.TrimSpace(c.Reason)
		*s.state = next
	}
	next.Objective = s.goal
	return next, nil
}
func (g *gateRun) FinalizeRun(_ context.Context, result agentcore.RunResult, failure error) error {
	if !g.lifecycle {
		return nil
	}
	g.account(result.Usage)
	g.lastTick = time.Time{}
	g.store.mu.Lock()
	defer g.store.mu.Unlock()
	s := g.store.state
	if s.Status != Active {
		return nil
	}
	switch result.StopReason {
	case "max_turns", "max_tool_calls", "budget_exhausted":
		s.Status = BudgetLimited
		s.Reason = result.StopReason
	default:
		if failure == nil && !result.Parked && (result.StopReason == "stop" || result.StopReason == "end_turn" || result.StopReason == "") && satisfied(result.Final) {
			lines := strings.Split(strings.TrimSpace(result.Final), "\n")
			if strings.Contains(strings.ToUpper(lines[len(lines)-1]), Done) {
				s.Status = Complete
				s.Reason = "completion accepted"
			}
		}
	}
	return nil
}
func (g *gateRun) NativeState() (json.RawMessage, error) {
	g.store.mu.Lock()
	defer g.store.mu.Unlock()
	if g.store.state != nil {
		g.store.state.Objective = g.store.goal
	}
	return json.Marshal(struct {
		State *State `json:"lifecycle,omitempty"`
	}{g.store.state})
}
func (g *gateRun) RestoreNativeState(raw json.RawMessage) error {
	var checkpoint struct {
		State *State `json:"lifecycle"`
	}
	if err := json.Unmarshal(raw, &checkpoint); err != nil {
		return err
	}
	if checkpoint.State == nil {
		return nil
	}
	if !g.lifecycle {
		return fmt.Errorf("goal: lifecycle checkpoint requires lifecycle plugin")
	}
	s := checkpoint.State
	switch s.Status {
	case Active, Paused, BudgetLimited, Complete, Dropped:
	default:
		return fmt.Errorf("goal: invalid status")
	}
	if s.TokenBudget < 0 || s.TokensUsed < 0 || s.TimeBudgetMS < 0 || s.ElapsedMS < 0 || len(s.Reason) > 2000 {
		return fmt.Errorf("goal: invalid accounting")
	}
	g.store.mu.Lock()
	g.store.state = s
	g.store.mu.Unlock()
	return nil
}

type stateTool struct{ run *gateRun }

func (*stateTool) Name() string      { return "get_goal" }
func (*stateTool) Bookkeeping() bool { return true }
func (*stateTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{Name: "get_goal", Description: "Read the current goal lifecycle and remaining token/time budget. Only the host can pause, resume or drop a goal.", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}
}
func (t *stateTool) Run(ctx context.Context, _ string) (string, error) {
	raw, err := t.run.HandleRunCommand(ctx, json.RawMessage(`{"action":"get"}`))
	return string(raw), err
}

// ObserveMessages starts a new host task after a terminal goal only when the
// host opts in. Paused and budget-limited goals require an explicit command.
func (g *gateRun) ObserveMessages(_ context.Context, phase agentcore.ObservePhase, turn int, msgs []agentcore.Message) {
	if !g.lifecycle || !g.newTaskOnInput || phase != agentcore.PhaseExternalInput || turn != 0 || len(msgs) == 0 {
		return
	}
	g.store.mu.Lock()
	defer g.store.mu.Unlock()
	s := g.store.state
	if s.Status == Complete || s.Status == Dropped {
		s.Status = Active
		s.TokensUsed = 0
		s.ElapsedMS = 0
		s.Reason = "new user task"
		g.accounted = 0
		g.lastTick = time.Time{}
	}
}
