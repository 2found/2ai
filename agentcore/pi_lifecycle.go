package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	nativehost "github.com/2found/2ai/agentcore/host"
)

// ObservePiMessages delivers a detached display projection to existing plugin
// observers. Native request/history JSON remains authoritative and cannot be
// rewritten by a legacy observer.
func (h *PiToolHost) ObservePiMessages(ctx context.Context, phase ObservePhase, turn int, raw json.RawMessage) error {
	h.gate.RLock()
	defer h.gate.RUnlock()
	if h.closed || !h.lifecycle.Load() {
		return errors.New("Pi tool host is not running")
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(raw, &messages); err != nil {
		return err
	}
	projected := make([]Message, 0, len(messages))
	for _, message := range messages {
		value, err := nativehost.ProjectMessage(message)
		if err != nil {
			return err
		}
		projected = append(projected, value)
	}
	h.exts.observe(piToolContext{Context: ctx, values: h.ctx}, phase, turn, projected)
	return nil
}

// PiTurnPreparation contains host-authored additions for the next native turn.
// Append Messages through Pi's prompt/prepareNextTurnWithContext lifecycle;
// never rebuild provider history from these legacy host message values.
type PiTurnPreparation struct {
	Messages     []Message
	DisableTools bool
	StopReason   string
	Note         string
}

// PiGoal is the composed host's completion contract. Native persistence keeps
// it separately from provider messages so a resumed host can re-arm the gate.
func (h *PiToolHost) PiGoal() string {
	h.gate.RLock()
	defer h.gate.RUnlock()
	return h.goal
}

// RefreshPiGoal drains committed extension updates between native turns. The
// returned prompt is a new named system section, never a history replacement.
func (h *PiToolHost) RefreshPiGoal() (system string, changed bool, err error) {
	h.gate.Lock()
	defer h.gate.Unlock()
	if h.closed || !h.lifecycle.Load() {
		return "", false, errors.New("Pi tool host is not running")
	}
	next := h.goal
	for _, goal := range h.exts.goalRevisions() {
		next = goal
	}
	if next != h.committedGoal {
		return "", false, errors.New("native goal extension differs from the committed condition")
	}
	changed = next != h.goal
	h.goal = next
	if changed {
		system = h.baseSystem
		for _, section := range h.exts.systemPrompt() {
			system += "\n\n" + section
		}
	}
	return system, changed, nil
}

// PiCompactionPolicy exposes the composed host's existing context limits to
// native consumer transforms, including limits inherited by a fork.
func (h *PiToolHost) PiCompactionPolicy() (budget, keepRecent int) {
	budget = effectiveBudget(h.agent.limits.MaxContextTokens, h.agent.contextWindow)
	return budget, effectiveCompaction(h.agent.compaction, budget).KeepRecentTokens
}

// ValidatePiSession checks that native persistence and the governed tool host
// agree on the session identity used for tool idempotency and extension scope.
func (h *PiToolHost) ValidatePiSession(id string, durable bool) error {
	if h.agent.sessionID != id || (h.agent.session != nil) != durable {
		return errors.New("Pi session binding differs from the composed tool host")
	}
	return nil
}

// StartPiRun claims the host for one native run and assembles the composed definition, recalled memory, skills,
// and extension instructions. It does not execute legacy transcript-editing
// hooks; callers integrating those hooks must preserve the native transcript.
func (h *PiToolHost) StartPiRun(ctx context.Context, task string) (string, error) {
	return h.StartPiRunWithLifecycle(ctx, task, nil)
}

// StartPiRunWithLifecycle is StartPiRun with an optional observer for bounded
// setup stages. It exposes the stage, never recalled memory content.
func (h *PiToolHost) StartPiRunWithLifecycle(ctx context.Context, task string, lifecycle func(string)) (string, error) {
	h.gate.Lock()
	defer h.gate.Unlock()
	if h.closed {
		return "", errors.New("Pi tool host is closed")
	}
	if !h.lifecycle.CompareAndSwap(false, true) {
		return "", errors.New("Pi tool host already bound to a run")
	}
	ctx, cancel := context.WithCancel(ctx)
	detach := context.AfterFunc(h.ctx, cancel)
	defer func() { detach(); cancel() }()
	if h.ctx.Err() != nil {
		cancel()
	}
	ctx = piToolContext{Context: ctx, values: h.ctx}
	var recalled []MemoryEntry
	if h.agent.memory != nil {
		if lifecycle != nil {
			lifecycle("Checking relevant memory")
		}
		if got, err := h.agent.memory.Recall(ctx, h.agent.def.ScopeID, task, 8); err == nil {
			recalled = got
			if lifecycle != nil {
				lifecycle(fmt.Sprintf("Memory context ready (%d)", len(recalled)))
			}
		} else if lifecycle != nil {
			lifecycle("Memory lookup unavailable; continuing without recalled memory")
		}
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	system := buildSystemPrompt(h.agent.def, recalled, h.agent.def.enabledSkills())
	h.baseSystem = system
	for _, section := range h.exts.systemPrompt() {
		system += "\n\n" + section
	}
	return system, nil
}

// PreparePiTurn applies the composed step gate and step extensions. A ceiling
// allows one tool-free wrap-up; FinishPiTurn ends it regardless of stop guards.
// The native host calls this serially, once before each provider turn.
func (h *PiToolHost) PreparePiTurn(ctx context.Context, info StepInfo) (PiTurnPreparation, error) {
	h.gate.RLock()
	defer h.gate.RUnlock()
	if h.closed {
		return PiTurnPreparation{}, errors.New("Pi tool host is closed")
	}
	ctx, cancel := context.WithCancel(ctx)
	detach := context.AfterFunc(h.ctx, cancel)
	defer func() { detach(); cancel() }()
	if h.ctx.Err() != nil {
		cancel()
	}
	ctx = piToolContext{Context: ctx, values: h.ctx}
	info.Usage = addUsage(info.Usage, h.agent.peekChildUsage())
	if err := ctx.Err(); err != nil {
		return PiTurnPreparation{}, err
	}
	if h.agent.stepGate != nil {
		if err := h.agent.stepGate(ctx, info.Turn); err != nil {
			return PiTurnPreparation{}, err
		}
	}
	if err := h.agent.hooks.runTurnHooks(ctx, h.agent.hooks.TurnStart, "turn_start", TurnInfo{Turn: info.Turn, Model: info.Model, Usage: info.Usage}); err != nil {
		return PiTurnPreparation{}, err
	}
	var reason string
	switch {
	case info.Turn-info.BookkeepingTurns > h.agent.limits.MaxTurns:
		reason = "max_turns"
	case h.budget.exhausted():
		reason = "max_tool_calls"
	case h.agent.budgetGate != nil && h.agent.budgetGate(ctx, info.Usage):
		reason = "budget_exhausted"
	}
	if reason != "" {
		h.finalizing.Store(true)
		return PiTurnPreparation{Messages: []Message{{Role: RoleUser, Content: finalizeSteer(reason)}}, DisableTools: true, StopReason: reason, Note: finalizeNote(reason)}, nil
	}
	messages := h.exts.beforeStep(ctx, info)
	if h.agent.getSteering != nil {
		steering := h.agent.getSteering(ctx)
		h.exts.observe(ctx, PhaseExternalInput, info.Turn, steering)
		messages = append(messages, steering...)
	}
	return PiTurnPreparation{Messages: messages}, nil
}

// PiCompletedTurn is a display/control projection of one completed native turn.
// Outcomes must be in assistant source order, not parallel completion order.
type PiCompletedTurn struct {
	Info              StopInfo
	Usage             Usage
	Model, StopReason string
	Calls             []ToolCall
	Outcomes          []PiToolOutcome
}

// PiTurnDecision tells the native host whether to end or schedule another turn.
type PiTurnDecision struct {
	StopDecision
	End    bool
	Parked bool
}

// FinishPiTurn runs batch and stop policies over a completed native turn. Tool
// effects and native transcript placement have already settled at this point.
func (h *PiToolHost) FinishPiTurn(ctx context.Context, turn PiCompletedTurn) (PiTurnDecision, error) {
	h.gate.RLock()
	defer h.gate.RUnlock()
	if h.closed {
		return PiTurnDecision{}, errors.New("Pi tool host is closed")
	}
	ctx, cancel := context.WithCancel(ctx)
	detach := context.AfterFunc(h.ctx, cancel)
	defer func() { detach(); cancel() }()
	if h.ctx.Err() != nil {
		cancel()
	}
	ctx = piToolContext{Context: ctx, values: h.ctx}

	if err := h.agent.hooks.runTurnHooks(ctx, h.agent.hooks.TurnEnd, "turn_end", TurnInfo{Turn: turn.Info.Turns, Model: turn.Model, Usage: addUsage(turn.Usage, h.agent.peekChildUsage()), StopReason: turn.StopReason}); err != nil {
		return PiTurnDecision{}, err
	}
	if turn.StopReason == "error" || turn.StopReason == "aborted" {
		return PiTurnDecision{}, nil
	}
	// Pi retains the original response in native history. Reject invalid text
	// through its failure lifecycle before a stop guard can accept the answer.
	if len(turn.Calls) == 0 && h.agent.outputValidator != nil {
		if err := h.agent.outputValidator(turn.Info.Final); err != nil {
			return PiTurnDecision{}, err
		}
	}
	batch, err := h.finishPiBatch(ctx, turn.Calls, turn.Outcomes)
	if err != nil || batch.End {
		return batch, err
	}

	reason, controlErr := h.controlRun(ctx, StepInfo{Turn: turn.Info.Turns + 1, Model: turn.Model, Usage: turn.Usage})
	if controlErr != nil {
		return PiTurnDecision{}, controlErr
	}
	if reason != "" {
		return PiTurnDecision{End: true, StopDecision: StopDecision{StopReason: reason}}, nil
	}
	if len(turn.Calls) > 0 {
		// Pi decides whether this batch schedules another request. A native
		// before/after-tool hook may have terminated every result; extra host
		// context must not resurrect that deliberately ended batch.
		return batch, nil
	}
	stop, _ := h.exts.turnStopping(ctx, turn.Info)
	if stop.Continue {
		h.exts.observe(ctx, PhaseExternalInput, turn.Info.Turns, stop.Inject)
	} else {
		var follow []Message
		// A correction may arrive while the last answer is streaming. Poll
		// steering here too, otherwise no next-turn preparation would run.
		if h.agent.getSteering != nil {
			follow = h.agent.getSteering(ctx)
		}
		if len(follow) == 0 && h.agent.getFollowUp != nil {
			follow = h.agent.getFollowUp(ctx)
		}
		if len(follow) > 0 {
			h.exts.observe(ctx, PhaseExternalInput, turn.Info.Turns, follow)
			stop = StopDecision{Continue: true, Inject: follow}
		}
	}
	return PiTurnDecision{StopDecision: stop}, nil
}

// FinishPiDelegationBatch applies batch policy after every parked call in the
// original assistant batch has settled. Calls and outcomes must retain source
// order and include nondelegated siblings. This is not another model turn: it
// does not rerun TurnEnd, output validation, steering, or stop guards. The
// durable host must record the decision before delivering its contexts.
func (h *PiToolHost) FinishPiDelegationBatch(ctx context.Context, calls []ToolCall, outcomes []PiToolOutcome) (PiTurnDecision, error) {
	h.gate.RLock()
	defer h.gate.RUnlock()
	if h.closed || !h.lifecycle.Load() {
		return PiTurnDecision{}, errors.New("Pi tool host is not running")
	}
	ctx, cancel := context.WithCancel(ctx)
	detach := context.AfterFunc(h.ctx, cancel)
	defer func() { detach(); cancel() }()
	if h.ctx.Err() != nil {
		cancel()
	}
	ctx = piToolContext{Context: ctx, values: h.ctx}
	return h.finishPiBatch(ctx, calls, outcomes)
}

// Caller holds the host gate and has bound cancellation and host values.
func (h *PiToolHost) finishPiBatch(ctx context.Context, calls []ToolCall, outcomes []PiToolOutcome) (PiTurnDecision, error) {
	if err := ctx.Err(); err != nil {
		return PiTurnDecision{}, err
	}
	var extra []Message
	var parked, terminal bool
	for _, outcome := range outcomes {
		extra = append(extra, outcome.AdditionalContexts...)
		parked = parked || outcome.Parked
		terminal = terminal || outcome.Terminate
	}
	if parked || terminal || h.finalizing.Load() {
		return PiTurnDecision{End: true, Parked: parked}, nil
	}
	if len(calls) > 0 {
		// Extension callbacks may edit their slice; source history is immutable.
		extra = append(extra, h.exts.interceptBatch(ctx, slices.Clone(calls))...)
	}
	if err := ctx.Err(); err != nil {
		return PiTurnDecision{}, err
	}
	return PiTurnDecision{StopDecision: StopDecision{Inject: extra}}, nil
}

// CompletePiRun closes run resources, folds child spend once, and emits terminal
// observers. Provider history stays native; result.Messages is a read-only
// display projection for observers, never input to another provider request.
func (h *PiToolHost) CompletePiRun(ctx context.Context, result RunResult) RunResult {
	h.cancel()
	h.gate.Lock()
	defer h.gate.Unlock()
	if !h.completed.CompareAndSwap(false, true) {
		return result
	}
	if !h.closed {
		h.closed = true
		h.exts.closeRun()
		defer h.agent.release()
	}
	result.Usage = addUsage(result.Usage, h.agent.takeChildUsage())
	h.agent.hooks.runAgentEnd(piToolContext{Context: ctx, values: h.ctx}, result)
	return result
}

// TransformPiContext applies native request-view hooks without rewriting the
// durable transcript. A failing or mutating hook cannot damage the last valid
// view: each receives an isolated copy, and invalid output is discarded.
func (h *PiToolHost) TransformPiContext(ctx context.Context, raw json.RawMessage) json.RawMessage {
	h.gate.RLock()
	defer h.gate.RUnlock()
	if h.closed || (len(h.agent.hooks.PiContext) == 0 && len(h.exts.contexts) == 0) {
		return raw
	}
	ctx, cancel := context.WithCancel(ctx)
	detach := context.AfterFunc(h.ctx, cancel)
	defer func() { detach(); cancel() }()
	if h.ctx.Err() != nil {
		cancel()
	}
	ctx = piToolContext{Context: ctx, values: h.ctx}
	var current []json.RawMessage
	if err := json.Unmarshal(raw, &current); err != nil {
		return raw
	}
	hooks := slices.Clone(h.agent.hooks.PiContext)
	sources := make([]string, len(hooks))
	for i := range hooks {
		sources[i] = fmt.Sprintf("pi_context[%d]", i)
	}
	for _, extension := range h.exts.contexts {
		hooks = append(hooks, extension.TransformNativeContext)
		sources = append(sources, "extension_context["+extension.Name()+"]")
	}
	for i, hook := range hooks {
		if ctx.Err() != nil {
			return raw
		}
		input := make([]json.RawMessage, len(current))
		for j, m := range current {
			input[j] = append(json.RawMessage(nil), m...)
		}
		var next []json.RawMessage
		var err error
		if panicErr := safe(func() { next, err = hook(ctx, input) }); panicErr != nil {
			err = panicErr
		}
		if err == nil && next != nil {
			for _, m := range next {
				var header struct{ Role string }
				if decodeErr := json.Unmarshal(m, &header); decodeErr != nil || header.Role == "" {
					err = errors.New("invalid native context message")
					break
				}
			}
		}
		if err != nil {
			// The native transform contract is non-throwing, including when an
			// error reporter itself panics or legacy hooks request HookThrow.
			_ = safe(func() { _ = h.agent.hooks.emitErr(sources[i], err) })
			continue
		}
		if next != nil {
			current = next
		}
	}
	out, err := json.Marshal(current)
	if err != nil {
		return raw
	}
	return out
}
