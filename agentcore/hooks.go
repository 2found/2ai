package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
)

// BeforeToolCall fires after a tool's arguments are schema-validated and before
// execution. Returning a blocked Decision stops the call; the reason is fed
// back to the model. The permission gate is itself a BeforeToolCall hook;
// consumers may add more (PII redaction, cost ceilings, human-approval gates).
type BeforeToolCall func(ctx context.Context, call ToolCall) Decision

// AfterToolCall fires after a tool executes. It may annotate or rewrite the
// result string, and may set terminate to end the run cleanly (a terminal tool
// such as submit_recommendation/finish).
type AfterToolCall func(ctx context.Context, call ToolCall, result string, runErr error) (rewritten string, terminate bool)

// ContextHook transforms the message list immediately before a provider request
// (pi's `context` event). It sees a copy of the live history and returns the
// view the model should reason over — e.g. redaction, trimming, injecting a
// reminder. Returning nil keeps the input unchanged. It does not mutate the
// persisted run history; only the outgoing request is affected.
type ContextHook func(ctx context.Context, msgs []Message) []Message

// PiContextHook transforms only the outgoing native message view. Provider
// messages retain their original JSON, including signed and custom content.
// Returning nil keeps the input. Failures are reported and retain the last
// valid view, as Pi requires transformContext to resolve without throwing.
type PiContextHook func(context.Context, []json.RawMessage) ([]json.RawMessage, error)

// ProviderRequestHook inspects or rewrites the assembled ChatRequest right
// before it is sent (pi's `before_provider_request`). Use it to pin a stop
// sequence, cap tools, or tee the request to a logger. Returning a zero-value
// request (no messages) is treated as "no change".
type ProviderRequestHook func(ctx context.Context, req ChatRequest) ChatRequest

// MessageEndHook observes a completed assistant message (pi's `message_end`).
// It is read-only: a return value is not threaded back. Use it for metrics,
// audit, or surfacing the turn to an external sink.
type MessageEndHook func(ctx context.Context, msg Message)

// RunStart is the assembled starting state of a run, handed to the
// BeforeAgentStart hooks after the system prompt is built (definition + recalled
// memory + skill headers + any goal contract) and before the first turn. It is
// the only seam that can shape the *first* request; PrepareNextTurn shapes every
// turn after one has completed.
type RunStart struct {
	// System is the assembled system prompt. It has not yet been prepended to
	// Messages — the loop does that after the hooks run.
	System string
	// Messages are the seed messages (the user prompt, or a continued thread).
	// They never include the system message.
	Messages []Message
	// Task is the recall/skill-selection task string for this run. Read-only:
	// changing it here has no effect (memory recall already happened).
	Task string
}

// BeforeAgentStartHook shapes a run before its first provider request (pi's
// before_agent_start). Return the input with your edits: an empty System keeps
// the assembled prompt, and nil Messages keep the seeds, so a hook that only
// wants to append a reminder can return RunStart{Messages: append(...)} without
// restating the prompt. Runs before the seed messages are persisted to the
// durable log, so an injected message is part of the recorded conversation and
// survives resume.
type BeforeAgentStartHook func(ctx context.Context, start RunStart) RunStart

// TurnInfo describes one reasoning turn at its boundary (pi's turn_start /
// turn_end). Usage is the run total accumulated so far, not this turn's slice.
type TurnInfo struct {
	Turn int
	// Model is the rung actually in use for this turn.
	Model string
	// Usage is the run-to-date total at the boundary.
	Usage Usage
	// StopReason is set at turn end only (empty at turn start) and carries the
	// provider's reason for the turn that just completed.
	StopReason string
}

// TurnHook observes a turn boundary. Unlike the StreamTurnStart / StreamTurnEnd
// stream events — which only reach a viewer on a *streamed* run — turn hooks
// fire on every run, so metering and audit do not depend on someone watching.
// Read-only: a return value is not threaded back.
type TurnHook func(ctx context.Context, info TurnInfo)

// CompactRequest describes an imminent context reduction. The default strategy
// may be at its half-budget deterministic-pruning threshold or at its full
// summarization threshold.
type CompactRequest struct {
	Turn int
	// Messages is the candidate history the active strategy would compact. For
	// the default strategy it already contains deterministic pruning; Skip still
	// leaves the original live transcript untouched.
	Messages []Message
	// Budget is the run's effective MaxContextTokens ceiling.
	Budget int
	// Settings are the effective compaction settings for this run (already
	// clamped to the budget).
	Settings CompactionSettings
}

// CompactDecision is a BeforeCompact hook's answer. The zero value means "no
// opinion" — the built-in summarize-and-elide compaction runs as usual.
type CompactDecision struct {
	// Skip abandons compaction for this turn. The transcript is left as-is, so a
	// hook that skips forever will run the context past its budget; use it to
	// defer (e.g. mid-tool-sequence), not to disable compaction.
	Skip bool
	// Messages, when non-nil, replaces the transcript with a consumer-supplied
	// compaction and suppresses the built-in one. The loop persists the same
	// compaction bracket either way, but no summarization call is billed.
	Messages []Message
}

// BeforeCompactHook is consulted immediately before the loop rewrites a
// transcript for pruning or summary compaction (pi's session_before_compact).
// The first hook that returns Skip, or a non-nil Messages, decides; later hooks
// are not consulted. Use it to pin content the default policy would drop, or
// to swap in a domain-specific reducer.
type BeforeCompactHook func(ctx context.Context, req CompactRequest) CompactDecision

// AgentEndHook observes a finished run (pi's agent_end), after sub-agent usage is
// folded in and any synthesized failure turn is appended — so res is exactly what
// the caller receives. Read-only, fires on every exit path including error and
// abort.
type AgentEndHook func(ctx context.Context, res RunResult)

// HookErrorPolicy governs what happens when a hook handler panics or returns an
// error. The default (continue) keeps a single bad hook from taking down a run.
type HookErrorPolicy string

const (
	// HookContinue logs/attributes the failure and proceeds (default). A
	// panicking observer never aborts the run.
	HookContinue HookErrorPolicy = "continue"
	// HookThrow aborts the run, surfacing the attributed hook error from the
	// loop. Opt-in, for consumers that treat a hook failure as fatal.
	HookThrow HookErrorPolicy = "throw"
)

// Hooks is the ordered set of lifecycle interceptors for a run. The tool hooks
// (Before/After) gate and rewrite tool execution; the typed event hooks
// (Context/BeforeProviderRequest/MessageEnd) participate in or observe the
// reasoning turn. All handlers are panic-hardened per ErrorPolicy.
type Hooks struct {
	Before []BeforeToolCall
	After  []AfterToolCall

	// BeforeAgentStart runs in order once per run, on the assembled system
	// prompt + seed messages, each seeing the previous one's output.
	BeforeAgentStart []BeforeAgentStartHook
	// TurnStart observers run in order at the top of every turn, before any of
	// the turn's work (compaction, steering, the provider call).
	TurnStart []TurnHook
	// TurnEnd observers run in order once a turn is complete, on every path out
	// of that turn (final answer, tool round, guard stop, abort).
	TurnEnd []TurnHook
	// BeforeCompact runs in order when the active strategy proposes pruning or
	// compaction; the first decisive answer (Skip, or replacement Messages) wins.
	BeforeCompact []BeforeCompactHook
	// AgentEnd observers run in order as the run returns, on every exit path.
	AgentEnd []AgentEndHook

	// Context runs in order before every provider request, each transforming the
	// message view the next one sees (a reducer over the message list).
	Context []ContextHook
	// PiContext is the native counterpart of Context, used by the Pi host.
	PiContext []PiContextHook
	// BeforeProviderRequest runs in order on the assembled request, each seeing
	// the previous one's output.
	BeforeProviderRequest []ProviderRequestHook
	// MessageEnd observers run in order after each assistant message completes.
	MessageEnd []MessageEndHook

	// ErrorPolicy governs handler panics/errors. Zero value == HookContinue.
	ErrorPolicy HookErrorPolicy
	// OnError, when set, receives every attributed hook failure (source + error)
	// regardless of policy, so a consumer can log/meter it. Source is a stable
	// identifier like "context[1]" or "before_tool_call[0]".
	OnError func(source string, err error)
}

// emitErr applies the error policy to one attributed handler failure: it always
// reports via OnError (if set), then either swallows it (continue) or returns a
// wrapped error for the loop to abort on (throw).
func (h Hooks) emitErr(source string, err error) error {
	if err == nil {
		return nil
	}
	if h.OnError != nil {
		h.OnError(source, err)
	}
	if h.ErrorPolicy == HookThrow {
		return fmt.Errorf("hook %s: %w", source, err)
	}
	return nil
}

// safe runs fn, converting a panic into an error so one reckless handler cannot
// crash the run goroutine. The recovered value is wrapped for attribution.
func safe(fn func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()
	fn()
	return nil
}

// runBefore evaluates the before-hooks in order, short-circuiting on the first
// block. A panicking before-hook is attributed and skipped (treated as
// non-blocking under HookContinue); under HookThrow it propagates as a block so
// the call is refused rather than silently executed.
func (h Hooks) runBefore(ctx context.Context, call ToolCall) Decision {
	for i, hook := range h.Before {
		var d Decision = Allowed()
		if perr := safe(func() { d = hook(ctx, call) }); perr != nil {
			if e := h.emitErr(fmt.Sprintf("before_tool_call[%d]", i), perr); e != nil {
				return Blocked(e.Error())
			}
			continue
		}
		if !d.Allow {
			return d
		}
	}
	return Allowed()
}

// runAfter threads the result through the after-hooks in order; terminate is
// sticky once any hook sets it. A panicking after-hook is attributed and
// skipped (the prior result is kept); under HookThrow the run is unaffected
// here — after-hooks observe/annotate and never block — but the failure is
// still reported via OnError.
func (h Hooks) runAfter(ctx context.Context, call ToolCall, result string, runErr error) (string, bool) {
	terminate := false
	for i, hook := range h.After {
		var rewritten string
		var t bool
		if perr := safe(func() { rewritten, t = hook(ctx, call, result, runErr) }); perr != nil {
			_ = h.emitErr(fmt.Sprintf("after_tool_call[%d]", i), perr)
			continue
		}
		result = rewritten
		terminate = terminate || t
	}
	return result, terminate
}

// runContext folds the context hooks over the message list, each seeing the
// previous one's output. A panicking hook is attributed and skipped (its input
// is passed through). Under HookThrow a failure aborts via a returned error.
func (h Hooks) runContext(ctx context.Context, msgs []Message) ([]Message, error) {
	for i, hook := range h.Context {
		var out []Message
		source := fmt.Sprintf("context[%d]", i)
		if perr := safe(func() { out = hook(ctx, msgs) }); perr != nil {
			if e := h.emitErr(source, perr); e != nil {
				return msgs, e
			}
			continue
		}
		if out != nil {
			msgs = out
		}
	}
	return msgs, nil
}

// runBeforeProviderRequest folds the request hooks over the assembled request.
// A hook that returns a request with no messages is treated as "no change".
func (h Hooks) runBeforeProviderRequest(ctx context.Context, req ChatRequest) (ChatRequest, error) {
	for i, hook := range h.BeforeProviderRequest {
		var out ChatRequest
		source := fmt.Sprintf("before_provider_request[%d]", i)
		if perr := safe(func() { out = hook(ctx, req) }); perr != nil {
			if e := h.emitErr(source, perr); e != nil {
				return req, e
			}
			continue
		}
		if len(out.Messages) > 0 {
			req = out
		}
	}
	return req, nil
}

// runMessageEnd dispatches the read-only message_end observers in order. A
// panicking observer is attributed and skipped; under HookThrow a failure
// aborts via a returned error.
func (h Hooks) runMessageEnd(ctx context.Context, msg Message) error {
	for i, hook := range h.MessageEnd {
		source := fmt.Sprintf("message_end[%d]", i)
		if perr := safe(func() { hook(ctx, msg) }); perr != nil {
			if e := h.emitErr(source, perr); e != nil {
				return e
			}
		}
	}
	return nil
}

// runBeforeAgentStart folds the run-start hooks over the assembled prompt and
// seeds. Per-field "empty means unchanged": a hook that returns an empty System
// keeps the prompt it was given, and nil Messages keep the seeds, so a hook may
// edit one without restating the other. A panicking hook is attributed and
// skipped (its input passes through); under HookThrow it aborts the run.
func (h Hooks) runBeforeAgentStart(ctx context.Context, start RunStart) (RunStart, error) {
	for i, hook := range h.BeforeAgentStart {
		var out RunStart
		source := fmt.Sprintf("before_agent_start[%d]", i)
		if perr := safe(func() { out = hook(ctx, start) }); perr != nil {
			if e := h.emitErr(source, perr); e != nil {
				return start, e
			}
			continue
		}
		if out.System != "" {
			start.System = out.System
		}
		if out.Messages != nil {
			start.Messages = out.Messages
		}
	}
	return start, nil
}

// runTurnHooks dispatches one read-only turn-boundary observer list in order.
// source names the list ("turn_start"/"turn_end") for attribution.
func (h Hooks) runTurnHooks(ctx context.Context, list []TurnHook, source string, info TurnInfo) error {
	for i, hook := range list {
		if perr := safe(func() { hook(ctx, info) }); perr != nil {
			if e := h.emitErr(fmt.Sprintf("%s[%d]", source, i), perr); e != nil {
				return e
			}
		}
	}
	return nil
}

// runBeforeCompact asks the compaction hooks in order, stopping at the first
// decisive answer (a Skip, or a non-nil replacement transcript). A panicking hook
// is attributed and skipped so a broken hook cannot suppress compaction — the
// loop falls through to the built-in path, which is the safe direction. Under
// HookThrow the failure aborts the run.
func (h Hooks) runBeforeCompact(ctx context.Context, req CompactRequest) (CompactDecision, error) {
	for i, hook := range h.BeforeCompact {
		var d CompactDecision
		source := fmt.Sprintf("before_compact[%d]", i)
		if perr := safe(func() { d = hook(ctx, req) }); perr != nil {
			if e := h.emitErr(source, perr); e != nil {
				return CompactDecision{}, e
			}
			continue
		}
		if d.Skip || d.Messages != nil {
			return d, nil
		}
	}
	return CompactDecision{}, nil
}

// runAgentEnd dispatches the read-only agent_end observers in order. It is called
// from a deferred close on the run, so it never returns an error to abort with:
// the run is already over. Failures are still attributed via OnError.
func (h Hooks) runAgentEnd(ctx context.Context, res RunResult) {
	for i, hook := range h.AgentEnd {
		if perr := safe(func() { hook(ctx, res) }); perr != nil {
			_ = h.emitErr(fmt.Sprintf("agent_end[%d]", i), perr)
		}
	}
}
