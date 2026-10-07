package agentcore

import "context"

// Forking is core because it is the ONE thing a delegation plugin cannot build
// from the outside: an Agent's capability-bearing fields are unexported, and
// they must stay that way — a child assembled field-by-field from a package
// that cannot see them would inherit whatever the author remembered, which is
// exactly how a child ends up with more scope than its parent.
//
// So the core answers "give me a child of yourself", and a plugin decides
// whether, when, and to whom to delegate. Core owns the inheritance rule;
// the plugin owns the policy.

// Fork builds an ephemeral child Agent for one delegated task.
//
// The child inherits every capability-bearing field verbatim — provider,
// ladder, tools, policy (including the permission gate already installed in
// hooks), memory, definition, limits, env, compaction, retry, caching, output
// cap, and the parent's extensions — so scope can only NARROW, never widen.
//
// It deliberately drops the run-control seams: no durable session, no
// steering/follow-up queues, no step gate, no PrepareNextTurn. A child is one
// bounded task, not a conversation.
//
// childSessionID, when non-empty AND the parent is durable, gives the child a
// durable log of its own and marks it resumable. Callers derive that id
// deterministically from the spawn (parent session + tool call id) so a
// replayed spawn REATTACHES — a completed child returns its recorded answer
// without re-running, and an interrupted one resumes from its own log —
// instead of duplicating the spend and the side effects. Empty leaves the
// child's history purely in memory.
//
// Depth is not a field: it rides the context (see WithDelegationDepth), so a
// cap holds across an A -> B -> A cycle the same as a straight chain.
func (a *Agent) Fork(childSessionID string) *Agent {
	child := &Agent{
		provider:           a.provider,
		nativeProvider:     a.nativeProvider,
		model:              a.model,
		modelCapabilities:  a.modelCapabilities,
		contextWindow:      a.contextWindow,
		tools:              a.tools,
		policy:             a.policy,
		hooks:              a.hooks,
		memory:             a.memory,
		def:                a.def,
		limits:             a.limits,
		env:                a.env,
		compaction:         a.compaction,
		compactionProvider: a.compactionProvider,
		compactionModel:    a.compactionModel,
		compactor:          a.compactor,
		refreshKey:         a.refreshKey,
		escalation:         a.escalation,
		retry:              a.retry,
		cacheKey:           a.cacheKey,
		cacheRetention:     a.cacheRetention,
		maxTokens:          a.maxTokens,
		reasoningEffort:    a.reasoningEffort,
		outputSchema:       a.outputSchema,
		outputValidator:    a.outputValidator,
		toolChoice:         a.toolChoice,
		parallelToolCalls:  cloneBool(a.parallelToolCalls),
		providerSession:    a.providerSession,
		providerSessionID:  childSessionID,
		extensions:         a.extensions,
	}
	if childSessionID != "" && a.session != nil && a.sessionID != "" {
		child.session = a.session
		child.sessionID = childSessionID
		child.resumeSession = true
	}
	return child
}

// IsDurable reports whether this agent writes an append-only session log, which
// is what makes a deterministic child-session id worth deriving.
func (a *Agent) IsDurable() bool { return a.session != nil && a.sessionID != "" }

// SessionID returns the durable log this agent writes to, or "" when the run is
// purely in-memory.
func (a *Agent) SessionID() string { return a.sessionID }

// AddChildUsage folds a delegated run's usage into this agent's accumulator, so
// the parent's RunResult and its budget gate both account for what its children
// spent. A delegation plugin calls this for every child it drives — including
// one that FAILED, whose tokens were still burned.
func (a *Agent) AddChildUsage(u Usage) { a.addChildUsage(u) }

// Delegation depth is core, not a plugin concern, even though nothing in the
// core loop spawns anything.
//
// The reason is the cap: a delegation plugin bounds recursion by refusing to
// advertise its spawn tool past a maximum depth, and that only works if the
// depth an A -> B -> A cycle has reached survives crossing an agent boundary
// where the two agents may be composed with DIFFERENT plugins. So the counter
// rides the context, the loop reads it into RunInfo.Depth, and every extension
// sees the same number.

// delegationDepthKey carries how many delegation hops deep the current run is.
// Depth travels on ctx — not on the Agent — because a cross-agent delegate is a
// freshly built Agent whose struct fields know nothing of the caller; ctx is
// the only thread that survives the hop, so it is what stops A→B→A recursion.
type delegationDepthKey struct{}

// DelegationDepth returns the current delegation depth (0 for a top-level run).
func DelegationDepth(ctx context.Context) int {
	if v, ok := ctx.Value(delegationDepthKey{}).(int); ok {
		return v
	}
	return 0
}

// WithDelegationDepth returns ctx marked as depth hops deep. Consumers normally
// never call this — the spawn tool wraps the child ctx itself — but a consumer
// embedding a run inside another delegation system may seed a floor.
func WithDelegationDepth(ctx context.Context, depth int) context.Context {
	return context.WithValue(ctx, delegationDepthKey{}, depth)
}

// Which agent is making this call?
//
// A spawned sub-agent shares its parent's provider (Fork copies it) and its
// parent's context, so anything decorating the provider — pricing, tracing,
// metering — sees the parent's calls and every descendant's calls arrive
// through the same object, in interleaved order, with nothing to tell them
// apart. A run that delegated three hundred tasks therefore reads back as one
// agent that inexplicably did all of it itself, and the six hundred child calls
// inflate the parent's own turn count.
//
// The run's session id is the natural discriminator: the loop already derives a
// distinct one per child ("<parent session>/<tool call id>"), and it is the same
// key the durable log is written under, so a trace tagged with it lines up with
// the log row for row. Depth (DelegationDepth) says how deep; this says which.
//
// It rides the context for the same reason depth does: a child is a different
// Agent value, and ctx is the only thread that survives the hop.

// runSessionKey carries the session id of the run currently making a call.
type runSessionKey struct{}

// WithRunSession tags ctx with the session id of the run about to execute. The
// loop sets it once per run, before the first turn; an empty id is a no-op, so
// an in-memory run (which has no session) leaves the tag absent rather than
// stamping a meaningless empty string over an outer one.
func WithRunSession(ctx context.Context, id string) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, runSessionKey{}, id)
}

// RunSessionFrom returns the session id of the run making the current call, or
// "" outside any run (a connectivity probe, a classifier call) and on a run with
// no durable session. Consumers must treat "" as "attribute it to the run", not
// as an error.
func RunSessionFrom(ctx context.Context) string {
	if v, ok := ctx.Value(runSessionKey{}).(string); ok {
		return v
	}
	return ""
}
