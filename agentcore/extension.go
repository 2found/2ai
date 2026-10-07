package agentcore

import (
	"context"
	"encoding/json"
	"strings"
)

// Extension points: the only way a plugin reaches a running agent.
//
// The rule this file exists to enforce: THE LOOP NAMES NO PLUGIN. It emits at
// fixed points and dispatches over whatever extensions the composition
// installed. Delete every plugin package and the loop still compiles and runs —
// it just does less. That is the difference between a plugin and a
// configuration flag, and it is the property the previous design failed: the
// loop used to call spill, jobs, and the repeat guard by name, so ejecting one
// meant editing the core.
//
// What is deliberately NOT an extension point: anything the loop always does.
// The turn loop, the provider call, tool dispatch, history, the durable log,
// usage accounting, and the permission gate are core. Making an always-on
// mechanism "pluggable" buys nothing and costs a seam that can be misconfigured
// into an ungoverned agent.
//
// Ported from deepseek-harness's event taxonomy (agent/pre-step,
// tools/pre-execute, tools/post-execute with additionalContexts buffering,
// agent/turn-stopping), rebuilt as Go interfaces rather than a string-keyed
// event bus: the compiler checks the wiring, and a typo is a build error
// instead of a listener that silently never fires.

// RunInfo describes the run an extension is being created for. It is what a
// plugin needs to build per-run state without reaching into the Agent.
type RunInfo struct {
	// SessionID is the durable session, or "" on an in-memory run.
	SessionID string
	// Owner is a stable token identifying THIS run, used to fence run-scoped
	// resources (a background job started here must not be visible to another
	// run). It equals SessionID on a durable run and is otherwise unique.
	Owner string
	// ScopeID is the running agent's own memory/persona scope — the same value
	// recall reads under (def.ScopeID). Extensions that write agent-private
	// state pin to it so a model-supplied value can never widen the scope.
	ScopeID string
	// Limits are the run's effective bounds.
	Limits Limits
	// Depth is the delegation depth: 0 for a top-level run, higher inside a
	// spawned sub-agent.
	Depth int
	// Durable reports whether a session store is recording this run.
	Durable bool
	// Session is the store recording this run, or nil. It is handed over so a
	// retrieval extension can read the run's OWN log without the consumer
	// wiring the same store twice and risking the two disagreeing about which
	// session is being searched. Read-only by convention: the loop owns every
	// write, which is what keeps "model-visible means logged" enforceable.
	Session SessionStore
	// Goal is the run's completion condition when one is set — the configured
	// goal, or the one recovered from the durable log on a resume. Empty means
	// the run is ungated. It is carried here so a gate extension never has to
	// read the log itself; only the loop writes and recovers the record.
	Goal string
	// Bookkeeping reports whether a tool's calls are administrative rather than
	// progress toward the task (see BookkeepingTool). Never nil.
	//
	// It exists so an extension can treat such calls differently — a loop
	// detector must not count a plan updater's repeats as a stuck chain —
	// WITHOUT depending on the package that provides the tool. Asking the
	// question generically is what keeps capabilities from importing each
	// other.
	Bookkeeping func(tool string) bool
	// Agent is the running agent, non-nil. It is here for the one capability
	// that cannot be built without it — delegation, which needs Fork to produce
	// a child whose scope provably only narrows. Most extensions ignore it.
	Agent *Agent
}

// ExtensionFactory is what a plugin registers. The loop calls BeginRun exactly
// once per run, so a fresh run gets a fresh instance without resetting shared
// state. State accessed by parallel tools still needs synchronization.
//
// Returning a nil Extension is how a plugin declines to participate in a
// particular run (delegation already at max depth, no session to query, a
// disabled feature). It is not an error.
type ExtensionFactory interface {
	// Name identifies the extension in diagnostics and in Agent.Describe.
	Name() string
	// BeginRun builds the per-run instance. An error aborts the run before any
	// provider call, so a misconfigured extension fails loudly rather than
	// silently doing nothing.
	BeginRun(ctx context.Context, info RunInfo) (Extension, error)
}

// Extension is one plugin's per-run instance.
//
// The interface is deliberately near-empty: capabilities are OPTIONAL
// interfaces discovered by type assertion, the same way net/http treats
// Flusher or Hijacker. An extension implements only the points it cares about,
// and adding a new point never breaks an existing plugin.
type Extension interface {
	// Name identifies this extension instance.
	Name() string
}

// NativeContextContributor transforms the outgoing native request view using
// this run's extension state. It must not reconstruct provider messages from
// display projections. Like Pi's transformContext, failures retain the prior
// valid view; returned additions are not written into authoritative history.
type NativeContextContributor interface {
	Extension
	TransformNativeContext(context.Context, []json.RawMessage) ([]json.RawMessage, error)
}

// NativeStateContributor persists a plugin's bounded state in the opaque native
// checkpoint. Restore runs before lifecycle hooks; the consumer commits the
// resulting checkpoint together with its successful run transaction.
type NativeStateContributor interface {
	NativeState() (json.RawMessage, error)
	RestoreNativeState(json.RawMessage) error
}

// --- contribution points -------------------------------------------------

// ToolContributor adds tools to the run's registry. Implement it to give the
// model a capability.
//
// Contributing a tool is not granting it: the permission gate still decides
// whether the model may call it, unless the tool also declares itself
// self-gated (see SelfGated).
type ToolContributor interface {
	// Tools returns this extension's tools for this run. Called once, after
	// BeginRun, so the set may depend on run state (depth, durability).
	Tools() []Tool
}

// SelfGated is an optional Tool capability: a tool that CANNOT reach anything
// the agent does not already have may run without consulting the permission
// policy.
//
// This is a real security surface, so it is narrow and explicit. The rule: a
// self-gated tool may only return something this run already produced, or
// something authored into this agent's own definition. read_skill (definition
// bodies), read_spill (output this run generated and had truncated), and
// session_query (this session's own log) qualify. A tool that reaches the
// network, the filesystem, or another run does NOT, whatever its author
// believes.
//
// Installing a plugin is already a trusted act — it is Go code compiled into
// the binary — so a plugin declaring its own exemption is no weaker than the
// hardcoded list it replaces. It is strictly more auditable: Agent.Describe
// prints every gate-exempt tool.
type SelfGated interface {
	// SelfGated reports that this tool bypasses the permission gate.
	SelfGated() bool
}

// BookkeepingTool is an optional Tool capability: a tool whose calls are
// administrative rather than progress toward the task.
//
// A turn spent ONLY on bookkeeping calls is refunded against MaxTurns, so
// keeping a plan current — or any similar self-management — cannot starve the
// turn budget on a long task. The MaxToolCalls budget still backstops a runaway
// bookkeeping loop, so this cannot be used to buy unbounded turns.
//
// The loop asks the TOOL rather than consulting a list of names, which is what
// lets a planning capability live in its own package: a bookkeeping tool from a
// plugin earns the refund without the loop knowing the tool exists.
type BookkeepingTool interface {
	// Bookkeeping reports that calls to this tool are not task progress.
	Bookkeeping() bool
}

// PromptContributor appends to the system prompt. Implement it to give the
// model a standing instruction (a completion contract, a capability
// explanation) that must survive compaction.
//
// The contribution is appended after the definition and recalled memory, in
// extension registration order, so the prompt prefix stays stable across turns
// and the provider's KV cache is not invalidated mid-run.
type PromptContributor interface {
	// SystemPrompt returns text to append, or "" to contribute nothing.
	SystemPrompt() string
}

// ContextContributor lets an extension attach a value to the RUN context, so a
// capability is reachable from inside a tool call without the tool importing
// the extension. Background jobs use it: a long-running tool finds its launcher
// on the context it was handed.
//
// Contributions are folded in registration order before the first turn. Prefer
// a contributed Tool where one works — a context value is invisible in the
// composition, so it is the weaker form of dependency.
type ContextContributor interface {
	RunContext(ctx context.Context) context.Context
}

// RunCloser releases whatever the extension acquired for this run. The loop
// calls it on EVERY exit path, including error and abort, so an extension that
// starts goroutines cannot leak them past the run that owns them.
type RunCloser interface {
	CloseRun()
}

// --- interception points -------------------------------------------------

// ToolResultDecision is a ToolInterceptor's answer.
//
// The zero value means "no opinion": the result passes through untouched. This
// matters — an interceptor that only wants to OBSERVE returns the zero value
// and cannot accidentally blank a result.
type ToolResultDecision struct {
	// Result replaces the tool's output when Replace is true. Replacing is
	// explicit rather than inferred from a non-empty string, so an interceptor
	// cannot erase a result by returning a zero value.
	// Replacing also tells the loop the result is already bounded, so its own
	// default truncation is skipped — that is what lets a lossless bounding
	// strategy exist at all, since the default is lossy and runs first
	// otherwise.
	Result  string
	Replace bool
	// Meta annotates the trace without entering the model's context (for example
	// a cache verdict). Empty leaves the trace unchanged.
	Meta string
	// ResultRef is an opaque handle that recovers content omitted from Result
	// (for example a spill locator). It is persisted on the tool message and
	// context-reduction paths preserve and surface it in replacement text. Keep
	// unrelated trace metadata in Meta: the model may be shown ResultRef later.
	ResultRef string
	// AdditionalContexts are messages to add to the conversation because of
	// this call. The LOOP appends them after ALL tool results in the batch —
	// never interleaved, which would break tool-call/result adjacency — and
	// persists them to the durable log.
	//
	// Persistence is the loop's job precisely so it cannot be forgotten: an
	// extension that injects a message the model sees but the log misses would
	// make a resumed run replay a different conversation. Plugins get the
	// injection; the core keeps the invariant.
	AdditionalContexts []Message
	// Terminate ends the run cleanly after this batch (a terminal tool).
	Terminate bool
}

// ToolInterceptor wraps tool execution. It runs AFTER the permission gate and
// after the tool has executed, so it observes only calls that were actually
// allowed and run.
//
// Interceptors are chained in registration order, each seeing the previous
// one's Result — a waterfall, so a size limiter and a redactor compose.
type ToolInterceptor interface {
	InterceptToolResult(ctx context.Context, call ToolCall, result string, runErr error) ToolResultDecision
}

// BatchDecision is a BatchInterceptor's answer. The zero value adds nothing.
type BatchDecision struct {
	// AdditionalContexts are appended after the batch's tool results and
	// persisted, exactly like ToolResultDecision.AdditionalContexts.
	AdditionalContexts []Message
}

// BatchInterceptor observes a whole batch of tool calls at once. Implement it
// when a decision depends on the SET of calls rather than any single one —
// detecting that the model repeated itself, or that a group of calls made no
// progress.
//
// It runs after every call in the batch has been intercepted individually.
type BatchInterceptor interface {
	InterceptBatch(ctx context.Context, calls []ToolCall) BatchDecision
}

// StepInfo describes the step about to run.
type StepInfo struct {
	Turn int
	// BookkeepingTurns is the number of completed administrative turns refunded
	// against MaxTurns. Turn itself remains monotonic for hooks and observers.
	BookkeepingTurns int
	// Model is the rung in use for this step.
	Model string
	// Usage is the run-to-date total.
	Usage Usage
}

// StepDecision is a StepInterceptor's answer. The zero value adds nothing.
type StepDecision struct {
	// AdditionalContexts are prepended to the step's messages and persisted.
	// Use it to tell the model something that became true between turns — a
	// background job finished, an external state changed.
	AdditionalContexts []Message
}

// StepInterceptor runs at the top of each turn, before the provider request is
// assembled. Implement it to inject information the model needs for THIS turn.
type StepInterceptor interface {
	BeforeStep(ctx context.Context, info StepInfo) StepDecision
}

// GoalReviser is an extension that can change what the run is trying to
// achieve, partway through achieving it.
//
// A long run discovers things. The requirement it was given can turn out to be
// impossible, to have been based on a wrong assumption, or to be the wrong shape
// for what the work actually found — and a run that can only ever pursue the
// condition it started with either grinds against that discovery until MaxTurns
// or lies about being done. So an extension may revise the condition, and the
// loop drains the revision once per turn.
//
// The loop owns what follows, because both consequences are the loop's to
// guarantee. It records the new condition as an EntryGoal, so a crashed run
// resumes gated on the CURRENT objective rather than the original one; and it
// rebuilds the system prompt, so the contract the model reads is the contract
// being enforced. An extension that changed the condition privately would leave
// the model working against a stale prompt and recovery re-arming a stale gate.
//
// This is a real transfer of authority: an agent that can redefine success can
// declare success. The gate stops being a contract the model cannot relax and
// becomes one it can renegotiate — which is the point, and the cost. What keeps
// it accountable is that every revision is durable and attributable: the log
// holds each condition the run has held, in order, so a narrowing is visible
// afterwards rather than silent.
type GoalReviser interface {
	// Native Pi integrations must call RecordGoalRevision in a governed tool
	// context before publishing the condition. The drain then refreshes the
	// prompt; durability already belongs to that tool's physical effect.
	// ReviseGoal returns the run's new completion condition and true when one
	// has been set since the last call. It is drained, not polled: returning
	// (x, true) twice for the same revision would write the same EntryGoal every
	// turn for the rest of the run.
	ReviseGoal() (goal string, ok bool)
}

// StopInfo describes a run that is about to finish normally.
type StopInfo struct {
	// Final is the answer the model produced.
	Final string
	Turns int
	// Tools is the run's tool trace, so a guard can check what evidence exists.
	Tools []ToolTrace
	// Attempt counts how many times THIS extension has already re-opened the
	// run, so a guard can bound itself.
	Attempt int
}

// StopDecision is a StopInterceptor's answer. The zero value accepts the
// finish.
type StopDecision struct {
	// Continue re-opens the run instead of accepting the answer.
	Continue bool
	// Inject is the message explaining why, added to the conversation and
	// persisted. Continue with nothing to inject would spin the loop against an
	// unchanged conversation, so it is treated as accepting the finish.
	Inject []Message
	// Note is a short, user-facing progress line. The raw Inject text is
	// internal rail scaffolding and is deliberately NOT shown to the end user.
	Note string
	// StopReason overrides the recorded reason when this extension gives up
	// (a stalled goal, an exhausted guard).
	StopReason string
}

// StopInterceptor is consulted when the model produces a final answer and the
// run would end normally. Implement it to hold the run to a contract — a
// completion sentinel, a verification pass, an evidence requirement.
//
// It is NOT consulted on a stop the run did not choose: a budget wrap-up, a
// tool-budget stop, a MaxTurns stop, an abort, or a terminal tool. Nudging
// those would wedge a run that is already being shut down.
//
// Interceptors are consulted in registration order and the FIRST one that
// returns Continue wins; the rest are not consulted, so two guards cannot both
// inject into the same turn.
type StopInterceptor interface {
	TurnStopping(ctx context.Context, info StopInfo) StopDecision
}

// StreamDecision is a StreamInterceptor's answer. The zero value lets the
// stream continue untouched.
type StreamDecision struct {
	// Abort cuts the in-flight assistant message: the loop cancels the
	// provider stream, discards the partial text, appends Inject to the
	// conversation, and retries the turn's provider call from the same point.
	// Abort with nothing to Inject would retry against an unchanged
	// conversation — the model would emit the same tokens and abort again — so
	// it is treated as "no opinion" and the stream completes.
	Abort bool
	// Inject is the correction the retried turn must see — the rule body the
	// matched pattern violated. The LOOP appends it to the history and persists
	// it like a steer, so a resumed run replays the conversation the retry was
	// actually given.
	Inject []Message
}

// StreamInterceptor observes the assistant's output WHILE it is being produced
// and may cut it off mid-token. It is the only extension point that fires on
// partial output: every other interceptor sees a finished artifact (a tool
// result, a batch, a final answer), which is too late for a rule whose whole
// purpose is that the model must never complete the pattern — a leaked secret,
// a forbidden phrase, a banned construct. Aborting mid-stream is what keeps
// the completed violation out of the transcript entirely.
//
// The interceptor is handed the assistant text accumulated SO FAR this turn —
// the same buffer the loop is building — once per content delta, and again once
// on the non-streaming path with the completed response. It is called
// synchronously inside the delta loop, so a check must be cheap: a regex over
// the buffer, not a network call. Interceptors are consulted in registration
// order and the FIRST abort wins; the rest are not asked.
//
// The loop retries the turn after an abort, so an interceptor MUST bound
// itself (a per-turn injection cap) or it spins the provider call forever.
// The loop backstops a runaway interceptor at maxStreamAbortsPerTurn and then
// lets the stream complete unmodified — pass-through, not failure, because a
// guard that can take the run down is worse than the output it was policing.
type StreamInterceptor interface {
	InterceptStreamDelta(ctx context.Context, accumulated string) StreamDecision
}

// --- observation ---------------------------------------------------------

// RunObserver watches a run without changing it. Implement it for metering,
// audit, or invariant checking.
//
// Every method is called for effect only; return values are not threaded back,
// so an observer cannot alter the run even by mistake.
type RunObserver interface {
	// ObserveMessages is called with the live history immediately before each
	// provider request, and again after any rewrite (compaction, context edit)
	// so an observer tracking history can rebase.
	ObserveMessages(ctx context.Context, phase ObservePhase, turn int, msgs []Message)
}

// ObservePhase distinguishes why an observer is being called.
type ObservePhase string

const (
	// PhaseRestore supplies the original checkpoint history before new input.
	PhaseRestore ObservePhase = "restore"
	// PhaseAppend reports messages newly added to the history.
	PhaseAppend ObservePhase = "append"
	// PhaseRebase reports that the history was deliberately REPLACED
	// (compaction, a context edit), so any prior baseline is void.
	PhaseRebase ObservePhase = "rebase"
	// PhaseRequest reports the history about to be sent to the provider.
	PhaseRequest ObservePhase = "request"
	// PhaseExternalInput reports that new input arrived from OUTSIDE the model
	// (a steer, a follow-up). An extension tracking the model's own behavior
	// should treat this as a break: the model has been given information it did
	// not have, so what it does next is a fresh decision rather than a
	// continuation of what it was doing.
	PhaseExternalInput ObservePhase = "external_input"
)

// --- dispatch ------------------------------------------------------------

// extensionSet is the loop's view of the installed extensions: the optional
// interfaces, pre-sorted so the hot path does no type assertions per turn.
type extensionSet struct {
	all       []Extension
	contexts  []NativeContextContributor
	tools     []ToolContributor
	prompts   []PromptContributor
	toolIntcp []ToolInterceptor
	batch     []BatchInterceptor
	steps     []StepInterceptor
	streams   []StreamInterceptor
	stops     []StopInterceptor
	revisers  []GoalReviser
	observers []RunObserver
	closers   []RunCloser
	// stopAttempts counts re-opens per extension so a StopInterceptor can bound
	// itself without holding the count.
	stopAttempts map[string]int
}

// ToolResultArchiver is an optional durable-output seam for request-only
// context rescue. It must save the complete text before returning a locator
// served by the run's existing retrieval tool. It grants no new effects.
type ToolResultArchiver interface {
	ArchiveToolResult(context.Context, ToolCall, string) (string, error)
}

// beginExtensions instantiates every registered factory for one run.
func beginExtensions(ctx context.Context, factories []ExtensionFactory, info RunInfo) (*extensionSet, error) {
	set := &extensionSet{stopAttempts: map[string]int{}}
	for _, f := range factories {
		ext, err := f.BeginRun(ctx, info)
		if err != nil {
			set.closeRun()
			return nil, err
		}
		if ext == nil {
			// Declining this run is normal, not a failure.
			continue
		}
		set.add(ext)
	}
	return set, nil
}

// add classifies one extension by the optional interfaces it implements.
func (s *extensionSet) add(ext Extension) {
	if contributor, ok := ext.(NativeContextContributor); ok {
		s.contexts = append(s.contexts, contributor)
	}
	s.all = append(s.all, ext)
	if v, ok := ext.(ToolContributor); ok {
		s.tools = append(s.tools, v)
	}
	if v, ok := ext.(PromptContributor); ok {
		s.prompts = append(s.prompts, v)
	}
	if v, ok := ext.(ToolInterceptor); ok {
		s.toolIntcp = append(s.toolIntcp, v)
	}
	if v, ok := ext.(BatchInterceptor); ok {
		s.batch = append(s.batch, v)
	}
	if v, ok := ext.(StepInterceptor); ok {
		s.steps = append(s.steps, v)
	}
	if v, ok := ext.(StreamInterceptor); ok {
		s.streams = append(s.streams, v)
	}
	if v, ok := ext.(StopInterceptor); ok {
		s.stops = append(s.stops, v)
	}
	if v, ok := ext.(GoalReviser); ok {
		s.revisers = append(s.revisers, v)
	}
	if v, ok := ext.(RunObserver); ok {
		s.observers = append(s.observers, v)
	}
	if v, ok := ext.(RunCloser); ok {
		s.closers = append(s.closers, v)
	}
}

// runContext folds the extensions' context contributions.
func (s *extensionSet) runContext(ctx context.Context) context.Context {
	for _, e := range s.all {
		if c, ok := e.(ContextContributor); ok {
			if next := c.RunContext(ctx); next != nil {
				ctx = next
			}
		}
	}
	return ctx
}

// closeRun releases every extension's run-scoped resources.
func (s *extensionSet) closeRun() {
	for _, c := range s.closers {
		closer := c
		_ = safe(func() { closer.CloseRun() })
	}
}

// contributedTools returns every extension-contributed tool, in registration
// order, plus the names that declared themselves gate-exempt.
func (s *extensionSet) contributedTools() ([]Tool, map[string]bool) {
	var out []Tool
	exempt := map[string]bool{}
	for _, c := range s.tools {
		for _, t := range c.Tools() {
			if t == nil {
				continue
			}
			out = append(out, t)
			if g, ok := t.(SelfGated); ok && g.SelfGated() {
				exempt[t.Name()] = true
			}
		}
	}
	return out, exempt
}

// systemPrompt returns the extensions' appended prompt sections.
func (s *extensionSet) systemPrompt() []string {
	var out []string
	for _, p := range s.prompts {
		if text := p.SystemPrompt(); text != "" {
			out = append(out, text)
		}
	}
	return out
}

// interceptToolResult folds the interceptors over one tool result. Each sees
// the previous one's output, so a spill and a redactor compose.
func (s *extensionSet) interceptToolResult(ctx context.Context, call ToolCall, result string, runErr error) (out, meta, resultRef string, extra []Message, replaced, terminate bool) {
	out = result
	for _, ic := range s.toolIntcp {
		var d ToolResultDecision
		// A panicking interceptor must degrade to "no opinion", not crash the
		// run — same contract as a panicking hook.
		if perr := safe(func() { d = ic.InterceptToolResult(ctx, call, out, runErr) }); perr != nil {
			continue
		}
		if d.Replace {
			out = d.Result
			replaced = true
		}
		if d.Meta != "" {
			meta = d.Meta
		}
		if d.ResultRef != "" {
			resultRef = d.ResultRef
		}
		extra = append(extra, d.AdditionalContexts...)
		terminate = terminate || d.Terminate
	}
	return out, meta, resultRef, extra, replaced, terminate
}

// interceptBatch collects whole-batch injections.
func (s *extensionSet) interceptBatch(ctx context.Context, calls []ToolCall) []Message {
	var extra []Message
	for _, ic := range s.batch {
		var d BatchDecision
		if perr := safe(func() { d = ic.InterceptBatch(ctx, calls) }); perr != nil {
			continue
		}
		extra = append(extra, d.AdditionalContexts...)
	}
	return extra
}

// goalRevisions drains every extension that changed the run's objective since
// the last turn. Returned in registration order; the LAST one wins, the same way
// the durable log's last EntryGoal wins on resume, so the two agree by
// construction rather than by the caller remembering to make them.
func (s *extensionSet) goalRevisions() []string {
	var out []string
	for _, r := range s.revisers {
		var g string
		var ok bool
		if perr := safe(func() { g, ok = r.ReviseGoal() }); perr != nil {
			continue
		}
		if ok && strings.TrimSpace(g) != "" {
			out = append(out, strings.TrimSpace(g))
		}
	}
	return out
}

// beforeStep collects per-turn injections.
func (s *extensionSet) beforeStep(ctx context.Context, info StepInfo) []Message {
	var extra []Message
	for _, ic := range s.steps {
		var d StepDecision
		if perr := safe(func() { d = ic.BeforeStep(ctx, info) }); perr != nil {
			continue
		}
		extra = append(extra, d.AdditionalContexts...)
	}
	return extra
}

// turnStopping consults the stop interceptors. The first that wants to
// continue wins; the rest are not asked.
func (s *extensionSet) turnStopping(ctx context.Context, info StopInfo) (StopDecision, string) {
	for _, ic := range s.stops {
		name := "extension"
		if e, ok := ic.(Extension); ok {
			name = e.Name()
		}
		in := info
		in.Attempt = s.stopAttempts[name]
		var d StopDecision
		if perr := safe(func() { d = ic.TurnStopping(ctx, in) }); perr != nil {
			continue
		}
		// Continuing with nothing to say would re-run the same turn against an
		// unchanged conversation, so it is treated as accepting the finish.
		if d.Continue && len(d.Inject) > 0 {
			s.stopAttempts[name]++
			return d, name
		}
		if d.StopReason != "" {
			return StopDecision{StopReason: d.StopReason}, name
		}
	}
	return StopDecision{}, ""
}

// interceptStreamDelta consults the stream interceptors over the assistant text
// accumulated so far. The first abort wins; the rest are not asked. A panicking
// interceptor degrades to "no opinion" — same contract as every other dispatch
// here — and an abort with nothing to inject is ignored, because retrying
// against an unchanged conversation would just re-emit the same tokens.
func interceptStreamDelta(ctx context.Context, interceptors []StreamInterceptor, accumulated string) StreamDecision {
	for _, ic := range interceptors {
		var d StreamDecision
		if perr := safe(func() { d = ic.InterceptStreamDelta(ctx, accumulated) }); perr != nil {
			continue
		}
		if d.Abort && len(d.Inject) > 0 {
			return d
		}
	}
	return StreamDecision{}
}

// observe dispatches to the read-only observers.
func (s *extensionSet) observe(ctx context.Context, phase ObservePhase, turn int, msgs []Message) {
	for _, o := range s.observers {
		obs := o
		_ = safe(func() { obs.ObserveMessages(ctx, phase, turn, msgs) })
	}
}

// --- text bounding, shared with extensions -------------------------------
//
// Bounding a string for the model is the loop's default job, but an extension
// that takes the job over (or that renders its own model-facing text — a job
// listing, a search hit) needs the SAME rules, or two capabilities in one run
// disagree about what "bounded" means. These two are the loop's own helpers,
// exported so a plugin in another package uses the implementation rather than
// a copy of it.

// TruncateBytes trims s to at most maxBytes without splitting a UTF-8 rune,
// appending a marker when it cuts. maxBytes <= 0 disables truncation.
func TruncateBytes(s string, maxBytes int) string { return truncateBytes(s, maxBytes) }

// TruncateMiddle trims s to at most maxBytes by cutting its MIDDLE out, keeping
// head and tail verbatim with an omission marker between them. This is the
// shape tool results get: the end of a long result usually carries the signal.
// maxBytes <= 0 disables truncation.
func TruncateMiddle(s string, maxBytes int) string { return truncateMiddle(s, maxBytes) }
