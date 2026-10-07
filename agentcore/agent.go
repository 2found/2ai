package agentcore

import (
	"context"
	"errors"
	"fmt"
	"github.com/2found/2ai/ai"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Agent is a configured runtime instance: a provider + model, an injected
// ToolSet, a Policy, lifecycle Hooks, an optional MemoryStore, and an
// AgentDefinition. It is product-agnostic — the same Agent type powers the
// Growth Analyst and any future consumer.
type Agent struct {
	// extensions are the per-run capability providers installed by plugins. The
	// loop dispatches to them through the interfaces in extension.go and never
	// names one, so removing a plugin removes its behavior with no core edit.
	extensions     []ExtensionFactory
	provider       LLMProvider
	nativeProvider *ai.FallbackProvider
	model          string
	tools          *ToolSet
	policy         Policy
	hooks          Hooks
	memory         MemoryStore // optional; nil disables recall/persistence
	def            AgentDefinition
	limits         Limits
	env            Env
	// compaction tunes how the loop summarizes a long transcript once it
	// approaches limits.MaxContextTokens.
	compaction CompactionSettings
	// compactionProvider + compactionModel, when both set, pin the compaction
	// summary call to a dedicated tier (e.g. a cheap "lite" rung) instead of
	// borrowing whichever rung the run has escalated to. Leaving either unset
	// preserves the default: compaction uses the active rung's model.
	compactionProvider LLMProvider
	compactionModel    string
	// compactor is the strategy that actually shrinks the transcript. Never
	// nil on a composed agent — the registry defaults it to DefaultCompactor.
	compactor Compactor
	// refreshKey, when set, re-resolves the provider's API key before each turn
	// (pi's per-turn getApiKey) so expiring BYO tokens don't kill long runs. The
	// returned key is applied only if the provider implements KeyUpdater.
	refreshKey func(ctx context.Context, provider string) (string, error)
	// escalation is the ordered fallback ladder tried when the primary
	// provider/model errors (a retryable failure, not a cancellation). It is
	// product-agnostic: a consumer's tier system maps onto it, but agentcore only
	// sees an ordered list of provider+model rungs.
	escalation []ModelRung
	// contextWindow is the primary model's input window in tokens (0 = unknown).
	// The loop treats the primary as rung zero of the ladder, so this is the same
	// fact ModelRung.ContextWindow carries for the rest.
	contextWindow int
	// modelCapabilities is a discovery snapshot for the primary rung. Known
	// values override adapter defaults; unknown fields leave them intact.
	modelCapabilities ModelCapabilities
	// getSteering, when set, is drained at the top of every turn: any messages it
	// returns are threaded into the conversation before the model reasons, so a
	// user can inject a mid-run correction honored on the next turn (pi's steering
	// queue). The consumer owns the source (channel, DB, SSE input).
	getSteering func(ctx context.Context) []Message
	// getFollowUp, when set, is drained once the model produces a final answer: if
	// it returns messages, they are appended and the loop restarts instead of
	// returning, so a conversation continues within one bounded run (pi's
	// follow-up queue).
	getFollowUp func(ctx context.Context) []Message
	// goal, when non-empty, activates the run-level goal gate (session.go): the
	// completion contract is added to the system prompt and a normal finish
	// without a STATUS: DONE / STATUS: BLOCKED sentinel re-opens the run.
	goal string
	// prepareNextTurn, when set, is called after each completed turn with the
	// current TurnState; the returned state (model / tools / system) drives the
	// next turn without mutating the in-flight one. nil keeps the run static.
	prepareNextTurn func(ctx context.Context, state TurnState) TurnState
	// budgetGate, when set, is consulted at the top of each turn with the run's
	// accumulated usage. Returning true triggers a graceful stop (#4): the loop
	// injects a final "budget exhausted — summarize and stop" user turn, strips
	// tools so the model can only write a wrap-up, and returns with StopReason
	// "budget_exhausted". The consumer owns the ceiling + spend lookup; agentcore
	// only sees the boolean verdict against the running Usage.
	budgetGate func(ctx context.Context, u Usage) bool
	// session + sessionID, when both set, make the run durable: the loop appends
	// typed entries (messages, compaction brackets, leaf) to the append-only log
	// so a crashed or compacted run can be reduced and resumed (P9). nil disables
	// durability — the run is purely in-memory.
	session   SessionStore
	sessionID string
	// providerSession is mutable provider-private state for the logical user
	// conversation. providerSessionID is the routing identity exposed on each
	// request; unlike sessionID it remains stable across per-turn run logs.
	providerSession   *ProviderSession
	providerSessionID string
	// stepGate, when set, is called at the top of every turn before any work
	// (compaction, steering, reason) happens. It blocks until the consumer permits
	// the turn to proceed, returning a non-nil error to halt the run. This is how
	// the Lab's explain mode pauses a live run before each step without changing
	// any other run behavior: gating, secret resolution, budgets, and escalation
	// all still run after the gate releases. nil (the default) never pauses, so
	// production runs are unaffected.
	stepGate func(ctx context.Context, turn int) error
	// running is the single-flight guard (pi's phase machine, reduced to a binary
	// busy/idle for our run-only surface): a run sets it via CAS and clears it on
	// exit, so a second concurrent Prompt on the same Agent fails fast with ErrBusy
	// instead of racing on the shared run state. A hook may still drive a *different*
	// Agent instance reentrantly; only the same instance is single-flighted.
	running int32
	// inbox is the durable steering/follow-up queue (pi's durable inbox):
	// Steer/FollowUp append an EntryInbox side record to the session log AND
	// queue the item here, so the next drain point delivers it — and a crash
	// before then loses nothing, because a resume re-reads the log.
	inboxMu sync.Mutex
	inbox   []InboxItem
	// retry bounds the same-model backoff retry of a transient provider failure
	// (429/5xx/network blip) before the loop escalates down the ladder, so a brief
	// outage no longer jumps straight to a pricier rung or aborts the run.
	retry RetryPolicy
	// cacheKey + cacheRetention, when cacheKey is non-empty, opt every provider call
	// in the run into prompt caching (OpenAI prompt_cache_key; Anthropic cache_control
	// on the system prefix). Empty (the default) leaves caching off, so providers and
	// compat servers that don't support it are unaffected.
	cacheKey       string
	cacheRetention string
	// seedDisabledTools pre-populates the circuit breaker's disabled set at run
	// start, so a tool disabled in a crashed run stays disabled when that run is
	// resumed (the disable survives via the durable log → RecoverSession). Empty —
	// the default — starts every tool enabled.
	seedDisabledTools []string
	// resumeSession makes the run continue the existing durable log at sessionID
	// instead of opening a fresh one: drive rebuilds history from the log,
	// replays dangling retry-safe calls with their original call IDs, and skips
	// re-persisting the seeds (doing so would double every message in the
	// reduced history). A completed log short-circuits to its recorded answer.
	resumeSession bool
	// maxTokens caps the model's output tokens per turn. 0 — the default — lets
	// the provider apply its own default (the gateway's cap), which can truncate
	// large outputs with stop_reason:"length". Set a generous value for agents
	// that emit big artifacts (long documents, full HTML pages).
	maxTokens int
	// reasoningEffort, when set, is passed through to providers that support a
	// reasoning/thinking-effort knob ("low" | "medium" | "high"; OpenAI-wire
	// reasoning_effort). Providers without the knob ignore it. Empty — the
	// default — sends nothing, so strict compat servers are unaffected.
	reasoningEffort string
	// outputSchema, when set, constrains every text answer to a JSON Schema at
	// the provider (structured outputs). outputValidator is compiled at build
	// time and enforces the same contract locally when a provider ignores it.
	// Verdict-shaped agents only.
	outputSchema      *OutputSchema
	outputValidator   outputValidator
	toolChoice        ToolChoice
	parallelToolCalls *bool
	// childUsage accumulates the usage of sub-agent runs spawned during the
	// current run (written by spawn_subagent, possibly from parallel tool
	// goroutines); runLoop folds and resets it into the RunResult so a parent
	// run's accounting includes what its children spent.
	childMu    sync.Mutex
	childUsage Usage
}

// addChildUsage folds one child run's usage into the parent's accumulator.
func (a *Agent) addChildUsage(u Usage) {
	a.childMu.Lock()
	a.childUsage.InputTokens += u.InputTokens
	a.childUsage.OutputTokens += u.OutputTokens
	a.childUsage.CacheReadTokens += u.CacheReadTokens
	a.childUsage.CacheWriteTokens += u.CacheWriteTokens
	a.childUsage.CostUSD += u.CostUSD
	a.childUsage.CostUnpriced = a.childUsage.CostUnpriced || u.CostUnpriced
	a.childMu.Unlock()
}

// takeChildUsage returns and resets the accumulated child usage.
func (a *Agent) takeChildUsage() Usage {
	a.childMu.Lock()
	u := a.childUsage
	a.childUsage = Usage{}
	a.childMu.Unlock()
	return u
}

// peekChildUsage returns the accumulated child usage WITHOUT resetting it, so the
// mid-run budget gate can meter sub-agent spend the loop has not yet folded into
// the run's RunResult (takeChildUsage does that once, on run exit).
func (a *Agent) peekChildUsage() Usage {
	a.childMu.Lock()
	u := a.childUsage
	a.childMu.Unlock()
	return u
}

// addUsage returns u plus v across every metered dimension.
func addUsage(u, v Usage) Usage {
	u.InputTokens += v.InputTokens
	u.OutputTokens += v.OutputTokens
	u.CacheReadTokens += v.CacheReadTokens
	u.CacheWriteTokens += v.CacheWriteTokens
	u.CostUSD += v.CostUSD
	u.CostUnpriced = u.CostUnpriced || v.CostUnpriced
	return u
}

// mergeUsage folds one report of the SAME turn's usage into what is known so
// far, taking the newest non-zero value of each field. Reports within a turn are
// running totals, not increments, so they are overwritten rather than summed —
// adding them would double-count a provider that restates a field it already
// sent. Across turns, use addUsage: those are genuinely separate spends.
func mergeUsage(u, v Usage) Usage {
	if v.InputTokens != 0 {
		u.InputTokens = v.InputTokens
	}
	if v.OutputTokens != 0 {
		u.OutputTokens = v.OutputTokens
	}
	if v.CacheReadTokens != 0 {
		u.CacheReadTokens = v.CacheReadTokens
	}
	if v.CacheWriteTokens != 0 {
		u.CacheWriteTokens = v.CacheWriteTokens
	}
	// CostUSD and CostUnpriced are one fact from one source (observe's pricing
	// pass) and must be overwritten together: a fresher report of "unpriced,
	// $0" (CostUSD == 0, CostUnpriced == true) is real information, and gating
	// solely on "CostUSD != 0" (as every other field above does) would silently
	// drop it, leaving a stale "priced" flag from an earlier report.
	if v.CostUSD != 0 || v.CostUnpriced {
		u.CostUSD = v.CostUSD
		u.CostUnpriced = v.CostUnpriced
	}
	return u
}

// ErrBusy is returned by a run entry point (Prompt/Continue/…) when the Agent is
// already running. One Agent instance drives one run at a time; spin up a second
// instance (or wait for the first to finish) for concurrent work.
var ErrBusy = errors.New("agentcore: agent is busy with another run")

// tryAcquire claims the single-flight slot, reporting whether it was free.
func (a *Agent) tryAcquire() bool { return atomic.CompareAndSwapInt32(&a.running, 0, 1) }

// release frees the single-flight slot for the next run.
func (a *Agent) release() { atomic.StoreInt32(&a.running, 0) }

// ModelRung is one provider+model the loop may fall back to when the rung above
// it errors. Consumers build the ladder (e.g. lite→flash→pro); agentcore just
// walks it.
type ModelRung struct {
	Provider LLMProvider
	Model    string
	// Capabilities is an optional discovery/config snapshot for this exact
	// provider/model rung. It overlays the provider adapter's defaults.
	Capabilities ModelCapabilities
	// ContextWindow is this model's input window in tokens, which caps the
	// compaction budget while this rung is answering. 0 means unknown and the
	// configured MaxContextTokens stands alone.
	//
	// It lives on the rung rather than in Limits because a ladder is routinely
	// built from models with different windows, and the loop switches between
	// them mid-run. A single run-wide number is therefore wrong for every rung
	// but one — too high and the loop never compacts before the provider
	// rejects the request, which no retry or escalation can rescue.
	//
	// agentcore does not know any model's window and must not learn: the value
	// is supplied by whoever built the rung.
	ContextWindow int
}

// TurnState is the per-turn save-point: the model, tools, and system prompt that
// will drive the next provider request. After each turn the loop hands the
// current state to a consumer's PrepareNextTurn hook, which may return a modified
// copy; the change applies to the next turn only and never mutates the in-flight
// request (pi's prepareNextTurn). Messages is supplied read-only for the hook to
// inspect — returning a different slice does not replace the loop's history.
type TurnState struct {
	Model    string
	Tools    *ToolSet
	System   string
	Messages []Message
}

// Config wires an Agent. Provider, Model, Tools, and Policy are required; the
// rest have safe defaults (DenyAll policy, no memory, DefaultLimits, DefaultEnv).
type Config struct {
	// NativeProvider executes through engine with AI-owned fallback.
	NativeProvider    *ai.FallbackProvider
	Provider          LLMProvider
	Model             string
	ModelCapabilities ModelCapabilities
	// ContextWindow is the primary model's input window in tokens — the same
	// fact ModelRung.ContextWindow carries for the escalation rungs. 0 means
	// unknown.
	ContextWindow int
	Tools         *ToolSet
	Policy        Policy
	Hooks         Hooks
	Memory        MemoryStore
	Definition    AgentDefinition
	Limits        *Limits
	Env           *Env
	// Compaction overrides the default compaction settings (recent-token budget
	// kept verbatim). nil uses DefaultCompactionSettings().
	Compaction *CompactionSettings
	// CompactionProvider + CompactionModel, when both set, pin the in-loop
	// compaction summary call to a dedicated tier instead of borrowing the run's
	// active escalation rung. Leaving either unset keeps today's behavior (the
	// active rung summarizes). The consumer's tier system maps a "compaction"
	// task kind onto these.
	CompactionProvider LLMProvider
	CompactionModel    string
	// Compactor replaces the transcript-shrinking strategy. nil keeps
	// DefaultCompactor (summarize the older span with a model call).
	Compactor Compactor
	// RefreshKey is an optional per-turn API-key resolver. It is invoked before
	// each turn with the provider name; a non-empty result is pushed into the
	// provider via KeyUpdater. Use it for short-lived / rotating BYO credentials.
	RefreshKey func(ctx context.Context, provider string) (string, error)
	// Escalation is an optional ordered fallback ladder. When the primary
	// Provider/Model errors on a turn, the loop retries that turn down the ladder
	// before giving up, then sticks with the working rung for later turns.
	Escalation []ModelRung
	// GetSteeringMessages is an optional callback drained at the top of each turn;
	// returned messages are injected before the model reasons (mid-run steering).
	// The consumer owns the source (channel, DB, SSE input) — and its durability:
	// messages it has not yet returned are lost on a crash. Agent.Steer is the
	// durable alternative: it writes the queue into the session log itself.
	GetSteeringMessages func(ctx context.Context) []Message
	// GetFollowUpMessages is an optional callback drained when the agent would
	// stop; returned messages restart the loop instead of ending the run.
	// Agent.FollowUp is the durable equivalent.
	GetFollowUpMessages func(ctx context.Context) []Message
	// Goal, when non-empty, declares the condition under which this run may
	// stop (Claude Code /goal analog; see session.go). The completion contract is
	// appended to the system prompt, and a normal finish whose answer lacks a
	// STATUS: DONE or STATUS: BLOCKED sentinel is re-opened with a keep-going
	// nudge. Uncapped, but still bounded by MaxTurns / MaxToolCalls / the
	// budget gate, and a repeated identical answer breaks the loop with
	// StopReason "goal_stalled". The goal is recorded in the durable log
	// (EntryGoal), so a resumed run stays gated even when the resuming caller
	// cannot re-supply it. Empty — the default — disables the gate.
	Goal string
	// PrepareNextTurn is an optional save-point hook called after each turn; the
	// returned TurnState (model / tools / system) drives the next turn. nil keeps
	// the model, tools, and prompt fixed for the whole run. Model and tool-set
	// changes are durable (EntryModelChange / EntryActiveToolsChange), so a
	// crash-resumed run rebuilds them; a system change is not — the prompt is
	// re-derived on every run.
	PrepareNextTurn func(ctx context.Context, state TurnState) TurnState
	// BudgetGate is an optional per-turn ceiling check (#4). Consulted with the
	// run's accumulated usage at the top of each turn; returning true triggers a
	// one-turn graceful stop that summarizes and halts. nil leaves the run
	// uncapped (bounded only by MaxTurns / MaxToolCalls).
	BudgetGate func(ctx context.Context, u Usage) bool
	// Session + SessionID enable durable, resumable runs (P9): when both are set
	// the loop appends typed entries to the append-only SessionStore. Leaving
	// either unset keeps the run in-memory only.
	Session   SessionStore
	SessionID string
	// ProviderSession and ProviderSessionID are independent of durable run
	// logging. They retain provider transport/compatibility state for a logical
	// conversation even when each user turn gets a fresh Agent and run id.
	ProviderSession   *ProviderSession
	ProviderSessionID string
	// ResumeSession, when true (with Session + SessionID set), continues the
	// existing durable log at SessionID instead of starting a fresh run: the
	// loop rebuilds history from the log (the seed messages are used only if
	// the log turns out empty), re-issues dangling retry-safe tool calls with
	// their ORIGINAL call IDs — reproducing their idempotency keys and child
	// session IDs, so a replayed spawn_subagent reattaches instead of
	// re-running — and closes the remaining dangling calls with interrupted
	// notes. A log that already reached its leaf returns its recorded final
	// answer without any provider call.
	ResumeSession bool
	// StepGate is an optional pause-before-each-turn hook. When set, the loop calls
	// it at the top of every turn and blocks until it returns; a non-nil error
	// halts the run. The Lab's explain mode uses it to step a live run; leaving it
	// nil keeps runs continuous (the production default).
	StepGate func(ctx context.Context, turn int) error
	// Retry overrides the same-model backoff policy applied before escalation. nil
	// uses DefaultRetryPolicy(); a partial override fills its zero fields from it.
	Retry *RetryPolicy
	// PromptCacheKey, when set, opts every provider call into prompt caching under
	// this key (typically the session id). PromptCacheRetention hints the window
	// ("" | "short" | "long" | "24h"). Empty key leaves caching off — the default.
	PromptCacheKey       string
	PromptCacheRetention string
	// SeedDisabledTools pre-disables the named tools for this run's circuit
	// breaker. A resume passes the tools that were disabled in the crashed run
	// (recovered from its durable log) so a persistently broken tool is not
	// retried from scratch after resume. Empty starts every tool enabled.
	SeedDisabledTools []string
	// MaxTokens caps the model's output tokens per turn. 0 lets the provider use
	// its own default. Set this for agents that emit large artifacts so the
	// gateway's default cap doesn't truncate output with stop_reason:"length".
	MaxTokens int
	// ReasoningEffort, when set ("low" | "medium" | "high"), asks reasoning
	// models to spend that much thinking effort per turn (OpenAI-wire
	// reasoning_effort). Providers without the knob ignore it; empty sends
	// nothing.
	ReasoningEffort string
	// OutputSchema, when non-nil, constrains every text answer this agent
	// produces to the given JSON Schema (grammar-constrained decoding at the
	// provider: OpenAI response_format json_schema strict, Anthropic
	// structured-outputs output_format). Meant for verdict-shaped agents —
	// moderation / classification presets that must return machine-parseable
	// verdict×confidence JSON — not general chat: any plain-text turn must fit
	// the schema. Providers without the capability ignore it, so callers still
	// validate the answer. nil — the default — leaves output free-form.
	OutputSchema *OutputSchema
	// ToolChoice optionally constrains model tool use on every ordinary turn.
	// The zero value keeps provider defaults. Required/named choices are removed
	// automatically from the loop's borrowed tool-free finalization turn.
	ToolChoice ToolChoice
	// ParallelToolCalls asks capable providers whether they may emit several
	// tool calls in one assistant turn. nil omits the hint for compatibility.
	// AgentCore still executes only tools that independently opt into ParallelTool.
	ParallelToolCalls *bool
	// Extensions are the run capabilities this agent is built with — spill,
	// background jobs, session retrieval, delegation, the repeated-call
	// reminder, verify-on-stop, log-invariant observation, and whatever a
	// consumer writes next. Each is a value from a package under
	// agentcore/plugins/, and the loop reaches them ONLY through the interfaces
	// in extension.go: it never names one, so the set here is the complete
	// answer to "what can this agent do beyond the core loop?".
	//
	// Empty — the default — is a working agent. It reasons, calls tools, obeys
	// its policy, compacts, and logs; it just has no capability that a plugin
	// would have added.
	Extensions []ExtensionFactory
}

// New constructs an Agent from a Config.
//
// It is a plugin composition like any other: the Config is applied to a
// Registry through the same exported setters the plugin packages use, and the
// Agent is built from that Registry. Config stays the ergonomic front door for
// the common case; reach for Build with plugins from agentcore/plugins/... when
// you need to REPLACE a seam (your own governance plugin)
// rather than configure one.
//
// The permission gate is installed by Registry.UsePolicy at PriorityGate, so it
// is still consulted before any consumer hook — now as a property of the hook
// ordering rather than a side effect of prepending to a slice.
func New(cfg Config) (*Agent, error) {
	return Build(ConfigPlugin(cfg))
}

// Prompt runs a single interactive turn-loop from a user message and returns
// the result. task seeds skill selection and memory recall (defaults to the
// user input).
func (a *Agent) Prompt(ctx context.Context, userInput string) (RunResult, error) {
	return a.RunNative(ctx, NativeRun{Input: []Message{{Role: RoleUser, Content: userInput, Directive: true}}, Task: userInput})
}

// PromptStream runs the same turn-loop but streams the assistant's tokens and
// tool-call traces to sink as they are produced, for a live (SSE) viewer. The
// returned RunResult is identical to Prompt's — streaming is additive.
func (a *Agent) PromptStream(ctx context.Context, userInput string, sink StreamSink) (RunResult, error) {
	return a.RunNative(ctx, NativeRun{Input: []Message{{Role: RoleUser, Content: userInput, Directive: true}}, Task: userInput, Sink: sink})
}

// Continue submits host-authored input to the native engine. To resume provider
// history, pass the prior NativeState to RunNative instead of replaying Messages.
func (a *Agent) Continue(ctx context.Context, history []Message, task string) (RunResult, error) {
	return a.RunNative(ctx, NativeRun{Input: history, Task: task})
}

// ContinueStream submits host-authored input and streams the native run.
// Provider history resumes through RunNative with its opaque checkpoint.
func (a *Agent) ContinueStream(ctx context.Context, history []Message, task string, sink StreamSink) (RunResult, error) {
	return a.RunNative(ctx, NativeRun{Input: history, Task: task, Sink: sink})
}

// Steer queues a mid-run correction (pi's steer()): the loop drains it at the
// top of the next turn and threads it into the conversation before the model
// reasons. On a durable run the message is also appended to the session log as
// an EntryInbox side record the moment it is queued, so a crash between queue
// and drain cannot lose it — a resume re-reads the log and delivers it then.
// A nil error means queued; a non-nil error means the durable write failed and
// the message is in-memory only.
func (a *Agent) Steer(ctx context.Context, m Message) error {
	return a.enqueueInbox(ctx, InboxSteer, m)
}

// FollowUp queues work for after the agent would stop (pi's followUp()): the
// loop drains it when the model produces a final answer and restarts instead
// of returning. Same durability contract as Steer.
func (a *Agent) FollowUp(ctx context.Context, m Message) error {
	return a.enqueueInbox(ctx, InboxFollow, m)
}

// enqueueInbox records one queued message: durably when the agent has a
// session, always in memory. The durable entry is a side record — it does not
// join the session tree — so it can be written mid-turn without touching the
// loop's buffered chain.
func (a *Agent) enqueueInbox(ctx context.Context, lane string, m Message) error {
	item := InboxItem{Lane: lane, Message: m}
	var err error
	if a.session != nil && a.sessionID != "" {
		e := SessionEntry{Kind: EntryInbox, Lane: lane, Message: &m, CreatedAt: time.Now()}
		if e.ID = newEntryID(); e.ID == "" {
			// crypto/rand failed: a "#<n>" fallback would mint IDs that collide
			// with the fold's "#<seq>" convention for id-less entries, so an
			// EntryInboxDone could settle the wrong intent. "inbox-<n>" keeps
			// the ID non-empty (settlement works) and out of that namespace.
			e.ID = fmt.Sprintf("inbox-%d", time.Now().UnixNano())
		}
		item.ID = e.ID
		err = a.session.Append(ctx, a.sessionID, e)
	}
	a.inboxMu.Lock()
	a.inbox = append(a.inbox, item)
	a.inboxMu.Unlock()
	return err
}

// drainInbox removes and returns the queued items for one lane. Called by the
// loop at each lane's drain point; items carry their inbox entry ID so the
// loop can settle them with EntryInboxDone.
func (a *Agent) drainInbox(lane string) []InboxItem {
	a.inboxMu.Lock()
	defer a.inboxMu.Unlock()
	var out []InboxItem
	keep := a.inbox[:0]
	for _, it := range a.inbox {
		if it.Lane == lane {
			out = append(out, it)
		} else {
			keep = append(keep, it)
		}
	}
	a.inbox = keep
	return out
}

// Describe renders what an Agent is actually configured with.
//
// The question it answers is the one that is hard to answer from a config file
// or a plugin list: after every default, every override, and every plugin, what
// does this agent actually do? Reach for it when a run behaves in a way the
// configuration does not explain — an ungated tool, compaction that never
// fires, a resume that starts fresh.
//
// It reports presence, never values, for anything that could carry a secret or
// a closure: a credential resolver, a budget gate, and a steering source each
// print as "set". Tool NAMES are printed because the model already sees them.
//
// The output is stable and ordered, so two agents can be compared by string —
// which is how agentcore/plugins/preset proves that composing from plugin
// packages produces the same agent as New(Config).
func (a *Agent) Describe() string {
	var b strings.Builder

	line := func(k string, v any) { fmt.Fprintf(&b, "%-22s %v\n", k+":", v) }
	// present renders a closure or interface as set/unset — never its value.
	present := func(k string, set bool) {
		if set {
			line(k, "set")
		} else {
			line(k, "-")
		}
	}

	line("model", a.model)
	if n := len(a.escalation); n > 0 {
		rungs := make([]string, 0, n)
		for _, r := range a.escalation {
			rungs = append(rungs, r.Model)
		}
		line("escalation", strings.Join(rungs, " → "))
	}
	line("max_tokens", a.maxTokens)
	line("reasoning_effort", orDash(a.reasoningEffort))
	present("output_schema", a.outputSchema != nil)
	choice := string(a.toolChoice.Mode)
	if a.toolChoice.Mode == ToolChoiceNamed {
		choice += ":" + a.toolChoice.Name
	}
	line("tool_choice", orDash(choice))
	if a.parallelToolCalls == nil {
		line("parallel_tool_calls", "-")
	} else {
		line("parallel_tool_calls", *a.parallelToolCalls)
	}
	line("prompt_cache", orDash(a.cacheKey))
	present("refresh_key", a.refreshKey != nil)
	line("retry", fmt.Sprintf("%d attempts", a.retry.MaxAttempts))

	// Identity and bounds.
	line("scope", orDash(a.def.ScopeID))
	line("skills", len(a.def.Skills))
	line("limits", fmt.Sprintf("turns=%d tools=%d ctx=%d result=%d",
		a.limits.MaxTurns, a.limits.MaxToolCalls, a.limits.MaxContextTokens, a.limits.MaxToolResultLen))
	present("sandbox", a.env.Sandbox != nil)

	// Governance. The policy TYPE is named rather than its rules: the rules are
	// the policy's business, and printing them here would drift.
	line("policy", fmt.Sprintf("%T", a.policy))
	line("goal", orDash(a.goal))
	present("budget_gate", a.budgetGate != nil)
	present("step_gate", a.stepGate != nil)

	// Durability and context.
	present("session", a.session != nil)
	line("session_resume", a.resumeSession)
	line("seed_disabled", strings.Join(sorted(a.seedDisabledTools), ", "))
	present("memory", a.memory != nil)
	// The compaction budget an operator configured is not the one that applies:
	// the answering model's window caps it. Print what will actually be used,
	// since a run that compacts unexpectedly early or late is diagnosed here.
	line("context_window", a.contextWindow)
	budget := effectiveBudget(a.limits.MaxContextTokens, a.contextWindow)
	compaction := effectiveCompaction(a.compaction, budget)
	line("compaction", fmt.Sprintf("keep_recent=%d budget=%d prune_cache_suffix=%d prune_min_savings=%d",
		compaction.KeepRecentTokens, budget, compaction.PruneCacheWarmSuffixTokens, compaction.PruneMinimumSavingsTokens))
	line("compactor", compactorName(a.compactor))
	line("compaction_model", orDash(a.compactionModel))

	// Extensions — the capabilities this agent has beyond the core loop. Named
	// rather than described: what each one does is its own package's README, and
	// restating it here would drift. An empty list is a complete answer, not a
	// missing one.
	line("extensions", strings.Join(extensionNames(a.extensions), ", "))

	// Steering.
	present("steering", a.getSteering != nil)
	present("follow_up", a.getFollowUp != nil)
	present("prepare_next_turn", a.prepareNextTurn != nil)

	// Surface. Tool order is the order the model is shown, so it is NOT sorted.
	line("tools", strings.Join(a.tools.Names(), ", "))
	h := a.hooks
	line("hooks", fmt.Sprintf("before=%d after=%d context=%d turn_start=%d turn_end=%d message_end=%d agent_end=%d",
		len(h.Before), len(h.After), len(h.Context), len(h.TurnStart), len(h.TurnEnd),
		len(h.MessageEnd), len(h.AgentEnd)))
	line("hook_error_policy", string(h.ErrorPolicy))

	return b.String()
}

// orDash renders an empty string as a dash so a missing value is visibly
// missing rather than an empty column.
func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

// sorted returns a sorted copy, so a set-shaped field describes identically
// regardless of the order it was configured in.
func sorted(in []string) []string {
	out := append([]string{}, in...)
	sort.Strings(out)
	return out
}

// extensionNames lists the registered factories in composition order, which is
// also the order they intercept in.
func extensionNames(fs []ExtensionFactory) []string {
	if len(fs) == 0 {
		return nil
	}
	names := make([]string, 0, len(fs))
	for _, f := range fs {
		names = append(names, f.Name())
	}
	return names
}

// compactorName names the installed compaction strategy, so "what is actually
// running?" distinguishes the built-in summarizer from a replacement.
func compactorName(c Compactor) string {
	if c == nil {
		return DefaultCompactor().Name()
	}
	return c.Name()
}
