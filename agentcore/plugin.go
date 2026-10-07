package agentcore

import (
	"context"
	"errors"
	"fmt"
	"github.com/2found/2ai/ai"
	"slices"
	"sort"
	"strings"
)

// agentcore composes an agent from plugins.
//
// Host capabilities — the tool registry, the
// permission gate, durability, compaction, spill, jobs, retrieval and guards —
// is contributed by a Plugin that registers into a
// Registry. Plugins configure capabilities around the engine; they do not
// replace its turn loop.
//
// This is deepseek-harness's "everything is a plugin" idea (it runs on Cordis)
// rebuilt for Go. What is deliberately NOT copied is the dependency-injection
// framework: no reflection, no service container, no lifecycle graph. A plugin
// here is a value with a Register method, a service is a typed field, and
// composition is a function call — so the whole mechanism is readable in one
// file and costs nothing at run time.
//
// Two properties are worth the machinery, and they are the ones Cordis exists
// for:
//
//  1. Replaceability. A seam has exactly one provider and a plugin that
//     registers it declares so. Swapping the model, the spill store, or the
//     permission gate is a one-line change to a plugin list, not a fork of the
//     loop.
//  2. Reversibility. Registrations are effects owned by the plugin that made
//     them, so Unload unwinds a plugin cleanly. That is what makes a plugin
//     testable in isolation and a composition safe to build up and tear down.

// Plugin contributes capabilities to a Registry.
//
// A plugin must be idempotent with respect to its own configuration and must
// not depend on registration ORDER for correctness: services are keyed, and
// extension points are ordered by explicit Priority, not by when a plugin
// happened to run. Order-dependence is what makes plugin systems fragile, so
// the Registry refuses to provide it.
type Plugin interface {
	// Name identifies the plugin. Two plugins with the same name in one
	// composition is a configuration error, not a silent overwrite.
	Name() string
	// Register contributes services and extension points. Returning an error
	// aborts the whole composition — a plugin that cannot install its capability
	// must not leave a half-built agent behind.
	Register(r *Registry) error
}

// Priority orders listeners on an extension point. Lower runs first. The
// permission gate uses PriorityGate so it is always consulted before any
// consumer hook, whatever order plugins were listed in.
type Priority int

const (
	// PriorityGate is for authorization: it must run before anything that could
	// observe or rewrite a call it would have blocked.
	PriorityGate Priority = -100
	// PriorityDefault is ordinary consumer behavior.
	PriorityDefault Priority = 0
	// PriorityLate is for observers that want to see the final decision.
	PriorityLate Priority = 100
)

// Registry is the composition surface. Plugins write to it; Build reads from it.
//
// It is not a general service locator: the seams are named fields, so a missing
// or duplicated provider is a compile-time-shaped error rather than a runtime
// lookup failure, and reading the registry needs no type assertions.
type Registry struct {
	// current is the plugin being registered, used to attribute effects.
	current string
	// effects records what each plugin registered, so Unload can unwind it.
	effects map[string][]func()
	// order is plugin registration order, for diagnostics.
	order []string

	// --- capability seams: exactly one provider each -----------------------
	seams map[string]string // seam name -> plugin that claimed it

	provider          LLMProvider
	nativeProvider    *ai.FallbackProvider
	model             string
	tools             *ToolSet
	policy            Policy
	memory            MemoryStore
	session           SessionStore
	sessionID         string
	providerSession   *ProviderSession
	providerSessionID string

	// --- configuration contributed by plugins ------------------------------
	limits Limits
	// env configures the host's tool execution boundary. Its infrastructure
	// capabilities are not independently registered agent plugins.
	env               Env
	definition        AgentDefinition
	compaction        CompactionSettings
	compactionRung    ModelRung
	compactor         Compactor
	escalation        []ModelRung
	contextWindow     int
	modelCapabilities ModelCapabilities
	retry             RetryPolicy
	goal              string
	maxTokens         int
	reasoningEffort   string
	outputSchema      *OutputSchema
	toolChoice        ToolChoice
	parallelToolCalls *bool
	cacheKey          string
	cacheRetention    string
	promptResume      bool
	seedDisabled      []string

	budgetGate      func(ctx context.Context, u Usage) bool
	stepGate        func(ctx context.Context, turn int) error
	prepareNextTurn func(ctx context.Context, state TurnState) TurnState
	getSteering     func(ctx context.Context) []Message
	getFollowUp     func(ctx context.Context) []Message
	extensions      []ExtensionFactory
	refreshKey      func(ctx context.Context, provider string) (string, error)

	// --- extension points: many listeners, explicitly ordered --------------
	hooks []prioritizedHooks
}

// prioritizedHooks is one plugin's hook contribution with its ordering key.
type prioritizedHooks struct {
	plugin   string
	priority Priority
	seq      int // registration order, to break priority ties deterministically
	hooks    Hooks
}

// newRegistry builds an empty registry seeded with agentcore's defaults, so a
// plugin only has to override what it actually changes.
func newRegistry() *Registry {
	return &Registry{
		effects:    map[string][]func(){},
		seams:      map[string]string{},
		tools:      NewToolSet(),
		limits:     DefaultLimits(),
		env:        DefaultEnv(),
		compaction: DefaultCompactionSettings(),
		compactor:  DefaultCompactor(),
		retry:      DefaultRetryPolicy(),
		policy:     DenyAll{},
	}
}

// claim records that the current plugin is the sole provider of a seam,
// rejecting a second claim. Two plugins silently fighting over the model or
// the permission gate is precisely the failure a plugin system must not have.
func (r *Registry) claim(seam string) error {
	if owner, taken := r.seams[seam]; taken {
		return fmt.Errorf("agentcore: plugin %q cannot provide %s — plugin %q already does", r.current, seam, owner)
	}
	r.seams[seam] = r.current
	return nil
}

// onUnload registers an undo for the effect just applied.
func (r *Registry) onUnload(undo func()) {
	r.effects[r.current] = append(r.effects[r.current], undo)
}

// --- seam setters --------------------------------------------------------
//
// This is the plugin API. Everything a plugin can contribute goes through one
// of these, which is what lets plugins live in their own packages
// (agentcore/plugins/...) without reaching into Registry's unexported fields —
// and what lets a plugin outside this repo be a first-class citizen.
//
// Each seam setter returns an error when the seam is already claimed, so a
// composition mistake surfaces at Build time naming both plugins. Each also
// records its undo, so Unload restores the previous value.

// setSeam is the one implementation behind every seam setter: claim, assign,
// record the undo. Written once so a new seam cannot forget its reversibility.
func setSeam[T any](r *Registry, seam string, field *T, val T) error {
	if err := r.claim(seam); err != nil {
		return err
	}
	prev := *field
	*field = val
	r.onUnload(func() { *field = prev; delete(r.seams, seam) })
	return nil
}

// --- spine ---

// SetModel installs the primary provider and model.
func (r *Registry) SetModel(p LLMProvider, model string) error {
	if err := r.claim("model"); err != nil {
		return err
	}
	prevP, prevM := r.provider, r.model
	r.provider, r.model = p, model
	r.onUnload(func() { r.provider, r.model = prevP, prevM; delete(r.seams, "model") })
	return nil
}

// SetEscalation installs the ordered fallback ladder tried when the primary
// rung errors.
func (r *Registry) SetEscalation(rungs []ModelRung) error {
	return setSeam(r, "escalation", &r.escalation, rungs)
}

// SetContextWindow declares the primary model's input window in tokens, which
// caps the compaction budget. It is a separate seam from SetModel because the
// window is knowledge about the model rather than a way to reach it: the same
// provider serves models with different windows, and a composition that cannot
// find out leaves it at 0 rather than guessing.
func (r *Registry) SetContextWindow(tokens int) error {
	return setSeam(r, "context_window", &r.contextWindow, tokens)
}

// SetModelCapabilities installs live/configured facts for the primary model.
func (r *Registry) SetModelCapabilities(c ModelCapabilities) error {
	return setSeam(r, "model_capabilities", &r.modelCapabilities, c)
}

// SetRetry overrides the same-model backoff policy applied before escalation.
// Zero fields are filled from DefaultRetryPolicy().
func (r *Registry) SetRetry(p RetryPolicy) error {
	return setSeam(r, "retry", &r.retry, p.Normalized())
}

// SetRefreshKey installs the per-turn API-key resolver for rotating BYO
// credentials.
func (r *Registry) SetRefreshKey(fn func(ctx context.Context, provider string) (string, error)) error {
	return setSeam(r, "refresh_key", &r.refreshKey, fn)
}

// SetMaxTokens caps the model's output tokens per turn (0 = provider default).
func (r *Registry) SetMaxTokens(n int) error { return setSeam(r, "max_tokens", &r.maxTokens, n) }

// SetReasoningEffort asks reasoning models for that much thinking per turn.
func (r *Registry) SetReasoningEffort(e string) error {
	return setSeam(r, "reasoning_effort", &r.reasoningEffort, e)
}

// SetOutputSchema constrains every text answer to a JSON Schema at the provider.
func (r *Registry) SetOutputSchema(s *OutputSchema) error {
	return setSeam(r, "output_schema", &r.outputSchema, s)
}

// SetToolChoice configures provider-neutral tool routing for ordinary turns.
func (r *Registry) SetToolChoice(choice ToolChoice) error {
	if err := choice.Validate(); err != nil {
		return err
	}
	return setSeam(r, "tool_choice", &r.toolChoice, choice)
}

// SetParallelToolCalls configures the optional provider generation hint.
func (r *Registry) SetParallelToolCalls(enabled bool) error {
	value := enabled
	return setSeam(r, "parallel_tool_calls", &r.parallelToolCalls, &value)
}

// SetPromptCache opts every provider call into prompt caching under key.
func (r *Registry) SetPromptCache(key, retention string) error {
	if err := r.claim("prompt_cache"); err != nil {
		return err
	}
	prevK, prevR := r.cacheKey, r.cacheRetention
	r.cacheKey, r.cacheRetention = key, retention
	r.onUnload(func() { r.cacheKey, r.cacheRetention = prevK, prevR; delete(r.seams, "prompt_cache") })
	return nil
}

// --- identity ---

// SetDefinition installs the authored persona, skills, and instructions.
func (r *Registry) SetDefinition(d AgentDefinition) error {
	return setSeam(r, "definition", &r.definition, d)
}

// SetLimits overrides the run's bounds (turns, tool calls, context, result size).
func (r *Registry) SetLimits(l Limits) error { return setSeam(r, "limits", &r.limits, l) }

// SetEnv installs the host's tool execution environment as one value.
// Sandbox backends are bound to tools by the host; credential resolution runs
// inside the executor after permission checks, not as an agent extension.
func (r *Registry) SetEnv(e Env) error { return setSeam(r, "env", &r.env, e) }

// --- governance ---

// SetPolicy installs the permission gate. The gate hook itself is contributed
// separately (at PriorityGate) by whichever plugin owns governance.
func (r *Registry) SetPolicy(p Policy) error { return setSeam(r, "policy", &r.policy, p) }

// SetGoal declares the condition under which the run may stop.
func (r *Registry) SetGoal(goal string) error { return setSeam(r, "goal", &r.goal, goal) }

// SetBudgetGate installs the per-turn spend ceiling.
func (r *Registry) SetBudgetGate(fn func(ctx context.Context, u Usage) bool) error {
	return setSeam(r, "budget_gate", &r.budgetGate, fn)
}

// SetStepGate installs the pause-before-each-turn hook.
func (r *Registry) SetStepGate(fn func(ctx context.Context, turn int) error) error {
	return setSeam(r, "step_gate", &r.stepGate, fn)
}

// --- durability and context ---

// SetSession installs durability.
func (r *Registry) SetSession(s SessionStore, id string, resume bool) error {
	if err := r.claim("session"); err != nil {
		return err
	}
	prevS, prevID, prevR := r.session, r.sessionID, r.promptResume
	r.session, r.sessionID, r.promptResume = s, id, resume
	r.onUnload(func() {
		r.session, r.sessionID, r.promptResume = prevS, prevID, prevR
		delete(r.seams, "session")
	})
	return nil
}

// SetProviderSession installs provider-private state for a logical
// conversation. It is separate from SetSession because provider affinity spans
// run logs and remains useful when durable logging is disabled.
func (r *Registry) SetProviderSession(s *ProviderSession, id string) error {
	if s == nil {
		return fmt.Errorf("agentcore: plugin %q cannot install a nil provider session", r.current)
	}
	if err := r.claim("provider_session"); err != nil {
		return err
	}
	prevS, prevID := r.providerSession, r.providerSessionID
	r.providerSession, r.providerSessionID = s, id
	r.onUnload(func() {
		r.providerSession, r.providerSessionID = prevS, prevID
		delete(r.seams, "provider_session")
	})
	return nil
}

// SetSeedDisabledTools pre-disables tools for this run's circuit breaker, so a
// tool that was broken in a crashed run stays disabled across a resume.
func (r *Registry) SetSeedDisabledTools(names []string) error {
	return setSeam(r, "seed_disabled_tools", &r.seedDisabled, names)
}

// SetMemory installs the working-memory store.
func (r *Registry) SetMemory(m MemoryStore) error { return setSeam(r, "memory", &r.memory, m) }

// SetCompaction installs the transcript-shrinking retention policy (how much
// recent context survives). WHAT replaces the older span is SetCompactor.
func (r *Registry) SetCompaction(s CompactionSettings) error {
	s.PruneProtectedTools = slices.Clone(s.PruneProtectedTools)
	return setSeam(r, "compaction", &r.compaction, s)
}

// SetCompactor installs the compaction strategy. Unclaimed leaves
// DefaultCompactor (summarize the older span with a model call), so a
// composition only names this seam when it wants a different strategy.
func (r *Registry) SetCompactor(c Compactor) error {
	if c == nil {
		return fmt.Errorf("agentcore: plugin %q cannot install a nil compactor", r.current)
	}
	return setSeam(r, "compactor", &r.compactor, c)
}

// SetCompactionModel pins the compaction summary call to a dedicated tier
// instead of borrowing whichever rung the run has escalated to.
func (r *Registry) SetCompactionModel(p LLMProvider, model string) error {
	return setSeam(r, "compaction_model", &r.compactionRung, ModelRung{Provider: p, Model: model})
}

// --- steering and delegation ---

// SetSteeringSource installs the mid-run steering queue, drained at the top of
// every turn.
func (r *Registry) SetSteeringSource(fn func(ctx context.Context) []Message) error {
	return setSeam(r, "steering", &r.getSteering, fn)
}

// SetFollowUpSource installs the follow-up queue, drained when the run would
// otherwise end.
func (r *Registry) SetFollowUpSource(fn func(ctx context.Context) []Message) error {
	return setSeam(r, "follow_up", &r.getFollowUp, fn)
}

// SetPrepareNextTurn installs the per-turn save-point hook.
func (r *Registry) SetPrepareNextTurn(fn func(ctx context.Context, state TurnState) TurnState) error {
	return setSeam(r, "prepare_next_turn", &r.prepareNextTurn, fn)
}

// AddExtension contributes a run capability. Unlike a seam this is additive
// and unkeyed — several extensions may intercept the same point and they
// compose in registration order — because a capability is not a slot: two
// interceptors bounding a tool result is a waterfall, not a conflict.
//
// The registry never inspects what the factory does. It cannot: the loop
// discovers each extension's abilities by type assertion at run start, so the
// set of things a plugin may do is open, and adding a new kind of plugin does
// not touch this file.
func (r *Registry) AddExtension(f ExtensionFactory) {
	if f == nil {
		return
	}
	prev := append([]ExtensionFactory(nil), r.extensions...)
	r.extensions = append(r.extensions, f)
	r.onUnload(func() { r.extensions = prev })
}

// AddTools contributes tools to the run's registry. Unlike a seam this is
// additive: many plugins may each contribute capabilities.
func (r *Registry) AddTools(tools ...Tool) {
	prev := r.tools
	r.tools = r.tools.With(tools...)
	r.onUnload(func() { r.tools = prev })
}

// AddHooks contributes lifecycle listeners at the given priority. Listeners run
// in priority order, ties broken by registration order, so behavior never
// depends on how the plugin list happened to be sorted.
func (r *Registry) AddHooks(p Priority, h Hooks) {
	idx := len(r.hooks)
	r.hooks = append(r.hooks, prioritizedHooks{plugin: r.current, priority: p, seq: idx, hooks: h})
	r.onUnload(func() {
		for i := range r.hooks {
			if r.hooks[i].seq == idx {
				r.hooks = append(r.hooks[:i], r.hooks[i+1:]...)
				return
			}
		}
	})
}

// Plugins returns the names of registered plugins in registration order.
func (r *Registry) Plugins() []string { return append([]string{}, r.order...) }

// Provider returns the seam owner for diagnostics ("" when unclaimed).
func (r *Registry) Provider(seam string) string { return r.seams[seam] }

// Unload reverses everything a plugin registered. It is the property that makes
// a plugin composable: a plugin you can add but not remove is a patch.
func (r *Registry) Unload(name string) error {
	undos, ok := r.effects[name]
	if !ok {
		return fmt.Errorf("agentcore: no plugin named %q is loaded", name)
	}
	// Unwind in reverse so later effects come off before the ones they built on.
	for i := len(undos) - 1; i >= 0; i-- {
		undos[i]()
	}
	delete(r.effects, name)
	for i, n := range r.order {
		if n == name {
			r.order = append(r.order[:i], r.order[i+1:]...)
			break
		}
	}
	return nil
}

// register runs one plugin with effect attribution.
func (r *Registry) register(p Plugin) error {
	name := p.Name()
	if name == "" {
		return errors.New("agentcore: a plugin must have a name")
	}
	if _, dup := r.effects[name]; dup {
		return fmt.Errorf("agentcore: plugin %q is registered twice", name)
	}
	r.current = name
	r.effects[name] = nil
	r.order = append(r.order, name)
	if err := p.Register(r); err != nil {
		// Roll the failed plugin back so a caller inspecting the registry after a
		// failed Build never sees half of it.
		_ = r.Unload(name)
		r.current = ""
		return fmt.Errorf("agentcore: plugin %q: %w", name, err)
	}
	r.current = ""
	return nil
}

// mergedHooks flattens every contribution into one ordered Hooks value.
func (r *Registry) mergedHooks() Hooks {
	ordered := append([]prioritizedHooks{}, r.hooks...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].priority != ordered[j].priority {
			return ordered[i].priority < ordered[j].priority
		}
		return ordered[i].seq < ordered[j].seq
	})
	var out Hooks
	for _, ph := range ordered {
		h := ph.hooks
		out.Before = append(out.Before, h.Before...)
		out.After = append(out.After, h.After...)
		out.BeforeAgentStart = append(out.BeforeAgentStart, h.BeforeAgentStart...)
		out.TurnStart = append(out.TurnStart, h.TurnStart...)
		out.TurnEnd = append(out.TurnEnd, h.TurnEnd...)
		out.BeforeCompact = append(out.BeforeCompact, h.BeforeCompact...)
		out.AgentEnd = append(out.AgentEnd, h.AgentEnd...)
		out.Context = append(out.Context, h.Context...)
		out.PiContext = append(out.PiContext, h.PiContext...)
		out.BeforeProviderRequest = append(out.BeforeProviderRequest, h.BeforeProviderRequest...)
		out.MessageEnd = append(out.MessageEnd, h.MessageEnd...)
		// The strictest policy any plugin asked for wins: a plugin that treats a
		// hook failure as fatal must not be silently downgraded by one that does
		// not care.
		if h.ErrorPolicy == HookThrow {
			out.ErrorPolicy = HookThrow
		}
		if h.OnError != nil {
			prev := out.OnError
			next := h.OnError
			if prev == nil {
				out.OnError = next
			} else {
				out.OnError = func(source string, err error) { prev(source, err); next(source, err) }
			}
		}
	}
	return out
}

// Describe renders the composition for diagnostics: which plugin owns which
// seam, and how many listeners sit on the extension points. This is the Go
// equivalent of `dsh --dump-config` — the answer to "what is actually running?"
func (r *Registry) Describe() string {
	var b strings.Builder
	b.WriteString("plugins: " + strings.Join(r.order, ", ") + "\n")
	seams := make([]string, 0, len(r.seams))
	for s := range r.seams {
		seams = append(seams, s)
	}
	sort.Strings(seams)
	for _, s := range seams {
		fmt.Fprintf(&b, "seam %-14s -> %s\n", s, r.seams[s])
	}
	// Extensions are listed in composition order, which is also the order they
	// intercept in — for an unkeyed, additive registration that ordering IS the
	// diagnostic, the way seam ownership is for a keyed one.
	fmt.Fprintf(&b, "extensions: %s\n", strings.Join(extensionNames(r.extensions), ", "))
	h := r.mergedHooks()
	fmt.Fprintf(&b, "tools: %s\n", strings.Join(r.tools.Names(), ", "))
	fmt.Fprintf(&b, "hooks: before=%d after=%d context=%d turn_start=%d turn_end=%d agent_end=%d\n",
		len(h.Before), len(h.After), len(h.Context), len(h.TurnStart), len(h.TurnEnd), len(h.AgentEnd))
	return b.String()
}

// SetNativeProvider installs the native provider composition.
func (r *Registry) SetNativeProvider(p *ai.FallbackProvider) error {
	return setSeam(r, "native_provider", &r.nativeProvider, p)
}
