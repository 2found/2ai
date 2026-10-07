package agentcore

import (
	"context"
	"errors"
	"fmt"
)

// Composition: turning a set of plugins into a runnable Agent.
//
// Capabilities themselves do NOT live here — each is its own package under
// agentcore/plugins/ (spill, jobs, observe, todo, subagent, …).
// That split is deliberate and load-bearing:
//
//   - This package stays a leaf. It knows the Plugin interface and the Registry
//     API, and nothing about which plugins exist. Adding a capability is a new
//     folder, never an edit here.
//   - A plugin is a normal package with a normal import. A plugin written
//     outside this repo is indistinguishable from a built-in one — same
//     interface, same Registry API, same standing.
//
// Go's import graph is what forces the shape: agentcore/plugins/* import
// agentcore, so agentcore cannot import them back. New(Config) therefore keeps
// its own applier (ApplyConfig below) rather than reaching for the plugin
// packages. Both write through the same exported Registry setters — the setter
// is the single implementation of every rule — and
// agentcore/plugins/preset carries a parity test proving
// Build(preset.Plugins(cfg)...) and New(cfg) produce the same agent.

// Build composes an Agent from plugins.
//
// Registration order does not affect behavior: seams are keyed and hooks are
// prioritized, so a caller may reorder the list freely. A plugin that fails
// aborts the whole composition — a half-built agent is never returned.
func Build(plugins ...Plugin) (*Agent, error) {
	r, err := BuildRegistry(plugins...)
	if err != nil {
		return nil, err
	}
	return r.build()
}

// BuildRegistry composes the registry WITHOUT building an Agent, so a caller can
// inspect the composition (Describe, Provider, Plugins) or Unload a plugin
// first. Call Agent() on the result to finish.
func BuildRegistry(plugins ...Plugin) (*Registry, error) {
	r := newRegistry()
	for _, p := range plugins {
		if p == nil {
			continue
		}
		if err := r.register(p); err != nil {
			return nil, err
		}
	}
	return r, nil
}

// Agent assembles the composed Agent from a registry.
func (r *Registry) Agent() (*Agent, error) { return r.build() }

// configPlugin applies a whole Config through the Registry API, so New(Config)
// is a plugin composition like any other rather than a second construction
// path with its own rules.
type configPlugin struct{ cfg Config }

func (configPlugin) Name() string { return "config" }

func (p configPlugin) Register(r *Registry) error { return r.ApplyConfig(p.cfg) }

// ApplyConfig writes an entire Config into the registry.
//
// It is the Config-shaped front door to the same setters the granular plugins
// use. Every rule (validation, defaults, seam claiming, undo) lives in the
// setter, so this function is pure field routing: if it and a plugin package
// ever disagree, the disagreement is about WHICH field, never about what the
// field means.
//
// Only non-zero fields are applied, so a Config leaves unclaimed the seams it
// says nothing about — which is what lets ApplyConfig be combined with extra
// plugins in one composition.
func (r *Registry) ApplyConfig(cfg Config) error {
	for _, p := range []Plugin{
		ModelPlugin{
			Provider:          cfg.Provider,
			NativeProvider:    cfg.NativeProvider,
			Model:             cfg.Model,
			Capabilities:      cfg.ModelCapabilities,
			ContextWindow:     cfg.ContextWindow,
			Escalation:        cfg.Escalation,
			Retry:             cfg.Retry,
			RefreshKey:        cfg.RefreshKey,
			MaxTokens:         cfg.MaxTokens,
			ReasoningEffort:   cfg.ReasoningEffort,
			OutputSchema:      cfg.OutputSchema,
			ToolChoice:        cfg.ToolChoice,
			ParallelToolCalls: cfg.ParallelToolCalls,
			PromptCacheKey:    cfg.PromptCacheKey,
			CacheRetention:    cfg.PromptCacheRetention,
		},
		DefinitionPlugin{Definition: cfg.Definition, Limits: cfg.Limits, Env: cfg.Env},
		ToolsFromSet(cfg.Tools),
		PolicyPlugin{Policy: cfg.Policy},
		HooksPlugin{Priority: PriorityDefault, Hooks: cfg.Hooks},
		BudgetPlugin{Gate: cfg.BudgetGate, Step: cfg.StepGate},
		SessionPlugin{
			Store:             cfg.Session,
			ID:                cfg.SessionID,
			Resume:            cfg.ResumeSession,
			SeedDisabledTools: cfg.SeedDisabledTools,
			ProviderState:     cfg.ProviderSession,
			ProviderSessionID: cfg.ProviderSessionID,
		},
		CompactionPlugin{
			Settings: cfg.Compaction,
			Provider: cfg.CompactionProvider,
			Model:    cfg.CompactionModel,
			Strategy: cfg.Compactor,
		},
		SteeringPlugin{
			Steer:           cfg.GetSteeringMessages,
			FollowUp:        cfg.GetFollowUpMessages,
			PrepareNextTurn: cfg.PrepareNextTurn,
		},
	} {
		if err := p.Register(r); err != nil {
			return err
		}
	}
	if err := setIf(cfg.Goal != "", func() error { return r.SetGoal(cfg.Goal) }); err != nil {
		return err
	}
	if err := setIf(cfg.Memory != nil, func() error { return r.SetMemory(cfg.Memory) }); err != nil {
		return err
	}
	for _, f := range cfg.Extensions {
		if f != nil {
			r.AddExtension(f)
		}
	}
	return nil
}

// UsePolicy installs the permission gate AND the BeforeToolCall hook that
// enforces it, at PriorityGate so it is consulted before any consumer hook
// whatever order plugins were listed in.
//
// This pairing is the whole trust boundary, so it is one call: a composition
// cannot install the policy and forget the hook that reads it.
func (r *Registry) UsePolicy(p Policy) error {
	if p == nil {
		p = DenyAll{}
	}
	if err := r.SetPolicy(p); err != nil {
		return err
	}
	r.AddHooks(PriorityGate, Hooks{
		Before: []BeforeToolCall{func(ctx context.Context, call ToolCall) Decision { return p.Allow(ctx, call) }},
	})
	return nil
}

// setIf applies fn only when cond holds, so an unset Config field leaves its
// seam unclaimed rather than claiming it with a zero value.
func setIf(cond bool, fn func() error) error {
	if !cond {
		return nil
	}
	return fn()
}

// toolsOf flattens a ToolSet into a slice in registration order, so the model's
// advertised tool order survives composition unchanged.
func toolsOf(ts *ToolSet) []Tool {
	names := ts.Names()
	out := make([]Tool, 0, len(names))
	for _, name := range names {
		if t, ok := ts.Get(name); ok {
			out = append(out, t)
		}
	}
	return out
}

// build turns the composed registry into a runnable Agent.
func (r *Registry) build() (*Agent, error) {
	if r.provider == nil && r.nativeProvider == nil {
		return nil, errors.New("agentcore: provider is required")
	}
	if r.model == "" {
		return nil, errors.New("agentcore: model is required")
	}
	outputValidator, err := compileOutputValidator(r.outputSchema)
	if err != nil {
		return nil, err
	}
	if err := r.toolChoice.Validate(); err != nil {
		return nil, fmt.Errorf("agentcore: invalid tool choice: %w", err)
	}
	var parallelToolCalls *bool
	if r.parallelToolCalls != nil {
		value := *r.parallelToolCalls
		parallelToolCalls = &value
	}

	a := &Agent{
		extensions:         r.extensions,
		nativeProvider:     r.nativeProvider,
		provider:           r.provider,
		model:              r.model,
		tools:              r.tools,
		policy:             r.policy,
		hooks:              r.mergedHooks(),
		memory:             r.memory,
		def:                r.definition,
		limits:             r.limits,
		env:                r.env,
		compaction:         r.compaction,
		compactor:          r.compactor,
		compactionProvider: r.compactionRung.Provider,
		compactionModel:    r.compactionRung.Model,
		refreshKey:         r.refreshKey,
		escalation:         append([]ModelRung(nil), r.escalation...),
		contextWindow:      r.contextWindow,
		modelCapabilities:  r.modelCapabilities,
		getSteering:        r.getSteering,
		getFollowUp:        r.getFollowUp,
		goal:               r.goal,
		prepareNextTurn:    r.prepareNextTurn,
		budgetGate:         r.budgetGate,
		session:            r.session,
		sessionID:          r.sessionID,
		providerSession:    r.providerSession,
		providerSessionID:  r.providerSessionID,
		resumeSession:      r.promptResume,
		stepGate:           r.stepGate,
		retry:              r.retry,
		cacheKey:           r.cacheKey,
		cacheRetention:     r.cacheRetention,
		seedDisabledTools:  r.seedDisabled,
		maxTokens:          r.maxTokens,
		reasoningEffort:    r.reasoningEffort,
		outputSchema:       r.outputSchema,
		outputValidator:    outputValidator,
		toolChoice:         r.toolChoice,
		parallelToolCalls:  parallelToolCalls,
	}
	return a, nil
}

// The bounds of one run.
//
// Limits are read by the loop every turn and handed to extensions through
// RunInfo, so a capability can size its own behavior to the run it is in
// without asking the Agent. They are core because there is no composition in
// which a run is unbounded.

// Limits bound a run so long autonomous loops stay safe and cheap (§7).
type Limits struct {
	MaxTurns         int // hard cap on LLM calls
	MaxToolCalls     int // hard cap on tool executions across the run
	MaxToolResultLen int // byte cap per tool result before it reaches the LLM
	MaxContextTokens int // soft budget; old turns are compacted above it (§5.2)
}

// DefaultLimits are the caps a run gets when nobody says otherwise.
//
// The turn/tool numbers are measured, not guessed: at MaxTurns 12 an analytics
// agent answering a real question ("which feature drives retention?") ran out
// of budget on 2 of 3 first-run attempts — schema discovery, a couple of
// exploratory queries and one correction already spend a dozen turns before the
// answer is written. 24 turns / 40 tool calls clears that with room for a wrong
// turn.
//
// The graceful wrap-up turn a ceiling now triggers is a floor, not a substitute
// for enough budget: it buys an honest partial answer, never the answer. Size
// the budget so the wrap-up stays the exception.
func DefaultLimits() Limits {
	return Limits{MaxTurns: 24, MaxToolCalls: 40, MaxToolResultLen: defaultMaxToolResultBytes, MaxContextTokens: defaultContextTokenBudget}
}
