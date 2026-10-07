// Package preset composes agentcore's default agent out of the plugin
// packages.
//
// This is the "build agentcore from its own plugins" step made literal:
// Plugins(cfg) returns the plugin list that reproduces agentcore.New(cfg), and
// preset_test.go proves the two agree field by field. If the plugin surface
// ever drifts from Config, that test fails.
//
// Full(cfg, opts) is that same list plus the capabilities Config has no field
// for — spill, jobs, the repeat guard, session retrieval, telemetry supplied by the host — and
// is what a real deployment composes.
//
// Use either as the starting point for a custom agent. Take the list, drop or
// replace the entries you want to change, append your own, and hand the result
// to agentcore.Build:
//
//	ps := preset.Full(cfg, preset.Options{Spill: myStore})
//	ps = append(ps, myCapability{})
//	agent, err := agentcore.Build(ps...)
//
// Plugins configure the native engine through hooks and extensions.
package preset

import (
	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/advisor"
	"github.com/2found/2ai/agentcore/plugins/ask"
	"github.com/2found/2ai/agentcore/plugins/finishguard"
	"github.com/2found/2ai/agentcore/plugins/goal"
	"github.com/2found/2ai/agentcore/plugins/jobs"
	"github.com/2found/2ai/agentcore/plugins/memory"
	"github.com/2found/2ai/agentcore/plugins/repeatguard"
	"github.com/2found/2ai/agentcore/plugins/sessionquery"
	"github.com/2found/2ai/agentcore/plugins/spill"
	"github.com/2found/2ai/agentcore/plugins/subagent"
	"github.com/2found/2ai/agentcore/plugins/todo"
)

// Plugins returns the plugin set that reproduces agentcore.New(cfg).
//
// Core seams are provided by agentcore.*Plugin adapters in seams.go;
// capabilities and extensions come from their respective plugin packages.
func Plugins(cfg agentcore.Config) []agentcore.Plugin {
	list := []agentcore.Plugin{
		// Model transport and native fallback.
		agentcore.ModelPlugin{
			Provider:          cfg.Provider,
			NativeProvider:    cfg.NativeProvider,
			Model:             cfg.Model,
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
		agentcore.DefinitionPlugin{Definition: cfg.Definition, Limits: cfg.Limits, Env: cfg.Env},

		// tools + governance
		agentcore.ToolsFromSet(cfg.Tools),
		agentcore.PolicyPlugin{Policy: cfg.Policy},
		agentcore.HooksOf(cfg.Hooks),
		goal.Plugin{Goal: cfg.Goal},
		agentcore.BudgetPlugin{Gate: cfg.BudgetGate, Step: cfg.StepGate},

		// durability + context
		agentcore.SessionPlugin{
			Store:             cfg.Session,
			ID:                cfg.SessionID,
			Resume:            cfg.ResumeSession,
			SeedDisabledTools: cfg.SeedDisabledTools,
		},
		memory.Plugin{Store: cfg.Memory},
		agentcore.CompactionPlugin{Settings: cfg.Compaction, Provider: cfg.CompactionProvider, Model: cfg.CompactionModel, Strategy: cfg.Compactor},

		// steering
		agentcore.SteeringPlugin{
			Steer:           cfg.GetSteeringMessages,
			FollowUp:        cfg.GetFollowUpMessages,
			PrepareNextTurn: cfg.PrepareNextTurn,
		},
	}

	// Extensions are pass-through: Config carries them already built (they are
	// values from the plugin packages), and every one is an agentcore.Plugin
	// too, so the composition is one flat list. Nothing here inspects what an
	// extension does — that is precisely the property the split bought.
	for _, f := range cfg.Extensions {
		if f == nil {
			continue
		}
		if p, ok := f.(agentcore.Plugin); ok {
			list = append(list, p)
			continue
		}
		list = append(list, extensionPlugin{f})
	}
	return list
}

// Options configures the capabilities Full adds on top of Plugins. Every field
// is optional and every zero value degrades to "that capability is not
// installed" rather than to a broken one — a partially wired composition must
// lose a feature, never gain a silently wrong one.
type Options struct {
	// Plan pins and checkpoints this run's checklist; forks get an isolated plan.
	Plan *todo.Store
	// Subagents enables governed delegation with these host-selected bounds.
	Subagents *subagent.Plugin
	// NativeAdvisor reviews finished work with cfg.NativeProvider and accounts usage.
	NativeAdvisor  bool
	AdvisorOptions advisor.Plugin
	// GoalLifecycle opts into durable lifecycle controls; optional budgets are absolute.
	GoalLifecycle *goal.Plugin
	// ConsolidateMemory enables native distillation when Memory supports ConsolidationStore.
	ConsolidateMemory bool
	// MemoryWorker moves consolidation off the primary run when supplied.
	MemoryWorker *memory.ConsolidationWorker
	// Interactive exposes ask; the host must publish questions and resume answers.
	Interactive bool
	// NativeHistory enables retrieval from opaque native checkpoints.
	NativeHistory bool
	// FinishGuard supplies the host's completion evidence rule, if any.
	FinishGuard finishguard.Guard
	// Spill persists an oversized tool result and hands the model a locator for
	// the rest. nil leaves spilling OFF (the loop's head+tail truncation stands),
	// which is the honest default: the locator is written to the durable session
	// log, so a store that cannot outlive the process would mint locators that a
	// resumed run reads back as not-found. Supply a durable store to turn it on.
	Spill spill.SpillStore
	// Jobs owns background work. nil uses a fresh in-process store per run —
	// correct for a single server, since a job cannot outlive the run that
	// started it anyway.
	Jobs jobs.JobStore
	// SessionQuery backs the session_query tool. nil searches the run's OWN
	// durable log, which needs no external index.
	SessionQuery sessionquery.SessionQuery
}

// Full is the default agent plus the capabilities that agentcore.Config has no
// field for: retrieval over the run's own history, background work, oversized
// output, the repeat-loop nudge, and native lifecycle hooks.
//
// The split from Plugins is the point. Plugins is pinned to agentcore.New(cfg)
// parity — every entry there answers to a Config field, and preset_test proves
// the two cannot drift. The five below answer to no Config field at all; they
// are plugins in the sense the architecture means, added to a list. Full is
// where a real deployment starts; Plugins is where the parity proof lives.
//
// Everything here is still ejectable: preset.Without(preset.Full(cfg, o),
// "spill") is a complete agent minus spilling, with no trace of it left.
func Full(cfg agentcore.Config, o Options) []agentcore.Plugin {
	list := Plugins(cfg)
	if o.GoalLifecycle != nil {
		g := *o.GoalLifecycle
		g.Goal = cfg.Goal
		g.Lifecycle = true
		for i, p := range list {
			if p.Name() == g.Name() {
				list[i] = g
				break
			}
		}
	}
	if o.ConsolidateMemory && cfg.Memory != nil {
		provider := cfg.NativeProvider
		if provider != nil && cfg.Retry != nil {
			copy := *provider
			copy.Retry = *cfg.Retry
			provider = &copy
		}
		for i, p := range list {
			if p.Name() == "memory" {
				list[i] = memory.Plugin{Store: cfg.Memory, NativeProvider: provider, Worker: o.MemoryWorker}
				break
			}
		}
	}
	if o.Interactive {
		list = append(list, ask.Plugin{})
	}
	if o.Spill != nil {
		list = append(list, spill.To(o.Spill))
	}
	list = append(list,
		jobs.Plugin{Store: o.Jobs},
		repeatguard.Default(),
		sessionquery.Plugin{Provider: o.SessionQuery, Native: o.NativeHistory},
	)
	if o.Plan != nil {
		list = append(list, todo.With(o.Plan))
	}
	if o.Subagents != nil {
		list = append(list, *o.Subagents)
	}
	if o.FinishGuard != nil {
		list = append(list, finishguard.Of(o.FinishGuard))
	}
	if o.NativeAdvisor {
		provider := cfg.NativeProvider
		if provider != nil && cfg.Retry != nil {
			copy := *provider
			copy.Retry = *cfg.Retry
			provider = &copy
		}
		list = append(list, advisor.NativeWithOptions(provider, o.AdvisorOptions))
	}
	return list
}

// extensionPlugin adapts a bare ExtensionFactory into a Plugin, for a consumer
// that implemented the run interface without also implementing Register.
type extensionPlugin struct{ f agentcore.ExtensionFactory }

func (p extensionPlugin) Name() string { return p.f.Name() }

func (p extensionPlugin) Register(r *agentcore.Registry) error {
	r.AddExtension(p.f)
	return nil
}

// New builds an agent from the default plugin set. It is agentcore.New spelled
// through the plugin packages.
func New(cfg agentcore.Config) (*agentcore.Agent, error) {
	return agentcore.Build(Plugins(cfg)...)
}

// Without removes the named plugins from a list, so a caller can replace one
// without rebuilding the rest by hand. Names that are not present are ignored —
// dropping something already absent is not an error.
func Without(list []agentcore.Plugin, names ...string) []agentcore.Plugin {
	drop := make(map[string]bool, len(names))
	for _, n := range names {
		drop[n] = true
	}
	out := make([]agentcore.Plugin, 0, len(list))
	for _, p := range list {
		if !drop[p.Name()] {
			out = append(out, p)
		}
	}
	return out
}

// Replace swaps the plugin with the same name as p, appending it when no such
// plugin is present. Use it to change one seam and keep the rest of a
// composition intact.
func Replace(list []agentcore.Plugin, p agentcore.Plugin) []agentcore.Plugin {
	out := Without(list, p.Name())
	return append(out, p)
}
