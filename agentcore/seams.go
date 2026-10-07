package agentcore

import (
	"context"
	"github.com/2found/2ai/ai"
)

// ConfigPlugin adapts a flat Config into a Plugin that writes every configured
// seam to a Registry through ApplyConfig.
func ConfigPlugin(cfg Config) Plugin {
	return configPlugin{cfg: cfg}
}

// ModelPlugin installs the model spine into a Registry.
type ModelPlugin struct {
	NativeProvider    *ai.FallbackProvider
	Provider          LLMProvider
	Model             string
	Capabilities      ModelCapabilities
	ContextWindow     int
	Escalation        []ModelRung
	Retry             *RetryPolicy
	RefreshKey        func(ctx context.Context, provider string) (string, error)
	MaxTokens         int
	ReasoningEffort   string
	OutputSchema      *OutputSchema
	ToolChoice        ToolChoice
	ParallelToolCalls *bool
	PromptCacheKey    string
	CacheRetention    string
}

// Name identifies the plugin.
func (ModelPlugin) Name() string { return "model" }

// Register claims the model seam and decoding knobs.
func (p ModelPlugin) Register(r *Registry) error {
	if p.NativeProvider != nil {
		if err := r.SetNativeProvider(p.NativeProvider); err != nil {
			return err
		}
	}
	if err := r.SetModel(p.Provider, p.Model); err != nil {
		return err
	}
	if p.Escalation != nil {
		if err := r.SetEscalation(p.Escalation); err != nil {
			return err
		}
	}
	if p.ContextWindow > 0 {
		if err := r.SetContextWindow(p.ContextWindow); err != nil {
			return err
		}
	}
	if p.Capabilities != (ModelCapabilities{}) {
		if err := r.SetModelCapabilities(p.Capabilities); err != nil {
			return err
		}
	}
	if p.Retry != nil {
		if err := r.SetRetry(*p.Retry); err != nil {
			return err
		}
	}
	if p.RefreshKey != nil {
		if err := r.SetRefreshKey(p.RefreshKey); err != nil {
			return err
		}
	}
	if p.MaxTokens != 0 {
		if err := r.SetMaxTokens(p.MaxTokens); err != nil {
			return err
		}
	}
	if p.ReasoningEffort != "" {
		if err := r.SetReasoningEffort(p.ReasoningEffort); err != nil {
			return err
		}
	}
	if p.OutputSchema != nil {
		if err := r.SetOutputSchema(p.OutputSchema); err != nil {
			return err
		}
	}
	if p.ToolChoice != (ToolChoice{}) {
		if err := r.SetToolChoice(p.ToolChoice); err != nil {
			return err
		}
	}
	if p.ParallelToolCalls != nil {
		if err := r.SetParallelToolCalls(*p.ParallelToolCalls); err != nil {
			return err
		}
	}
	if p.PromptCacheKey != "" {
		return r.SetPromptCache(p.PromptCacheKey, p.CacheRetention)
	}
	return nil
}

// DefinitionPlugin installs identity, limits, and host environment into a Registry.
type DefinitionPlugin struct {
	Definition AgentDefinition
	Limits     *Limits
	Env        *Env
}

// Name identifies the plugin.
func (DefinitionPlugin) Name() string { return "definition" }

// Register claims the definition, limits, and env seams.
func (p DefinitionPlugin) Register(r *Registry) error {
	if !p.Definition.IsZero() {
		if err := r.SetDefinition(p.Definition); err != nil {
			return err
		}
	}
	if p.Limits != nil {
		if err := r.SetLimits(*p.Limits); err != nil {
			return err
		}
	}
	if p.Env != nil {
		if err := r.SetEnv(*p.Env); err != nil {
			return err
		}
	}
	return nil
}

// PolicyPlugin installs the permission gate into a Registry.
type PolicyPlugin struct {
	Policy Policy
}

// Name identifies the plugin.
func (PolicyPlugin) Name() string { return "policy" }

// Register claims the policy seam and registers the gate hook.
func (p PolicyPlugin) Register(r *Registry) error { return r.UsePolicy(p.Policy) }

// PolicyAllowList creates a PolicyPlugin permitting exactly the named tools.
func PolicyAllowList(names ...string) PolicyPlugin {
	return PolicyPlugin{Policy: NewAllowList(names...)}
}

// PolicyDenyAll creates a PolicyPlugin denying everything.
func PolicyDenyAll() PolicyPlugin {
	return PolicyPlugin{Policy: DenyAll{}}
}

// ToolsPlugin contributes tools to a Registry.
type ToolsPlugin struct {
	Label string
	Tools []Tool
}

// Name identifies the plugin.
func (p ToolsPlugin) Name() string {
	if p.Label != "" {
		return "tools:" + p.Label
	}
	return "tools"
}

// Register adds the tools.
func (p ToolsPlugin) Register(r *Registry) error {
	r.AddTools(p.Tools...)
	return nil
}

// ToolsOf builds a ToolsPlugin from a list of tools.
func ToolsOf(list ...Tool) ToolsPlugin { return ToolsPlugin{Tools: list} }

// ToolsFromSet builds a ToolsPlugin from an existing ToolSet.
func ToolsFromSet(ts *ToolSet) ToolsPlugin {
	if ts == nil {
		return ToolsPlugin{}
	}
	return ToolsPlugin{Tools: toolsOf(ts)}
}

// HooksPlugin contributes consumer lifecycle hooks to a Registry.
type HooksPlugin struct {
	Label    string
	Hooks    Hooks
	Priority Priority
}

// Name identifies the plugin.
func (p HooksPlugin) Name() string {
	if p.Label != "" {
		return "hooks:" + p.Label
	}
	return "hooks"
}

// Register adds the hooks.
func (p HooksPlugin) Register(r *Registry) error {
	r.AddHooks(p.Priority, p.Hooks)
	return nil
}

// HooksOf builds a HooksPlugin at PriorityDefault.
func HooksOf(h Hooks) HooksPlugin { return HooksPlugin{Hooks: h} }

// BudgetPlugin installs the per-turn spend ceiling and step gate into a Registry.
type BudgetPlugin struct {
	Gate func(ctx context.Context, u Usage) bool
	Step func(ctx context.Context, turn int) error
}

// Name identifies the plugin.
func (BudgetPlugin) Name() string { return "budget" }

// Register claims the budget and step gates.
func (p BudgetPlugin) Register(r *Registry) error {
	if p.Gate != nil {
		if err := r.SetBudgetGate(p.Gate); err != nil {
			return err
		}
	}
	if p.Step != nil {
		return r.SetStepGate(p.Step)
	}
	return nil
}

// SessionPlugin installs durable session logging into a Registry.
type SessionPlugin struct {
	Store             SessionStore
	ID                string
	Resume            bool
	SeedDisabledTools []string
	ProviderState     *ProviderSession
	ProviderSessionID string
}

// Name identifies the plugin.
func (SessionPlugin) Name() string { return "session" }

// Register claims the session seam.
func (p SessionPlugin) Register(r *Registry) error {
	if p.ProviderState != nil {
		if err := r.SetProviderSession(p.ProviderState, p.ProviderSessionID); err != nil {
			return err
		}
	}
	if len(p.SeedDisabledTools) > 0 {
		if err := r.SetSeedDisabledTools(p.SeedDisabledTools); err != nil {
			return err
		}
	}
	if p.Store == nil || p.ID == "" {
		return nil
	}
	return r.SetSession(p.Store, p.ID, p.Resume)
}

// CompactionPlugin installs context compaction settings and strategy into a Registry.
type CompactionPlugin struct {
	Settings *CompactionSettings
	Provider LLMProvider
	Model    string
	Strategy Compactor
}

// CompactionUsing swaps the compaction strategy, keeping the default retention
// policy.
func CompactionUsing(c Compactor) CompactionPlugin { return CompactionPlugin{Strategy: c} }

// Name identifies the plugin.
func (CompactionPlugin) Name() string { return "compaction" }

// Register claims the compaction seams.
func (p CompactionPlugin) Register(r *Registry) error {
	if p.Settings != nil {
		if err := r.SetCompaction(*p.Settings); err != nil {
			return err
		}
	}
	if p.Strategy != nil {
		if err := r.SetCompactor(p.Strategy); err != nil {
			return err
		}
	}
	if p.Provider != nil && p.Model != "" {
		return r.SetCompactionModel(p.Provider, p.Model)
	}
	return nil
}

// SteeringPlugin installs mid-run steering queues into a Registry.
type SteeringPlugin struct {
	Steer           func(ctx context.Context) []Message
	FollowUp        func(ctx context.Context) []Message
	PrepareNextTurn func(ctx context.Context, state TurnState) TurnState
}

// Name identifies the plugin.
func (SteeringPlugin) Name() string { return "steering" }

// Register claims the steering seams.
func (p SteeringPlugin) Register(r *Registry) error {
	if p.Steer != nil {
		if err := r.SetSteeringSource(p.Steer); err != nil {
			return err
		}
	}
	if p.FollowUp != nil {
		if err := r.SetFollowUpSource(p.FollowUp); err != nil {
			return err
		}
	}
	if p.PrepareNextTurn != nil {
		return r.SetPrepareNextTurn(p.PrepareNextTurn)
	}
	return nil
}
