// Package memory installs the working-memory store used for recall across runs,
// and the model-facing curation tools that let the agent revise what it
// remembered: `learn` captures a reusable lesson, `memory_edit` updates or
// retracts a stored entry by id.
package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

// Plugin installs the memory store. A nil Store leaves the agent memoryless,
// which disables recall and persistence rather than failing.
type Plugin struct {
	Store        agentcore.MemoryStore
	Consolidator Consolidator
	// NativeProvider binds consolidation to the owning agent with accounted usage.
	NativeProvider       *ai.FallbackProvider
	OnConsolidationError func(context.Context, error)
	// Worker defers model consolidation to a host-owned, bounded worker.
	// Evidence is still staged durably before the primary run returns.
	Worker *ConsolidationWorker
}

// Name identifies the plugin.
func (Plugin) Name() string { return "memory" }

// Register claims the memory seam and installs the curation extension.
func (p Plugin) Register(r *agentcore.Registry) error {
	r.AddExtension(p)
	if p.Store == nil {
		return nil
	}
	return r.SetMemory(p.Store)
}

// BeginRun builds the run's curation extension.
//
// Without a store there is nothing to curate, so the plugin DECLINES rather
// than advertising tools that can only ever fail — a tool the model will call,
// wait for, and learn nothing from is worse than no tool.
func (p Plugin) BeginRun(_ context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	if p.Store == nil {
		return nil, nil
	}
	ext := &curation{store: p.Store, scopeID: info.ScopeID}
	ext.curator, _ = p.Store.(agentcore.MemoryCurator)
	ext.consolidation, _ = p.Store.(ConsolidationStore)
	if info.Depth == 0 {
		ext.consolidator = p.Consolidator
		if ext.consolidator == nil && p.NativeProvider != nil {
			accounting := info
			if p.Worker != nil {
				// Background usage belongs to its own telemetry span, never a
				// completed agent's mutable usage/goal accounting.
				accounting = agentcore.RunInfo{ScopeID: info.ScopeID}
			}
			ext.consolidator = nativeConsolidator(p.NativeProvider, accounting)
		}
		ext.worker = p.Worker
	}
	ext.onConsolidationError = p.OnConsolidationError
	return ext, nil
}

// curation is one run's memory-curation capability.
type curation struct {
	worker               *ConsolidationWorker
	consolidation        ConsolidationStore
	consolidator         Consolidator
	onConsolidationError func(context.Context, error)
	store                agentcore.MemoryStore
	curator              agentcore.MemoryCurator // nil when the store cannot revise entries
	scopeID              string
}

// Name identifies the extension in composition diagnostics.
func (*curation) Name() string { return "memory" }

// Tools contributes the curation tools. learn needs only Remember; memory_edit
// is offered only when the store implements MemoryCurator — a store that can
// recall but not revise gets the capture tool without the edit tool.
func (c *curation) Tools() []agentcore.Tool {
	tools := []agentcore.Tool{&learnTool{store: c.store, scopeID: c.scopeID}, &recallTool{store: c.store, scopeID: c.scopeID}}
	if c.curator != nil {
		tools = append(tools, &memoryEditTool{curator: c.curator, scopeID: c.scopeID})
	}
	return tools
}

// ToolLearn is the model-facing lesson-capture tool, and the name a policy
// must permit.
const ToolLearn = "learn"

// learnTool persists a reusable lesson as a memory entry. It is the
// model-facing half of what the reflection pass does after a run: the pass
// infers lessons from the trace, this tool lets the agent file one the moment
// it is learned. Kind is pinned to learning — a fact or outcome goes through
// the consumer's own write path (the `remember` operation), not this tool.
type learnTool struct {
	store   agentcore.MemoryStore
	scopeID string
}

func (t *learnTool) Name() string { return ToolLearn }

// Bookkeeping: filing a lesson is self-management, not task progress, so a
// turn spent only on learn calls is refunded against MaxTurns.
func (t *learnTool) Bookkeeping() bool { return true }

func (t *learnTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolLearn,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Save a reusable lesson to long-term memory for future runs — a pitfall, a workaround, " +
			"a convention this project follows. Use it when you learn something that would still be true and " +
			"useful next session; do not use it for task state or one-off facts.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"lesson": map[string]any{
					"type":        "string",
					"description": "The lesson to persist, phrased so a future run can apply it.",
				},
				"tags": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Optional keywords the lesson should be recalled under.",
				},
			},
			"required": []string{"lesson"},
		},
	}
}

func (t *learnTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Lesson string   `json:"lesson"`
		Tags   []string `json:"tags"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if strings.TrimSpace(in.Lesson) == "" {
		return "", errors.New("learn: lesson is required")
	}
	// The fence: the scope comes from the run, never from the model, so an
	// agent can only ever write into its own memory.
	if err := t.store.Remember(ctx, agentcore.MemoryEntry{
		ScopeID: t.scopeID, Kind: agentcore.MemoryLearning,
		Content: in.Lesson, Tags: in.Tags, Confidence: 0.7,
	}); err != nil {
		return "", err
	}
	return "Lesson saved to memory.", nil
}

// ToolMemoryEdit is the model-facing curation tool, and the name a policy
// must permit.
const ToolMemoryEdit = "memory_edit"

// memoryEditTool revises one stored memory by id: update rewrites its content
// (the old row is kept, superseded by the new one), forget and invalidate both
// retract it — forget for a memory that is no longer worth keeping, invalidate
// for one that turned out wrong. All three are soft: the row stays in the
// store and is filtered out of recall, so the history of having held the
// belief survives the retraction.
type memoryEditTool struct {
	curator agentcore.MemoryCurator
	scopeID string
}

func (t *memoryEditTool) Name() string { return ToolMemoryEdit }

func (t *memoryEditTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolMemoryEdit,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Update or retract one of your own long-term memories by id. " +
			"'update' replaces the memory's content (the old version is kept as history); " +
			"'forget' retracts a memory that is no longer worth keeping; " +
			"'invalidate' retracts one that turned out to be wrong. " +
			"Retraction is soft — the memory stops appearing in recall but is never deleted.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id": map[string]any{
					"type":        "string",
					"description": "The id of the memory to edit, as shown in recalled entries.",
				},
				"action": map[string]any{
					"type":        "string",
					"enum":        []string{"update", "forget", "invalidate"},
					"description": "update: replace the content; forget: retract as no longer useful; invalidate: retract as wrong.",
				},
				"content": map[string]any{
					"type":        "string",
					"description": "Replacement content. Required for update, ignored otherwise.",
				},
				"tags": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Optional replacement tags for update.",
				},
			},
			"required": []string{"id", "action"},
		},
	}
}

func (t *memoryEditTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		ID      string   `json:"id"`
		Action  string   `json:"action"`
		Content string   `json:"content"`
		Tags    []string `json:"tags"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if strings.TrimSpace(in.ID) == "" {
		return "", errors.New("memory_edit: id is required")
	}
	// The fence: the scope comes from the run, never from the model, and the
	// store refuses an id outside it — a model cannot reach another agent's
	// or another scope's rows.
	switch in.Action {
	case "update":
		if strings.TrimSpace(in.Content) == "" {
			return "", errors.New("memory_edit: content is required for update")
		}
		if err := t.curator.Update(ctx, t.scopeID, in.ID, agentcore.MemoryEntry{
			ScopeID: t.scopeID, Content: in.Content, Tags: in.Tags, Confidence: 0.7,
		}); err != nil {
			return "", err
		}
		return "Memory updated.", nil
	case "forget", "invalidate":
		if err := t.curator.Supersede(ctx, t.scopeID, in.ID, ""); err != nil {
			return "", err
		}
		return "Memory retracted; it will no longer appear in recall.", nil
	default:
		return "", fmt.Errorf("memory_edit: unknown action %q (want update, forget, or invalidate)", in.Action)
	}
}
