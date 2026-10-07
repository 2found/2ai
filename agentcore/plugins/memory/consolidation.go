package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/2found/2ai/agentcore"
)

// Rollout is bounded evidence from a completed run. ID is a content digest,
// making retries idempotent even if publication and consolidation are retried.
type Rollout struct {
	ID       string              `json:"id"`
	Messages []agentcore.Message `json:"messages"`
	Final    string              `json:"final"`
}

// Change replaces/merges live entries, or retracts them when Entry is nil.
// Empty IDs adds a new lesson. Every named ID must come from the supplied snapshot.
type Change struct {
	IDs   []string               `json:"ids"`
	Entry *agentcore.MemoryEntry `json:"entry,omitempty"`
}
type Consolidation struct {
	Rollouts []Rollout               `json:"rollouts"`
	Memories []agentcore.MemoryEntry `json:"memories"`
}
type Consolidator func(context.Context, Consolidation) ([]Change, error)

// ConsolidationStore is optional. CommitConsolidation must atomically validate
// the memory snapshot, apply all changes, and mark rollouts consumed. A stale
// snapshot returns an error with no mutation. Every operation is scope-fenced.
type ConsolidationStore interface {
	StageRollout(context.Context, string, Rollout) error
	PendingRollouts(context.Context, string, int) ([]Rollout, error)
	CommitConsolidation(context.Context, string, Consolidation, []Change) error
}

func (c *curation) FinalizeRun(ctx context.Context, result agentcore.RunResult, failure error) (_ error) {
	defer func() {
		if p := recover(); p != nil {
			c.report(ctx, fmt.Errorf("memory consolidation panic: %v", p))
		}
	}()
	if c.consolidator == nil || c.consolidation == nil || failure != nil || result.Parked || result.Turns == 0 {
		return nil
	}
	if result.StopReason != "" && result.StopReason != "stop" && result.StopReason != "end_turn" {
		return nil
	}
	raw, err := json.Marshal(result.Messages)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(raw)
	r := Rollout{ID: hex.EncodeToString(sum[:]), Final: agentcore.TruncateMiddle(result.Final, 8000)}
	budget := 32000
	for i := len(result.Messages) - 1; i >= 0 && len(r.Messages) < 32 && budget > 0; i-- {
		m := result.Messages[i]
		if m.Role == agentcore.RoleSystem {
			continue
		}
		content := m.Content
		for _, call := range m.ToolCalls[:min(8, len(m.ToolCalls))] {
			content += "\nTool call " + agentcore.TruncateBytes(call.Name, 64) + ": " + agentcore.TruncateMiddle(call.Arguments, 1024)
		}
		m = agentcore.Message{Role: m.Role, Name: agentcore.TruncateBytes(m.Name, 64), ToolCallID: agentcore.TruncateBytes(m.ToolCallID, 128), Content: agentcore.TruncateMiddle(content, min(budget, 4000))}
		if m.Content == "" {
			continue
		}
		budget -= len(m.Content)
		r.Messages = append([]agentcore.Message{m}, r.Messages...)
	}
	if err := c.consolidation.StageRollout(ctx, c.scopeID, r); err != nil {
		c.report(ctx, err)
		return nil
	}
	if c.worker != nil {
		if !c.worker.submit(c) {
			c.report(ctx, fmt.Errorf("memory consolidation worker unavailable; evidence remains pending"))
		}
		return nil
	}
	c.consolidate(ctx)
	return nil
}

func (c *curation) consolidate(ctx context.Context) {
	defer func() {
		if p := recover(); p != nil {
			c.report(ctx, fmt.Errorf("memory consolidation panic: %v", p))
		}
	}()
	pending, err := c.consolidation.PendingRollouts(ctx, c.scopeID, 4)
	if err != nil {
		c.report(ctx, err)
		return
	}
	if len(pending) == 0 {
		return
	}
	memories, err := c.store.Recall(ctx, c.scopeID, "", 32)
	if err != nil {
		c.report(ctx, err)
		return
	}
	in := Consolidation{Rollouts: pending, Memories: memories}
	changes, err := c.consolidator(ctx, in)
	if err == nil {
		err = ValidateConsolidation(c.scopeID, in, changes)
	}
	if err == nil {
		err = c.consolidation.CommitConsolidation(ctx, c.scopeID, in, changes)
	}
	// A secondary model outage must leave durable evidence pending for retry,
	// never fail successful primary work or consume the pending rollouts.
	if err != nil {
		c.report(ctx, err)
	}
}
func (c *curation) report(ctx context.Context, err error) {
	defer func() { _ = recover() }()
	if c.onConsolidationError != nil {
		c.onConsolidationError(ctx, err)
	}
}

// ValidateConsolidation is shared with storage adapters at the commit boundary.
func ValidateConsolidation(scope string, in Consolidation, changes []Change) error {
	if len(in.Rollouts) == 0 || len(in.Rollouts) > 4 || len(in.Memories) > 32 || len(changes) > 16 {
		return fmt.Errorf("memory: consolidation exceeds bounds")
	}
	rollouts := map[string]bool{}
	for _, r := range in.Rollouts {
		if len(r.ID) != 64 || rollouts[r.ID] {
			return fmt.Errorf("memory: invalid rollout IDs")
		}
		rollouts[r.ID] = true
	}
	known := map[string]bool{}
	for _, m := range in.Memories {
		if m.ScopeID != scope || m.ID == "" {
			return fmt.Errorf("memory: invalid snapshot scope")
		}
		known[m.ID] = true
	}
	used := map[string]bool{}
	for _, ch := range changes {
		if len(ch.IDs) > 32 || (len(ch.IDs) == 0 && ch.Entry == nil) {
			return fmt.Errorf("memory: empty or oversized change")
		}
		for _, id := range ch.IDs {
			if !known[id] || used[id] {
				return fmt.Errorf("memory: unknown or duplicate source id")
			}
			used[id] = true
		}
		if e := ch.Entry; e != nil {
			if e.ScopeID != "" && e.ScopeID != scope {
				return fmt.Errorf("memory: scope widening")
			}
			if strings.TrimSpace(e.Content) == "" || len(e.Content) > 8192 || len(e.Tags) > 32 || len(strings.Join(e.Tags, " ")) > 2048 || e.Confidence < 0 || e.Confidence > 1 {
				return fmt.Errorf("memory: invalid lesson")
			}
		}
	}
	return nil
}
