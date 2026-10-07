package host

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"
	"unicode/utf8"
)

// ContextBudgetError contains only safe numeric diagnostics for public hosts.
type ContextBudgetError struct{ Estimated, Budget int }

func (e *ContextBudgetError) Error() string {
	return fmt.Sprintf("context still exceeds budget (%d > %d tokens); reduce the request or increase compact_at/context_window", e.Estimated, e.Budget)
}

// ContextTokens uses settled provider usage plus the appended tail, floored
// by a local estimate. Rewritten views invalidate older usage anchors without
// modifying the original billable usage or signed native messages.
func ContextTokens(raw json.RawMessage) int {
	floor := estimateTokens(string(raw))
	var messages []json.RawMessage
	if json.Unmarshal(raw, &messages) != nil {
		return floor
	}
	var rewrittenAt int64
	for _, raw := range messages {
		var m struct {
			Timestamp                int64
			AgentrayContextSummary   string
			AgentrayContextReducedAt int64
		}
		_ = json.Unmarshal(raw, &m)
		if m.AgentrayContextSummary != "" {
			rewrittenAt = max(rewrittenAt, m.Timestamp)
		}
		rewrittenAt = max(rewrittenAt, m.AgentrayContextReducedAt)
	}
	tail := 0
	for i := len(messages) - 1; i >= 0; i-- {
		var m struct {
			Role, StopReason string
			Timestamp        int64
			Usage            *struct{ Input, Output, CacheRead, CacheWrite float64 }
		}
		_ = json.Unmarshal(messages[i], &m)
		if m.Role == "assistant" && m.StopReason != "error" && m.StopReason != "aborted" && m.Usage != nil && (rewrittenAt == 0 || m.Timestamp > rewrittenAt) {
			u := m.Usage
			count := u.Input + u.Output + u.CacheRead + u.CacheWrite
			if count > 0 && !math.IsNaN(count) && !math.IsInf(count, 0) && count < float64(math.MaxInt-tail) {
				return max(floor, int(math.Ceil(count))+tail)
			}
		}
		tail += estimateTokens(string(messages[i]))
	}
	return floor
}

// Prepare verifies the actual outgoing view after compaction. Oversized tool
// text is recoverable through the existing archive/retrieval seam, never lost.
// If the request still cannot fit, fail before sending it instead of repeatedly
// paying for a summary over the same tail. The original transcript is untouched.
func (c *Compactor) Prepare(ctx context.Context, raw json.RawMessage, policy CompactionPolicy) (json.RawMessage, error) {
	view := c.TransformWithPolicy(ctx, raw, policy)
	fitBudget := policy.Budget
	if policy.MaxInputTokens > 0 {
		fitBudget = policy.MaxInputTokens
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if policy.Budget <= 0 || ContextTokens(view) <= policy.Budget {
		return view, nil
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(view, &messages); err != nil {
		return nil, err
	}
	if policy.Archive != nil {
		// Bound each result so several parallel calls also leave useful headroom.
		capBytes := max(256, policy.Budget*4/max(8, len(messages)))
		for i, m := range messages {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			var fields map[string]json.RawMessage
			if json.Unmarshal(m, &fields) != nil {
				continue
			}
			var role, name, id string
			_ = json.Unmarshal(fields["role"], &role)
			if role != "toolResult" {
				continue
			}
			_ = json.Unmarshal(fields["toolName"], &name)
			_ = json.Unmarshal(fields["toolCallId"], &id)
			var blocks []map[string]json.RawMessage
			if json.Unmarshal(fields["content"], &blocks) != nil {
				continue
			}
			changed := false
			for _, block := range blocks {
				var typ, text string
				_ = json.Unmarshal(block["type"], &typ)
				_ = json.Unmarshal(block["text"], &text)
				if typ != "text" || len(text) <= capBytes || name == "read_spill" {
					continue
				}
				locator, err := policy.Archive(ctx, name, id, text)
				if err != nil {
					return nil, fmt.Errorf("context archive failed: %w", err)
				}
				if locator == "" {
					return nil, fmt.Errorf("context archive returned no recoverable locator")
				}
				notice := fmt.Sprintf("\n[Context preview: full %d bytes saved. Use read_spill with locator %q and byte offset/limit to retrieve the rest.]", len(text), locator)
				preview := text[:min(len(text), max(0, capBytes-len(notice)))]
				// Never split UTF-8 text at a preview boundary.
				for len(preview) > 0 && !utf8.ValidString(preview) {
					preview = preview[:len(preview)-1]
				}
				block["text"], _ = json.Marshal(preview + notice)
				changed = true
			}
			if !changed {
				continue
			}
			fields["content"], _ = json.Marshal(blocks)
			fields["agentrayContextReducedAt"], _ = json.Marshal(time.Now().UnixMilli())
			messages[i], _ = json.Marshal(fields)
		}
		view, _ = json.Marshal(messages)
	}
	if ContextTokens(view) <= fitBudget {
		return view, nil
	}
	// Usage pressure may come from a complete newest turn, not a large text
	// result. A final fold is safe only when the original request has a completed
	// assistant/tool batch; Transform enforces that boundary and persistence.
	policy.Force = true
	next := c.TransformWithPolicy(ctx, raw, policy)
	if ContextTokens(next) < ContextTokens(view) {
		view = next
	}
	if ContextTokens(view) > fitBudget {
		return nil, &ContextBudgetError{Estimated: ContextTokens(view), Budget: fitBudget}
	}
	if strings.TrimSpace(string(view)) == "" {
		return nil, fmt.Errorf("empty compacted context")
	}
	return view, nil
}
