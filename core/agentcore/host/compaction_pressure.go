package host

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
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
	messages, err := splitMessages(raw)
	if err != nil {
		return max(1, estimateTokens(string(raw)))
	}
	facts := make([]*messageFacts, len(messages))
	for i, m := range messages {
		f := decodeFacts(m)
		facts[i] = &f
	}
	return factTokens(facts)
}

// Prepare verifies the actual outgoing view after compaction. Oversized tool
// text is recoverable through the existing archive/retrieval seam, never lost.
// If the request still cannot fit, fail before sending it instead of repeatedly
// paying for a summary over the same tail. The original transcript is untouched.
func (c *Compactor) Prepare(ctx context.Context, raw json.RawMessage, policy CompactionPolicy) (json.RawMessage, error) {
	messages, view, _ := c.transform(ctx, raw, policy, false)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	tokens := func(messages []json.RawMessage, raw json.RawMessage) int {
		if messages == nil {
			return ContextTokens(raw)
		}
		return factTokens(c.memo.of(messages))
	}
	fitBudget := policy.Budget
	if policy.MaxInputTokens > 0 {
		fitBudget = policy.MaxInputTokens
	}
	if policy.Budget <= 0 || tokens(messages, view) <= policy.Budget {
		return view, nil
	}
	if messages == nil {
		var err error
		if messages, err = splitMessages(view); err != nil {
			return nil, err
		}
	}
	if policy.Archive != nil {
		// Bound each result so several parallel calls also leave useful
		// headroom. The cap depends only on the budget, so a previewed result
		// is the same bytes on every request and the prompt cache still reads it.
		capBytes := max(256, policy.Budget/4)
		facts := c.memo.of(messages)
		var previewed []json.RawMessage
		for i, m := range messages {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			f := facts[i]
			if f.role != "toolResult" || f.toolName == "read_spill" || f.bytes <= capBytes {
				continue
			}
			next, err := c.preview(ctx, m, f, capBytes, policy.Archive)
			if err != nil {
				return nil, err
			}
			if previewed == nil {
				previewed = slices.Clone(messages)
			}
			previewed[i] = next
		}
		if previewed != nil {
			messages, view = previewed, joinMessages(previewed)
		}
	}
	if tokens(messages, view) <= fitBudget {
		return view, nil
	}
	// Usage pressure may come from a complete newest turn, not a large text
	// result. A final fold is safe only when the original request has a completed
	// assistant/tool batch; transform enforces that boundary and persistence.
	policy.Force = true
	if next, nextRaw, _ := c.transform(ctx, raw, policy, false); tokens(next, nextRaw) < tokens(messages, view) {
		messages, view = next, nextRaw
	}
	if estimated := tokens(messages, view); estimated > fitBudget {
		return nil, &ContextBudgetError{Estimated: estimated, Budget: fitBudget}
	}
	if strings.TrimSpace(string(view)) == "" {
		return nil, fmt.Errorf("empty compacted context")
	}
	return view, nil
}

type previewKey struct {
	mark     uint64
	capBytes int
}

// preview replaces each oversized text block of a tool result with its head
// and the archive locator of the full text. The rewrite is cached by message,
// so the archive is asked once and every request repeats the same bytes.
func (c *Compactor) preview(ctx context.Context, raw json.RawMessage, f *messageFacts, capBytes int, archive func(context.Context, string, string, string) (string, error)) (json.RawMessage, error) {
	key := previewKey{mark: c.memo.mark(raw), capBytes: capBytes}
	if cached, ok := c.previews[key]; ok {
		return cached, nil
	}
	var fields map[string]json.RawMessage
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil || json.Unmarshal(fields["content"], &blocks) != nil {
		return raw, nil
	}
	changed := false
	for _, block := range blocks {
		var typ, text string
		_ = json.Unmarshal(block["type"], &typ)
		_ = json.Unmarshal(block["text"], &text)
		if typ != "text" || len(text) <= capBytes {
			continue
		}
		locator, err := archive(ctx, f.toolName, f.toolCallID, text)
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
	out := raw
	if changed {
		fields["content"], _ = json.Marshal(blocks)
		fields["agentrayContextReducedAt"], _ = json.Marshal(c.clock().UnixMilli())
		out, _ = json.Marshal(fields)
	}
	if c.previews == nil {
		c.previews = map[previewKey]json.RawMessage{}
	}
	c.previews[key] = out
	return out, nil
}
