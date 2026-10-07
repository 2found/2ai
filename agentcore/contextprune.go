package agentcore

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// contextprune.go is the cheap first stage of the built-in compaction policy.
// Agent transcripts are mostly tool results, and many of those results stop
// being useful long before the surrounding reasoning does: a newer identical
// call superseded them, a later write made a file read stale, or the result is
// old, bulky, and cheap to reproduce. Once the transcript crosses half of its
// budget, the pruner replaces those old results in one batch. Batching is
// deliberate: every rewrite invalidates the provider's prompt-cache prefix
// from that point, so one thresholded pass is cheaper than editing every turn.

// Conventional file-tool names used by the stale-read rule. Compositions with
// other names simply skip that rule; superseded and old-result pruning remain
// tool-name agnostic.
const (
	pruneToolNameRead  = "read_file"
	pruneToolNameWrite = "write_file"
	pruneToolNameEdit  = "edit_file"
)

const (
	// Results below this size are not worth replacing on age alone.
	pruneOldResultMinBytes = 1024
	// Superseded or stale results can be cleared more aggressively because a
	// newer result already carries the truth (or proves the old result wrong).
	pruneStaleResultMinBytes = 128
)

// ContextPruner is an optional, deterministic first stage of a Compactor.
//
// The loop asks for pruning before ShouldCompact. This lets a strategy remove
// reproducible bulk at a soft threshold and avoid a summarization call when
// that alone restores headroom. It is deliberately optional: a custom
// Compactor that does not implement ContextPruner retains its exact behavior.
// The loop, not the pruner, owns hooks, durable checkpoints, and observer
// rebases around any returned rewrite.
type ContextPruner interface {
	// PruneContext returns a provider-valid replacement and whether it differs
	// from messages. Implementations must not mutate messages. Returning the
	// original slice with false is the no-op path.
	PruneContext(req ContextPruneRequest) ([]Message, bool)
}

// ContextPruneRequest carries the live pruning policy. PromptCacheActive is a
// run property rather than static compaction configuration: the same Agent can
// be composed with or without a provider cache key on laptops and servers.
type ContextPruneRequest struct {
	Messages          []Message
	Budget            int
	Settings          CompactionSettings
	PromptCacheActive bool
}

// pruneContextRequest batch-clears obsolete tool results after the transcript crosses
// half its context budget. Only the span before the keep-recent window is
// touched. Within that old span it clears, in confidence order:
//
//  1. results superseded by a newer call with the same name and arguments;
//  2. read_file results made stale by a later edit_file/write_file;
//  3. any old result large enough to be cheaper to reproduce than retain.
//
// Tool result identity and adjacency are preserved, so all provider tool-call
// invariants remain valid. The operation is idempotent because its placeholders
// are below every clearing threshold.
func pruneContextRequest(req ContextPruneRequest) ([]Message, bool) {
	messages, budget, settings := req.Messages, req.Budget, req.Settings
	if budget <= 0 {
		budget = defaultContextTokenBudget
	}
	settings = effectiveCompaction(settings, budget)
	if estimateContextTokens(messages) <= budget/2 {
		return messages, false
	}

	sysN := leadingSystemCount(messages)
	body := messages[sysN:]
	cut := findCutPoint(body, settings.KeepRecentTokens)
	if cut <= 0 {
		return messages, false
	}

	// Build the indexes over the whole body, not just the old span: a recent
	// read or write is exactly what makes an older result superseded or stale.
	callByID := make(map[string]ToolCall)
	newestResultByCall := make(map[string]int)
	lastWriteByPath := make(map[string]int)
	for i, m := range body {
		if m.Role == RoleAssistant {
			for _, tc := range m.ToolCalls {
				callByID[tc.ID] = tc
				if tc.Name == pruneToolNameWrite || tc.Name == pruneToolNameEdit {
					if path := toolArgPath(tc.Arguments); path != "" {
						lastWriteByPath[path] = i
					}
				}
			}
		}
		if m.Role == RoleTool && m.ToolCallID != "" {
			if tc, ok := callByID[m.ToolCallID]; ok {
				newestResultByCall[toolCallKey(tc)] = i
			}
		}
	}

	// Cache guard: suffix[i] is the estimated message-token total strictly after
	// body[i]. Rewriting a candidate invalidates that suffix in the provider's
	// warm prefix. Deep candidates wait for full compaction; cheap tail rewrites
	// still reclaim redundant output without a summary call.
	var suffix []int
	if req.PromptCacheActive && settings.PruneCacheWarmSuffixTokens > 0 {
		suffix = make([]int, len(body))
		acc := 0
		for i := len(body) - 1; i >= 0; i-- {
			suffix[i] = acc
			acc += estimateBytesTokens(body[i : i+1])
		}
	}

	type candidate struct {
		index       int
		placeholder string
		savings     int
	}
	var highConfidence, generic []candidate
	addCandidate := func(dst *[]candidate, i int, placeholder string) {
		m := body[i]
		if m.ResultRef != "" && !strings.Contains(placeholder, m.ResultRef) {
			placeholder = strings.TrimSuffix(placeholder, "]") + fmt.Sprintf("; full output remains available at %s through the artifact retrieval tool]", m.ResultRef)
		}
		replacement := m
		replacement.Content = placeholder
		replacement.ContentParts = nil
		savedTokens := estimateBytesTokens([]Message{m}) - estimateBytesTokens([]Message{replacement})
		if savedTokens <= 0 {
			return
		}
		*dst = append(*dst, candidate{index: i, placeholder: placeholder, savings: savedTokens})
	}

	for i := 0; i < cut; i++ {
		m := body[i]
		if m.Role != RoleTool || m.ToolCallID == "" {
			continue
		}
		if suffix != nil && suffix[i] > settings.PruneCacheWarmSuffixTokens {
			continue
		}
		tc, haveCall := callByID[m.ToolCallID]
		// Old durable logs predate ResultRef and carry a spill locator only in
		// this notice. Preserve those messages wholesale on every rule; there is
		// no structured handle from which a safe replacement can be rebuilt.
		if m.ResultRef == "" && strings.Contains(m.Content, "Full result saved at:") {
			continue
		}

		if haveCall && (len(m.Content) > pruneStaleResultMinBytes || len(m.ContentParts) > 0) && newestResultByCall[toolCallKey(tc)] > i {
			addCandidate(&highConfidence, i, fmt.Sprintf("[superseded by a newer identical %s call — re-run it if you need the content]", tc.Name))
			continue
		}
		if haveCall && tc.Name == pruneToolNameRead && (len(m.Content) > pruneStaleResultMinBytes || len(m.ContentParts) > 0) {
			if path := toolArgPath(tc.Arguments); path != "" && lastWriteByPath[path] > i {
				addCandidate(&highConfidence, i, fmt.Sprintf("[stale read_file result — %s was modified later; re-read it if needed]", path))
				continue
			}
		}
		// Generic old-result pruning is deliberately conservative. Skill bodies
		// are durable instructions, error output is often the only evidence needed
		// to diagnose a failure, and consumers can name non-repeatable/artifact
		// tools in settings. Superseded and stale results were handled above because
		// their content is no longer authoritative regardless of protection.
		if protectedPruneResult(m, tc, haveCall, settings.PruneProtectedTools) {
			continue
		}
		if len(m.Content) > pruneOldResultMinBytes || len(m.ContentParts) > 0 {
			name := m.Name
			if name == "" && haveCall {
				name = tc.Name
			}
			if name == "" {
				name = "the tool"
			}
			placeholder := fmt.Sprintf("[older %s result cleared after it was consumed; retain conclusions from subsequent reasoning]", name)
			if m.ResultRef != "" {
				placeholder = fmt.Sprintf("[older %s result cleared after it was consumed; full output remains available at %s through the artifact retrieval tool]", name, m.ResultRef)
			}
			addCandidate(&generic, i, placeholder)
		}
	}

	genericSavings := 0
	for _, c := range generic {
		genericSavings += c.savings
	}
	selected := highConfidence
	if genericSavings >= settings.PruneMinimumSavingsTokens {
		selected = append(selected, generic...)
	}
	if len(selected) == 0 {
		return messages, false
	}
	out := slices.Clone(messages)
	for _, c := range selected {
		m := body[c.index]
		// Preserve provider linkage and recoverable artifact metadata. Usage and
		// other assistant-only metadata are intentionally absent on tool messages.
		out[sysN+c.index] = Message{
			Role:       RoleTool,
			ToolCallID: m.ToolCallID,
			Name:       m.Name,
			Content:    c.placeholder,
			ResultRef:  m.ResultRef,
		}
	}
	return adjustUsageAfterPrune(messages, out), true
}

func protectedPruneResult(m Message, tc ToolCall, haveCall bool, protected []string) bool {
	content := strings.TrimSpace(m.Content)
	if strings.HasPrefix(content, "error:") || strings.HasPrefix(content, "blocked:") {
		return true
	}
	name := m.Name
	if name == "" && haveCall {
		name = tc.Name
	}
	return slices.Contains(protected, name)
}

func toolCallKey(tc ToolCall) string { return tc.Name + "\x00" + tc.Arguments }

// toolArgPath extracts the conventional path field from tool arguments.
func toolArgPath(args string) string {
	var in struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return ""
	}
	return strings.TrimSpace(in.Path)
}

// adjustUsageAfterPrune retains provider-reported usage—including system and
// tool-schema overhead that message-byte fallback cannot see—while subtracting
// only the estimated tokens removed before each observation. The Usage itself
// remains untouched because it is a billable historical fact; the adjustment
// is a separate context-pressure hint and survives durable resume.
func adjustUsageAfterPrune(before, after []Message) []Message {
	out := slices.Clone(after)
	saved := 0
	for i := range out {
		if i < len(before) && !sameForCache(before[i], after[i]) {
			saved += estimateBytesTokens(before[i:i+1]) - estimateBytesTokens(after[i:i+1])
		}
		if out[i].Role == RoleAssistant && out[i].Usage != nil && saved != 0 {
			out[i].ContextTokenAdjustment -= saved
		}
	}
	return out
}
