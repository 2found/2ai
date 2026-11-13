package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	nativehost "github.com/2found/2ai/core/agentcore/host"
	"github.com/2found/2ai/core/ai"
)

// defaultContextTokenBudget is the soft ceiling at which the loop compacts old
// turns so long autonomous runs stay inside the model window (§5.2, §7).
//
// It is the *configured* ceiling only — the ceiling an operator asked for. The
// budget actually applied is effectiveBudget(), which caps this against the
// answering model's own window, so this number never has to be a guess about
// any particular model. Consumers override it via Limits.MaxContextTokens.
const defaultContextTokenBudget = 300_000

// outputHeadroomTokens is subtracted from a model's window before it caps the
// budget, because a context window holds the reply as well as the prompt. Sizing
// the budget to the whole window leaves the loop compacting only once the input
// alone has filled it, at which point there is no room left to answer in and the
// provider rejects the request — the exact failure the window cap exists to
// prevent. Sized against the loop's own MaxTokens ceiling with room to spare.
const outputHeadroomTokens = 32_000

// effectiveBudget is the compaction ceiling actually applied: the configured
// budget, capped by what the answering model can physically hold.
//
//	budget = min(window - outputHeadroom, configured)
//
// window is 0 when nobody could determine it — a self-hosted endpoint, a model
// id no catalog knows — and then the configured budget stands alone, which is
// the old behaviour. That asymmetry is deliberate: compacting earlier than
// necessary costs tokens, while compacting later than the window allows costs
// the run, and no retry or escalation can rescue a transcript that no longer
// fits.
//
// A window smaller than the headroom is not treated as "no room at all": such a
// model cannot serve this loop usefully anyway, and returning 0 would disable
// compaction entirely, which is the worst possible response to the smallest possible window. Half the window is
// the floor instead.
func effectiveBudget(configured, window int) int {
	if configured <= 0 {
		configured = defaultContextTokenBudget
	}
	if window <= 0 {
		return configured
	}
	usable := window - outputHeadroomTokens
	if usable < window/2 {
		usable = window / 2
	}
	if usable < configured {
		return usable
	}
	return configured
}

// defaultKeepRecentTokens is the approximate recent-context budget preserved
// verbatim after a compaction; everything older is summarized.
const defaultKeepRecentTokens = 20_000

// CompactionSettings tunes how a native run keeps a long transcript inside its
// budget. The run's transcript never shrinks: compaction rewrites only the
// request view (see package host), first clearing old tool results, then
// replacing the settled prefix with a summary.
type CompactionSettings struct {
	// KeepRecentTokens is the approximate size of the recent messages a
	// summary leaves verbatim. Zero selects the default; it is capped at half
	// the budget.
	KeepRecentTokens int
	// Instructions is appended to the summary prompt of every compaction in
	// the run: the host's or the user's standing focus, such as "keep every
	// SQL statement and its row count".
	Instructions string
	// StandaloneSummary turns off the shared summary call. By default the
	// summary is asked for as one more turn of the live conversation, so the
	// provider reads the transcript from its prompt cache and one call
	// replaces a chain of folds over a re-sent copy. The reply is used only
	// when it is a text-only summary block; anything else, and any run with a
	// forced tool choice, an output schema or a live request too large to take
	// the instruction, uses the standalone summarizer: a separate request that
	// re-sends a rendered copy of the transcript under its own system prompt.
	// Set this for a provider that bills cached input at the full rate, to
	// have a compaction model (Registry.SetCompactionModel) write every
	// summary instead of only the fallback, or when a summary request must not
	// reach the provider as a conversation turn.
	StandaloneSummary bool
	// SharePromptCache is ignored.
	//
	// Deprecated: sharing the prompt cache is the default; set
	// StandaloneSummary to turn it off.
	SharePromptCache bool
	// ClearToolResults clears old tool results from the request view before a
	// summary is needed. The zero value clears with the defaults; read_skill
	// results are never cleared.
	ClearToolResults ToolResultClearing
}

// ToolResultClearing configures tool-result clearing; see package host.
type ToolResultClearing = nativehost.ToolResultClearing

// CompactionEvent reports one context-summary attempt of a native run.
type CompactionEvent = nativehost.CompactionEvent

// DefaultCompactionSettings returns conservative defaults.
func DefaultCompactionSettings() CompactionSettings {
	return CompactionSettings{KeepRecentTokens: defaultKeepRecentTokens}
}

// keepRecent is KeepRecentTokens with its default, capped at half the budget
// so a small budget still leaves something to summarize.
func (s CompactionSettings) keepRecent(budget int) int {
	keep := s.KeepRecentTokens
	if keep <= 0 {
		keep = defaultKeepRecentTokens
	}
	if budget > 0 && keep > budget/2 {
		keep = budget / 2
	}
	return keep
}

// CompactionReattacher is implemented by an extension that keeps context the
// work needs verbatim across a summary: the files being edited, an active
// plan, a skill in use. After a new summary the run calls each one with every
// native message the summary stands for; what it returns, within maxTokens
// shared by all of them, follows the summary text in the request. The run
// itself reattaches the skills read in the summarized messages.
type CompactionReattacher interface {
	Extension
	ReattachAfterCompaction(ctx context.Context, dropped []json.RawMessage, maxTokens int) string
}

// reattach is the run's CompactionPolicy.Reattach: extension context first,
// then the skills read in the summarized messages, within maxTokens in all.
func (h *ToolHost) reattach(ctx context.Context, dropped []json.RawMessage, maxTokens int) string {
	room := maxTokens * 4
	var parts []string
	add := func(text string) {
		if text = strings.TrimSpace(text); text == "" || room <= 0 {
			return
		}
		text = truncateMiddle(text, room)
		room -= len(text) + 2
		parts = append(parts, text)
	}
	for _, ext := range h.exts.all {
		reattacher, ok := ext.(CompactionReattacher)
		if !ok || room <= 0 {
			continue
		}
		// Each extension gets its own copy: a mutation must not reach the
		// request the summary is part of.
		copied := make([]json.RawMessage, len(dropped))
		for i, raw := range dropped {
			copied[i] = append(json.RawMessage(nil), raw...)
		}
		var text string
		if err := safe(func() { text = reattacher.ReattachAfterCompaction(ctx, copied, room/4) }); err != nil {
			_ = safe(func() { _ = h.agent.hooks.emitErr("reattach["+ext.Name()+"]", err) })
			continue
		}
		add(text)
	}
	add(reattachSkills(dropped, room))
	return strings.Join(parts, "\n\n")
}

// reattachSkills returns the skills read in messages, most recent first and
// each once, within room bytes. A summary keeps what a skill said only as a
// paraphrase, and the model keeps following its steps.
func reattachSkills(messages []json.RawMessage, room int) string {
	ids := map[string]string{} // read_skill call ID -> skill id
	for _, raw := range messages {
		message, err := ai.ProjectMessage(raw)
		if err != nil {
			continue
		}
		for _, call := range message.ToolCalls {
			var args readSkillArgs
			if call.Name == readSkillToolName && json.Unmarshal([]byte(call.Arguments), &args) == nil {
				ids[call.ID] = strings.TrimSpace(args.ID)
			}
		}
	}
	var b strings.Builder
	seen := map[string]bool{}
	for i := len(messages) - 1; i >= 0 && len(ids) > 0; i-- {
		var result struct {
			Role, ToolCallID, ToolName string
			IsError                    bool
			Content                    json.RawMessage
		}
		if json.Unmarshal(messages[i], &result) != nil || result.Role != "toolResult" || result.ToolName != readSkillToolName || result.IsError {
			continue
		}
		id := ids[result.ToolCallID]
		text, _, _, err := ai.ProjectContent(result.Content)
		if id == "" || seen[id] || err != nil || strings.TrimSpace(text) == "" {
			continue
		}
		seen[id] = true
		block := fmt.Sprintf("[Skill %q, loaded earlier with read_skill and still in force]\n%s\n\n", id, strings.TrimSpace(text))
		if b.Len()+len(block) > room {
			continue
		}
		b.WriteString(block)
	}
	return b.String()
}
