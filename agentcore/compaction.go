package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
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
// compaction entirely (shouldCompact reads 0 as "use the default"), which is the
// worst possible response to the smallest possible window. Half the window is
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
// verbatim after a compaction; everything older is summarized (pi's
// keepRecentTokens).
const defaultKeepRecentTokens = 20_000

// defaultMaxSummaryTokens is the ceiling on the checkpoint a compaction leaves
// behind, and minSummaryTokens is the floor that ceiling is never clamped below.
//
// The checkpoint needs a ceiling for the same reason the recent tail does: after
// a compaction the window holds the leading system head, the summary, and the
// tail, and a budget that bounds only the tail bounds nothing. Left unbounded
// the summary ratchets, because that is what the update prompt asks for —
// "preserve everything already captured, fold in the new messages" is a
// monotonic instruction, so every fold appends and none subtracts. The result is
// not a crash but a slow strangling: the checkpoint grows until it fills the
// budget by itself, the post-compaction transcript never drops back under the
// ceiling, and compaction re-fires on the very next turn. Measured by
// TestVeryLongRunKeepsItsCheckpointInsideTheBudget with these bounds removed —
// 1500 turns, a 4000-token budget, a summarizer that folds honestly — the
// checkpoint reached 60 KB against a 4 KB share, 790 of 1500 turns ran over
// budget, and compaction fired once per 1.6 turns: a summarization call for
// every turn and a half of actual work.
//
// A quarter of the budget is the share that leaves the arithmetic sound: the
// tail already takes half (effectiveCompaction), so head + summary + tail lands
// near three quarters and a compaction buys real turns of headroom before the
// next one. On a normal window the clamp never binds — a quarter of 190k is far
// above 2048 — so this only changes behaviour where it was broken: the small
// windows a cheap model gives you, which is exactly where the long cheap runs
// happen.
const (
	defaultMaxSummaryTokens = 2048
	minSummaryTokens        = 256
	// defaultPruneCacheWarmSuffixTokens is the largest already-sent suffix the
	// built-in pruner will invalidate while prompt caching is active. Rewriting
	// deeper history forces the provider to re-cache too much of a warm prefix;
	// summary compaction can reclaim it when the window is actually full.
	defaultPruneCacheWarmSuffixTokens = 8_000
	// defaultPruneMinimumSavingsTokens prevents a generic age-only rewrite from
	// churning the prompt cache for negligible savings. effectiveCompaction
	// scales it down for small local-model windows.
	defaultPruneMinimumSavingsTokens = 20_000
)

// summaryMarker prefixes a compaction summary message so later compactions can
// recognize a prior summary and fold it into the next one instead of
// re-summarizing it as raw history.
const summaryMarker = "[context summary of earlier conversation]"

// elideMarker prefixes the breadcrumb the deterministic elide path leaves when
// summarization is unavailable or fails. Surfacing it (alongside summaryMarker)
// lets observers see that a *degraded* compaction happened rather than none.
const elideMarker = "[context compaction]"

// goalMarker prefixes the pinned-requirement system message. The first time the
// loop compacts the task out of the live window, that task is lifted into a
// goal-marked system message kept verbatim by every later compaction (it sorts
// into the leading-system head, which is never summarized). This stops the run's
// objective from drifting as successive lossy summaries fold into one another —
// the literal requirement is always in front of the model. (pi keeps the first
// user task pinned for the same reason.)
//
// Unlike pi's, this pin is REBUILT on every compaction rather than written once,
// because a long run's requirement changes: see foldDirectivesIntoPin.
const goalMarker = "[pinned goal — the run's requirement, kept verbatim across compaction; keep working toward it. " +
	"Where entries below conflict, a LATER one supersedes an earlier one.]"

// CompactionSettings tunes how the loop compacts a long transcript. It sizes
// both halves of what a compaction leaves behind: KeepRecentTokens is the
// approximate token budget of recent messages kept verbatim, MaxSummaryTokens
// the ceiling on the checkpoint that stands for everything older. Both are
// clamped against the run's real budget by effectiveCompaction.
type CompactionSettings struct {
	KeepRecentTokens int
	MaxSummaryTokens int
	// PruneProtectedTools exempts tool results from generic age-based pruning.
	// Superseded identical results and stale file reads may still be pruned: a
	// newer result already carries the truth. Nil selects the safe built-ins;
	// an explicitly empty slice disables built-in protection. Add tools whose
	// outputs are non-repeatable artifacts or durable instructions.
	PruneProtectedTools []string
	// PruneCacheWarmSuffixTokens protects deep results while prompt caching is
	// active. A candidate is rewritten only when the estimated message suffix
	// after it is no larger than this value. Zero selects the default; a negative
	// value disables the guard.
	PruneCacheWarmSuffixTokens int
	// PruneMinimumSavingsTokens is the minimum aggregate saving required for
	// generic age-only pruning. Superseded and stale results bypass this floor.
	// <=0 selects the default; the effective value is capped to 10% of Budget.
	PruneMinimumSavingsTokens int
}

// DefaultCompactionSettings returns conservative defaults.
func DefaultCompactionSettings() CompactionSettings {
	return CompactionSettings{
		KeepRecentTokens:           defaultKeepRecentTokens,
		MaxSummaryTokens:           defaultMaxSummaryTokens,
		PruneProtectedTools:        []string{readSkillToolName},
		PruneCacheWarmSuffixTokens: defaultPruneCacheWarmSuffixTokens,
		PruneMinimumSavingsTokens:  defaultPruneMinimumSavingsTokens,
	}
}

// estimateContextTokens estimates how full the context window is, used only to
// decide when to compact — never for billing. It prefers the provider's real
// token count: it finds the most recent assistant message carrying Usage (the
// input+output of that turn ≈ the context size at that point) and adds a cheap
// byte estimate for the messages appended after it (the untracked trailing tool
// results / steering). When no message carries usage it falls back to a pure
// ~4-bytes/token heuristic over the whole transcript (pi's
// estimateContextTokens shape).
func estimateContextTokens(messages []Message) int {
	lastUsageIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == RoleAssistant && messages[i].Usage != nil {
			lastUsageIdx = i
			break
		}
	}
	if lastUsageIdx < 0 {
		return estimateBytesTokens(messages)
	}
	u := messages[lastUsageIdx].Usage
	measured := u.InputTokens + u.OutputTokens + messages[lastUsageIdx].ContextTokenAdjustment
	if measured < 0 {
		measured = 0
	}
	return measured + estimateBytesTokens(messages[lastUsageIdx+1:])
}

// estimateBytesTokens is the cheap ~4-bytes/token fallback over a message slice.
func estimateBytesTokens(messages []Message) int {
	bytes := 0
	for _, m := range messages {
		bytes += len(m.Content)
		// Base64 size is not prompt token size: vision adapters decode images and
		// bill by dimensions/tiles. Use a conservative fixed estimate until the
		// next provider Usage observation replaces this fallback.
		for _, part := range m.ContentParts {
			if part.Type == ContentPartImage {
				bytes += 4 * 1024
			} else {
				bytes += len(part.Text)
			}
		}
		for _, tc := range m.ToolCalls {
			bytes += len(tc.Name) + len(tc.Arguments)
		}
		for _, block := range m.ReasoningBlocks {
			bytes += len(block.Text) + len(block.Signature) + len(block.Data)
		}
	}
	return bytes / bytesPerTokenEstimate
}

// effectiveCompaction sizes what a compaction is allowed to leave behind
// against the budget it has to fit inside — both halves of it.
//
// The keep-recent window is clamped to half the budget. Without that clamp, a
// consumer who sets Limits.MaxContextTokens below KeepRecentTokens (default 20k)
// wedges the loop: shouldCompact fires every turn, but findCutPoint sees the
// entire transcript inside the "recent" window (cut 0) and falls back to the
// deterministic elide — which only collapses bulky tool results, so a transcript
// whose bulk lives in assistant tool-call arguments or prose passes through
// untouched, and compaction runs forever without ever shrinking anything.
//
// The checkpoint is clamped to a quarter, for the symmetric reason: a summary
// sized by a constant rather than by the budget is a second way for the
// post-compaction transcript to exceed the ceiling, and the model is being
// asked to grow it on every fold. The floor keeps a tiny budget from asking for
// a checkpoint too small to carry a goal and a next step, on the grounds that a
// useless summary is worse than a slightly oversized one.
func effectiveCompaction(settings CompactionSettings, budget int) CompactionSettings {
	if budget <= 0 {
		budget = defaultContextTokenBudget
	}
	if settings.KeepRecentTokens <= 0 {
		settings.KeepRecentTokens = defaultKeepRecentTokens
	}
	if settings.KeepRecentTokens > budget/2 {
		settings.KeepRecentTokens = budget / 2
	}
	if settings.MaxSummaryTokens <= 0 {
		settings.MaxSummaryTokens = defaultMaxSummaryTokens
	}
	if settings.MaxSummaryTokens > budget/4 {
		settings.MaxSummaryTokens = budget / 4
	}
	if settings.MaxSummaryTokens < minSummaryTokens {
		settings.MaxSummaryTokens = minSummaryTokens
	}
	if settings.PruneProtectedTools == nil {
		settings.PruneProtectedTools = slices.Clone(DefaultCompactionSettings().PruneProtectedTools)
	}
	if settings.PruneCacheWarmSuffixTokens == 0 {
		settings.PruneCacheWarmSuffixTokens = defaultPruneCacheWarmSuffixTokens
	}
	if settings.PruneMinimumSavingsTokens <= 0 {
		settings.PruneMinimumSavingsTokens = defaultPruneMinimumSavingsTokens
	}
	if cap := budget / 10; cap > 0 && settings.PruneMinimumSavingsTokens > cap {
		settings.PruneMinimumSavingsTokens = cap
	}
	return settings
}

// shouldCompact reports whether the estimated context exceeds the budget.
func shouldCompact(messages []Message, budget int) bool {
	if budget <= 0 {
		budget = defaultContextTokenBudget
	}
	return estimateContextTokens(messages) > budget
}

// leadingSystemCount counts the original leading system messages — those that
// are NOT a prior compaction summary. A prior summary (system role, summaryMarker
// prefix) is deliberately excluded so the next compaction folds it into the new
// summary rather than freezing it as permanent header.
func leadingSystemCount(messages []Message) int {
	n := 0
	for n < len(messages) && messages[n].Role == RoleSystem && !strings.HasPrefix(messages[n].Content, summaryMarker) {
		n++
	}
	return n
}

// findCutPoint returns the index in body at which the retained recent tail
// begins: walk back from the end accumulating an estimate until keepRecentTokens
// is reached, then snap earlier so the tail never starts on a tool-result
// message (which must stay attached to the assistant turn that issued it). A
// return of 0 means nothing is old enough to compact.
func findCutPoint(body []Message, keepRecentTokens int) int {
	if keepRecentTokens <= 0 {
		keepRecentTokens = defaultKeepRecentTokens
	}
	acc := 0
	cut := 0
	reached := false
	for i := len(body) - 1; i >= 0; i-- {
		acc += estimateBytesTokens(body[i : i+1])
		if acc >= keepRecentTokens {
			cut = i
			reached = true
			break
		}
	}
	if !reached {
		return 0 // whole transcript fits in the recent budget
	}
	for cut > 0 && body[cut].Role == RoleTool {
		cut-- // keep tool results with their owning assistant turn
	}
	return cut
}

// compactWithSummary replaces the older span of a long transcript with a single
// model-generated structured checkpoint, keeping the leading system prompt and
// the recent tail verbatim. The summary call uses the supplied provider/model
// (the active rung — typically the cheapest). On any failure (provider error,
// empty summary, nothing old enough) it falls back to the deterministic elide so
// a run is never broken by compaction. The returned Usage is what the
// summarization call itself spent (zero on the no-LLM paths) so the caller can
// fold it into run accounting.
func compactWithSummary(ctx context.Context, provider LLMProvider, model string, messages []Message, settings CompactionSettings) ([]Message, Usage) {
	sysN := leadingSystemCount(messages)
	head := messages[:sysN]
	body := messages[sysN:] // may begin with a prior summary (folded in below)

	cut := findCutPoint(body, settings.KeepRecentTokens)
	if cut <= 0 {
		return compact(messages, 6), Usage{} // nothing old enough for a clean cut; elide
	}
	older := body[:cut]
	// Tail-overflow guard (pi's split-turn case): findCutPoint snaps the cut back
	// so tool results stay attached to the assistant turn that issued them, so a
	// single turn with giant tool results can leave a "recent" tail far larger
	// than the keep budget — and since the next compaction would then find
	// nothing new to fold, the run would re-trigger compaction forever without
	// ever shrinking. Instead of cutting mid-turn (which would orphan tool
	// results from their call and break provider validation), bound the tail by
	// eliding its bulkiest older tool results in place.
	tail, tailShrunk := elideOversizedTail(body[cut:], settings.KeepRecentTokens)

	// Iterative update-summary: if the older span begins with a prior summary
	// (folded in by a previous compaction), lift it out and fold only the
	// genuinely-new older messages into it, rather than re-summarizing the prior
	// summary as raw transcript. This keeps long runs lossless on facts the prior
	// summary already captured and saves the model from re-derived drift. (pi's
	// UPDATE_SUMMARIZATION_PROMPT / generateSummary(previousSummary).)
	prevSummary, newOlder := splitPriorSummary(older)
	if prevSummary != "" && len(newOlder) == 0 {
		// Only the prior summary was old; nothing new to fold. Still honor the
		// tail guard, otherwise an oversized tail would wedge compaction (fire
		// every turn, change nothing).
		if !tailShrunk {
			return messages, Usage{}
		}
		out := make([]Message, 0, len(head)+len(older)+len(tail))
		out = append(out, head...)
		out = append(out, older...) // just the prior summary message
		out = append(out, tail...)
		return out, Usage{}
	}

	maxSummary := settings.MaxSummaryTokens
	if maxSummary <= 0 {
		maxSummary = defaultMaxSummaryTokens
	}
	summary, su, err := summarizeSpan(ctx, provider, model, newOlder, prevSummary, maxSummary)
	if err != nil || strings.TrimSpace(summary) == "" {
		// Degrade, don't break — but still report su: an empty summary from a
		// successful call was billed even though its output was unusable.
		return compact(messages, 6), su
	}
	// The request asked for a bounded checkpoint; this enforces it. MaxTokens is
	// a request, not a guarantee — a self-hosted or OpenAI-compatible endpoint
	// may ignore it — and an over-long checkpoint is not merely this turn's
	// problem: it is fed back as the next fold's "previous summary", so one
	// unbounded reply becomes the permanent floor of every window that follows.
	summary = clampSummary(summary, maxSummary)

	summaryMsg := Message{Role: RoleSystem, Content: summaryMarker + "\n" + strings.TrimSpace(summary)}
	out := make([]Message, 0, len(head)+2+len(tail))
	// The pin carries the run's REQUIREMENT past the summaries that would
	// otherwise erode it — and it is rebuilt here, not written once, because the
	// requirement is not fixed. Anything the user directed that is about to leave
	// the window is folded in before it goes.
	head, pinned := foldDirectivesIntoPin(head, older, maxSummary/2)
	out = append(out, head...)
	if !pinned {
		// No pin in head and nothing marked: either the first compaction of a run
		// whose seed predates the Directive stamp, or a log written by an older
		// version. Fall back to what compaction always did — pin the first user
		// message — so an old session still gets its objective held.
		if goal, ok := firstUserText(older); ok {
			out = append(out, Message{Role: RoleSystem, Content: goalMarker + "\n" + strings.TrimSpace(goal)})
		}
	}
	out = append(out, summaryMsg)
	out = append(out, tail...)
	return out, su
}

// elidedResultBytes is what an oversized tool result is cut down to, rather than
// removed outright.
//
// The tail guard used to replace such a result with a bare placeholder telling
// the model to re-run the tool. That is the wrong half of the content to keep
// and the wrong advice to give. A tool result puts its CONCLUSION at the end —
// the finding, the verdict, the error — so dropping everything loses precisely
// the part the call was made for. And "re-run it" assumes the call is cheap and
// repeatable: a spawn_subagent result is a whole child run that has already been
// paid for, and re-running it costs more than every byte this guard saves.
//
// Keeping both ends of a kilobyte is nearly free against a context budget and
// preserves the answer. truncateMiddle already does exactly this everywhere else
// oversized text meets a ceiling in this package.
const elidedResultBytes = 1024

// elideOversizedTail bounds the kept-verbatim tail of a compaction: while its
// estimate exceeds the keep budget, bulky tool results are cut down to
// elidedResultBytes (head and tail kept, middle removed), oldest first, never
// touching the final message (the newest state the model is acting on). Call
// linkage (ToolCallID/Name) is preserved so the transcript stays provider-valid.
// Returns the (possibly copied) tail and whether anything was elided; a tail
// that cannot shrink further is returned best-effort.
func elideOversizedTail(tail []Message, keepRecentTokens int) ([]Message, bool) {
	if keepRecentTokens <= 0 {
		keepRecentTokens = defaultKeepRecentTokens
	}
	if estimateBytesTokens(tail) <= keepRecentTokens {
		return tail, false
	}
	out := make([]Message, len(tail))
	copy(out, tail)
	shrunk := false
	for i := 0; i < len(out)-1 && estimateBytesTokens(out) > keepRecentTokens; i++ {
		m := out[i]
		if m.Role != RoleTool || (len(m.Content) <= elidedResultBytes && len(m.ContentParts) == 0) {
			continue
		}
		content := truncateResultWithRef(m.Content, m.ResultRef, elidedResultBytes)
		if images := imagePartCount(m); images > 0 {
			content = strings.TrimSpace(content) + fmt.Sprintf("\n[%d rich image output(s) elided from the compacted tail]", images)
		}
		out[i] = Message{
			Role:       RoleTool,
			ToolCallID: m.ToolCallID,
			Name:       m.Name,
			Content:    content,
			ResultRef:  m.ResultRef,
		}
		shrunk = true
	}
	if !shrunk {
		return tail, false
	}
	return out, true
}

// firstUserText returns the content of the first non-empty user message in span.
func firstUserText(span []Message) (string, bool) {
	for _, m := range span {
		if m.Role == RoleUser && strings.TrimSpace(m.Content) != "" {
			return m.Content, true
		}
	}
	return "", false
}

// goalPinIndex locates the pinned-requirement message, or -1.
func goalPinIndex(msgs []Message) int {
	for i, m := range msgs {
		if m.Role == RoleSystem && strings.HasPrefix(m.Content, goalMarker) {
			return i
		}
	}
	return -1
}

// pinUpdateSep separates the original task from each later correction inside the
// pin. It is a fixed string this package both writes and splits on, which is what
// lets the pin be re-read and rebuilt on every compaction rather than parsed out
// of prose.
//
// It is deliberately terse, and the precedence rule it used to restate lives in
// goalMarker instead. The pin is re-sent on every request for the rest of the
// run, so a separator is not paid once — it is paid per update, per turn, for
// thousands of turns. Saying "later supersedes earlier" once in the header says
// it just as clearly for a third of the bytes.
const pinUpdateSep = "\n\n--- requirement update ---\n"

// pinDroppedNote stands in for corrections the ceiling forced out of the pin.
// It names the loss rather than hiding it: the text is still in the durable log
// and, usually, in the checkpoint, so a model told that something was dropped
// can ask for it — one told nothing simply proceeds on a partial requirement.
const pinDroppedNote = "\n\n--- (earlier requirement updates omitted for space; the context summary covers them) ---\n"

// directiveTexts returns, in order, the human-authored requirements in span:
// the run's task and any correction steered in since. Framework-synthesized
// user messages — goal-gate nudges, budget wrap-ups, extension injections —
// are not marked and so are never mistaken for what the run is for.
func directiveTexts(span []Message) []string {
	var out []string
	for _, m := range span {
		if m.Role == RoleUser && m.Directive {
			if t := strings.TrimSpace(m.Content); t != "" {
				out = append(out, t)
			}
		}
	}
	return out
}

// foldDirectivesIntoPin brings the pinned requirement up to date before the
// messages carrying it are summarized away, and reports whether head now holds
// a pin at all.
//
// This is the half of "the objective must not drift" that pinning the first user
// message gets wrong. A run of any length is steered: the user says stop doing
// that, do this instead. The correction is an ordinary transcript message, so
// compaction summarizes it, and successive folds dilute it — while the pin,
// which every compaction preserves verbatim, still states the requirement that
// was cancelled and still says to keep working toward it. Measured before this
// existed, on a 1200-turn run steered at turn 150: the superseded task was in
// front of the model word for word at the end, and the correction had vanished
// from the window entirely.
//
// So the pin accumulates. The original task stays — a correction is rarely
// interpretable without the thing it corrects — and each later directive is
// appended under a separator that tells the model which one wins.
func foldDirectivesIntoPin(head []Message, older []Message, maxTokens int) ([]Message, bool) {
	fresh := directiveTexts(older)
	idx := goalPinIndex(head)
	if len(fresh) == 0 {
		return head, idx >= 0
	}

	existing := ""
	if idx >= 0 {
		existing = strings.TrimPrefix(head[idx].Content, goalMarker)
	}
	content := goalMarker + "\n" + composePin(existing, fresh, maxTokens)

	// Copy before writing: head aliases the caller's transcript, and compaction
	// must not mutate the messages it was handed.
	out := make([]Message, len(head), len(head)+1)
	copy(out, head)
	if idx >= 0 {
		out[idx] = Message{Role: RoleSystem, Content: content}
		return out, true
	}
	return append(out, Message{Role: RoleSystem, Content: content}), true
}

// composePin renders the original task plus its corrections into one bounded
// message.
//
// Bounded, because the pin is preserved verbatim forever and an unbounded
// verbatim message in a long run's window is the same ratchet the checkpoint
// ceiling exists to stop. When the budget binds, the OLDEST corrections go
// first: the original stays because it anchors everything after it, and the
// newest stay because they are the operative instruction. What is dropped is the
// middle — corrections that later ones have most likely already superseded, and
// that the checkpoint summarized on its way past.
func composePin(existing string, fresh []string, maxTokens int) string {
	parts := splitPin(existing)
	for _, f := range fresh {
		// Restating a requirement the pin already holds is not a NEW requirement,
		// and recording it as one is actively wrong: the pin's ordering is its
		// meaning ("a LATER one supersedes an earlier one"), so re-appending the
		// original task would push it back in front of the correction that
		// cancelled it. This is not hypothetical — a resumed run is handed the
		// transcript's last user message as its task, which is very often
		// something the pin already carries.
		if slices.Contains(parts, f) {
			continue
		}
		parts = append(parts, f)
	}
	if len(parts) == 0 {
		return ""
	}
	if maxTokens <= 0 {
		maxTokens = defaultMaxSummaryTokens / 2
	}
	budget := maxTokens * bytesPerTokenEstimate

	// Drop from the second entry inward while the rendered pin is over budget,
	// keeping the first (the task) and the tail (the live corrections).
	dropped := 0
	for {
		rendered := renderPin(parts, dropped)
		if len(rendered) <= budget || len(parts)-dropped <= 2 {
			// Nothing left to drop without losing the task or the newest
			// correction; truncate what remains rather than return an oversized pin.
			return truncateMiddle(rendered, budget)
		}
		dropped++
	}
}

// renderPin joins the task and its corrections, omitting `dropped` entries after
// the first and noting that it did.
func renderPin(parts []string, dropped int) string {
	var b strings.Builder
	b.WriteString(parts[0])
	rest := parts[1:]
	if dropped > 0 && dropped <= len(rest) {
		rest = rest[dropped:]
		b.WriteString(pinDroppedNote)
	}
	for _, p := range rest {
		b.WriteString(pinUpdateSep)
		b.WriteString(p)
	}
	return b.String()
}

// splitPin recovers the parts renderPin wrote, so a pin can be rebuilt from the
// one already in the window instead of from the transcript it replaced.
func splitPin(pin string) []string {
	pin = strings.TrimSpace(pin)
	if pin == "" {
		return nil
	}
	// The dropped-note is a rendering artifact, not a requirement; fold it into
	// the separator so it does not come back as a part of its own.
	pin = strings.ReplaceAll(pin, strings.TrimSuffix(pinDroppedNote, "\n"), "")
	var out []string
	for _, p := range strings.Split(pin, pinUpdateSep) {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// splitPriorSummary detects a prior compaction summary at the head of an older
// span and separates it from the genuinely-new messages that follow. The prior
// summary always sorts to the front of the body (it is a leading non-original
// system message), so it can only be older[0]. Returns the summary text (without
// its marker) and the remaining new messages; ("", older) when none is present.
func splitPriorSummary(older []Message) (string, []Message) {
	if len(older) > 0 && older[0].Role == RoleSystem && strings.HasPrefix(older[0].Content, summaryMarker) {
		prev := strings.TrimSpace(strings.TrimPrefix(older[0].Content, summaryMarker))
		return prev, older[1:]
	}
	return "", older
}

// summarizeSpan asks the model to distill a span of conversation into the
// structured checkpoint format. The span is serialized to a single transcript
// (roles preserved) and handed to a one-shot, non-streaming Chat call with no
// tools. When previousSummary is non-empty the model is asked to UPDATE it in
// place — preserving what it already captured and folding in only the new
// messages — instead of summarizing from scratch. The returned Usage is the
// summarization call's own spend, so it can be billed against the run.
//
// maxTokens is the checkpoint's ceiling, and it is applied twice over: as the
// call's MaxTokens, and as a stated rule in the prompt. Both, because they fail
// differently. MaxTokens truncates mid-sentence at whatever token the limit
// falls on, which on the update path means the next fold inherits a checkpoint
// that stops mid-word; the prompt rule instead asks the model to spend the
// budget deliberately, consolidating what it has rather than being cut off.
func summarizeSpan(ctx context.Context, provider LLMProvider, model string, span []Message, previousSummary string, maxTokens int) (string, Usage, error) {
	if len(span) == 0 {
		return "", Usage{}, fmt.Errorf("empty span")
	}
	if maxTokens <= 0 {
		maxTokens = defaultMaxSummaryTokens
	}
	var userContent string
	if strings.TrimSpace(previousSummary) != "" {
		userContent = "## Previous summary\n" + strings.TrimSpace(previousSummary) +
			"\n\n## New messages since that summary\n" + serializeConversation(span) +
			"\n\n" + updateSummarizationPrompt + sizeRule(maxTokens)
	} else {
		userContent = serializeConversation(span) + "\n\n" + summarizationPrompt + sizeRule(maxTokens)
	}
	req := ChatRequest{
		Model: model,
		Messages: []Message{
			{Role: RoleSystem, Content: summarizationSystemPrompt},
			{Role: RoleUser, Content: userContent},
		},
		MaxTokens: maxTokens,
	}
	resp, err := provider.Chat(ctx, req)
	if err != nil {
		return "", Usage{}, err
	}
	return resp.Message.Content, resp.Usage, nil
}

// sizeRule states the checkpoint's ceiling to the model, in words, and says what
// to do on reaching it. Naming the limit is not enough on its own: told only
// "stay under N words" while also told to preserve everything, a model has been
// handed two rules that eventually contradict, and it will keep the one stated
// last. So the rule also fixes the tie-break — consolidate the accumulated
// record of finished work, which is where a long run's checkpoint puts nearly
// all its growth, and never at the expense of the parts a resuming agent needs
// to act: the goal, live blockers, next steps, identifiers.
//
// Words rather than tokens because a model can approximately count words and
// cannot count its own tokens; ~3/4 of a word per token is the usual ratio for
// English prose.
func sizeRule(maxTokens int) string {
	return fmt.Sprintf("\n- SIZE LIMIT: keep the whole checkpoint under about %d words. "+
		"If including everything would exceed that, CONSOLIDATE instead of growing: "+
		"collapse older completed work into one summarizing line per theme, and drop detail "+
		"that later work has superseded. Never drop the goal, unresolved blockers, the next "+
		"steps, or identifiers still needed to continue — compress the record of finished work first.",
		maxTokens*3/4)
}

// clampSummary enforces the checkpoint ceiling on a reply that overshot it.
//
// The cut takes the middle out rather than the end, because the checkpoint's
// format puts its most load-bearing sections last: a head-only truncation keeps
// Goal and the oldest Done items and throws away Next Steps and Critical
// Context — the two sections a resuming agent actually reads. Keeping both ends
// costs the middle of the Done list, which is the most compressible thing in the
// document and the part the run's own transcript still records.
// The cut lands on LINE boundaries, and that part is not cosmetic. A checkpoint
// is a list of facts, and a byte-exact cut through one produces a fact that is
// wrong while still reading as complete — "code FINDING-4=SHARD00" for a code
// that ends 002OK. Nothing downstream can tell the difference: this checkpoint
// is handed to the next fold as the previous summary and the model is asked to
// carry it forward, so the mutilated identifier propagates for the rest of the
// run and is reported as a finding. Dropping a fact is honest and recoverable —
// the run can look it up again. Half a fact presented as whole is a fabrication
// the run will defend. So whole lines go, and the marker says how many, which is
// something the next fold can read and act on.
func clampSummary(summary string, maxTokens int) string {
	summary = strings.TrimSpace(summary)
	if maxTokens <= 0 {
		return summary
	}
	maxBytes := maxTokens * bytesPerTokenEstimate
	if len(summary) <= maxBytes {
		return summary
	}
	lines := strings.Split(summary, "\n")
	// Reserve room for the marker, whose digits vary.
	const reserve = 80
	budget := maxBytes - reserve
	if len(lines) < 3 || budget < 2 {
		// No line structure to respect, or no room to say what went: a byte cut
		// is the only thing left, and some cut beats an unbounded checkpoint
		// becoming the permanent floor of every window after it.
		return truncateMiddle(summary, maxBytes)
	}

	// Whole lines from the front, then whole lines from the end with whatever
	// budget is left. Two thirds to the head so the Goal and the oldest Done
	// items survive alongside Next Steps and Critical Context at the tail.
	head, headBytes := 0, 0
	for head < len(lines) && headBytes+len(lines[head])+1 <= budget*2/3 {
		headBytes += len(lines[head]) + 1
		head++
	}
	tail, tailBytes := len(lines), 0
	for tail > head && tailBytes+len(lines[tail-1])+1 <= budget-headBytes {
		tailBytes += len(lines[tail-1]) + 1
		tail--
	}
	if tail <= head {
		// One line is bigger than the whole budget; nothing can be kept whole.
		return truncateMiddle(summary, maxBytes)
	}

	kept := make([]string, 0, head+1+len(lines)-tail)
	kept = append(kept, lines[:head]...)
	kept = append(kept, fmt.Sprintf("…[%d earlier lines dropped to fit the checkpoint budget]…", tail-head))
	kept = append(kept, lines[tail:]...)
	return strings.Join(kept, "\n")
}

// bytesPerTokenEstimate is the ~4-bytes-per-token rule the loop already sizes
// context with (estimateBytesTokens). Reused here so a ceiling expressed in
// tokens converts to bytes the same way everywhere.
const bytesPerTokenEstimate = 4

// Bounds applied when serializing a span for the summarizer, so one giant tool
// result (a full query dump, pages of build output) cannot blow the compaction
// call's own context window (pi truncates serialized tool results the same
// way). Head+tail truncation keeps the end of the result, where the signal
// usually is.
//
// Arguments get a per-VALUE ceiling *before* the per-call one, the ordering omp
// uses in snapcompact's serializer. With only the per-call cap, a single fat
// value (a file body, a SQL blob) eats the whole budget and truncates its
// siblings — including their *names* — out of the text the summarizer reads, so
// the checkpoint's "Key Decisions" section is written blind about exactly the
// calls that matter most: writes, edits, SQL.
const (
	maxSerializedToolResult   = 2000
	maxSerializedToolArgs     = 600
	maxSerializedToolArgValue = 200
)

// serializeConversation renders a span of messages into a plain-text transcript
// the summarizer can read (roles labeled; tool calls and results inlined, both
// bounded so the summarization request itself stays small).
func serializeConversation(span []Message) string {
	var b strings.Builder
	for _, m := range span {
		switch m.Role {
		case RoleSystem:
			fmt.Fprintf(&b, "SYSTEM: %s\n", strings.TrimSpace(m.Content))
		case RoleUser:
			fmt.Fprintf(&b, "USER: %s\n", strings.TrimSpace(m.Content))
		case RoleAssistant:
			if c := strings.TrimSpace(m.Content); c != "" {
				fmt.Fprintf(&b, "ASSISTANT: %s\n", c)
			}
			for _, tc := range m.ToolCalls {
				fmt.Fprintf(&b, "ASSISTANT called tool %s(%s)\n", tc.Name, serializeToolArgs(tc.Arguments))
			}
		case RoleTool:
			content := truncateMiddle(strings.TrimSpace(m.Content), maxSerializedToolResult)
			if images := imagePartCount(m); images > 0 {
				content = strings.TrimSpace(content) + fmt.Sprintf(" [%d rich image output(s)]", images)
			}
			fmt.Fprintf(&b, "TOOL %s -> %s\n", m.Name, content)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// serializeToolArgs renders a tool call's JSON arguments for the summarizer,
// applying the per-value ceiling before the per-call one (omp's ordering in
// snapcompact's serializer). Capping the whole blob alone let one fat argument
// starve its siblings: a write call's 40 KB "content" consumed the entire
// budget and the name of every argument after it vanished, so the summarizer
// described a call it could not read. Per value first, every key name survives.
//
// Keys are emitted sorted, which makes the rendering deterministic — the
// summarization request participates in the provider's prefix cache, and Go's
// randomized map iteration would churn it on every compaction.
//
// Arguments that are not a JSON object (a bare string, an array, malformed
// JSON) fall through to the previous whole-blob truncation unchanged.
func serializeToolArgs(raw string) string {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &fields); err != nil || fields == nil {
		return truncateMiddle(raw, maxSerializedToolArgs)
	}
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+truncateMiddle(string(fields[k]), maxSerializedToolArgValue))
	}
	return truncateMiddle(strings.Join(parts, ", "), maxSerializedToolArgs)
}

const summarizationSystemPrompt = `You are a context summarization assistant. Read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`

const summarizationPrompt = `The transcript above is a conversation to summarize. Create a structured context checkpoint that another assistant will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Constraints, preferences, or requirements mentioned] (or "(none)")

## Progress
### Done
- [x] [Completed tasks/changes]
### In Progress
- [ ] [Current work]
### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Data, examples, identifiers, or references needed to continue] (or "(none)")

Keep each section concise. Preserve exact identifiers, names, and error messages.`

const updateSummarizationPrompt = `Above is a "Previous summary" (a structured context checkpoint from earlier in this same conversation) followed by the "New messages since that summary".

Produce an UPDATED checkpoint in the EXACT same format as the previous summary:

## Goal
## Constraints & Preferences
## Progress
### Done
### In Progress
### Blocked
## Key Decisions
## Next Steps
## Critical Context

Rules for the update:
- PRESERVE what the previous summary established and is still true — do not drop facts, decisions, identifiers, or critical context just because they are not mentioned again in the new messages. Preserving a fact does not require preserving its original wording: several finished items may be carried as one line.
- FOLD IN what the new messages add: record new completed work in Done, move finished items out of In Progress into Done, add new decisions and next steps, and update or clear blockers that were resolved.
- Do NOT invent anything and do NOT re-derive or contradict the previous summary; only correct it when the new messages explicitly supersede it.
- Output only the updated checkpoint, no preamble.`

// compact collapses old tool-result messages into a short placeholder while
// preserving the system message, the original user task, and the most recent
// keepRecent messages verbatim. This is the deterministic, no-LLM fallback used
// when model summarization is unavailable or fails.
func compact(messages []Message, keepRecent int) []Message {
	if keepRecent < 2 {
		keepRecent = 2
	}
	if len(messages) <= keepRecent+2 {
		return messages
	}

	out := make([]Message, 0, len(messages))
	cutoff := len(messages) - keepRecent
	collapsed := 0
	for i, m := range messages {
		// Always keep the leading system prompt and the recent tail verbatim.
		if m.Role == RoleSystem || i >= cutoff {
			out = append(out, m)
			continue
		}
		// Collapse bulky tool results in the older region; keep their linkage.
		if m.Role == RoleTool && (len(m.Content) > 256 || len(m.ContentParts) > 0) {
			collapsed++
			content := "[older tool result elided to fit context]"
			if m.ResultRef != "" {
				content = fmt.Sprintf("[older tool result elided to fit context; full output remains available at %s through the artifact retrieval tool]", m.ResultRef)
			}
			out = append(out, Message{
				Role:       RoleTool,
				ToolCallID: m.ToolCallID,
				Name:       m.Name,
				Content:    content,
				ResultRef:  m.ResultRef,
			})
			continue
		}
		out = append(out, m)
	}
	if collapsed > 0 {
		// Leave a breadcrumb so the model knows history was trimmed, inserted
		// right after any leading system prompt.
		note := Message{Role: RoleSystem, Content: fmt.Sprintf(
			"%s %d older tool results were elided to stay within the model window. Re-run a tool if you need the detail.", elideMarker, collapsed)}
		at := 0
		if len(out) > 0 && out[0].Role == RoleSystem {
			at = 1
		}
		out = append(out[:at:at], append([]Message{note}, out[at:]...)...)
	}
	return out
}

func imagePartCount(message Message) int {
	count := 0
	for _, part := range message.ContentParts {
		if part.Type == ContentPartImage {
			count++
		}
	}
	return count
}

// truncateResultWithRef bounds a tool result while ensuring an opaque recovery
// handle remains visible to the model. The field itself is durable metadata and
// is not serialized by provider adapters, so preserving only ResultRef would
// make the artifact recoverable to observers but unreachable to the agent.
func truncateResultWithRef(content, ref string, maxBytes int) string {
	truncated := truncateMiddle(content, maxBytes)
	if ref == "" || strings.Contains(truncated, ref) {
		return truncated
	}
	notice := fmt.Sprintf("\n[full output remains available at %s through the artifact retrieval tool]", ref)
	if len(notice) >= maxBytes {
		return truncateBytes(notice, maxBytes)
	}
	return truncateMiddle(content, maxBytes-len(notice)) + notice
}

// Compactor is transcript shrinking, as a seam.
//
// The loop owns WHEN to consider compacting (once per turn, past the context
// budget) and owns the durable bracket around it. A Compactor owns WHAT
// compaction does: which span is old enough to lose, what replaces it, and what
// that costs. Those are strategy decisions with more than one right answer —
// summarize the older span with a model call, prune stale tool results with no
// call at all, keep a domain-specific span pinned — so they belong behind an
// interface rather than inside loop.go.
//
// DefaultCompactor is the built-in provider,
// and a composition that wants different behaviour registers its own through
// Registry.SetCompactor without touching the package.
type Compactor interface {
	// Name identifies the compactor in diagnostics.
	Name() string
	// ShouldCompact reports whether the transcript has grown past the point
	// where this strategy wants to act. The loop asks once per turn; a
	// compactor that measures pressure differently (message count, a real
	// tokenizer, a provider-reported context length) answers here.
	ShouldCompact(messages []Message, budget int) bool
	// Compact returns the shrunk transcript. It must preserve provider
	// validity — every tool result stays adjacent to the assistant turn that
	// called it — because the loop hands the result straight to the next
	// request. Returning the input unchanged is a legal no-op.
	//
	// An error degrades to "no compaction this turn" rather than failing the
	// run: a transcript that could not be shrunk is still a transcript the
	// model can work with, while a killed run loses everything.
	Compact(ctx context.Context, req CompactionRequest) (CompactionResult, error)
}

// CompactionRequest is everything a compactor needs to decide and act. It is a
// struct rather than a parameter list so a new input is an additive change that
// no existing implementation has to acknowledge.
type CompactionRequest struct {
	// Messages is the live transcript, leading system prompt included.
	Messages []Message
	// Budget is the run's context ceiling in estimated tokens
	// (Limits.MaxContextTokens), already defaulted.
	Budget int
	// Turn is the turn about to be taken, for correlating with the durable log.
	Turn int
	// Settings is the retention policy, already clamped against Budget.
	Settings CompactionSettings
	// Provider and Model are the tier the summarization call should use: the
	// dedicated compaction rung when the composition pinned one, otherwise the
	// rung the run has escalated to. A compactor that makes no model call
	// ignores both.
	Provider LLMProvider
	Model    string
}

// CompactionResult is what a compactor produced.
type CompactionResult struct {
	// Messages is the replacement transcript.
	Messages []Message
	// Usage is what the compaction itself spent — real billable spend on a
	// summarizing backend, zero on a deterministic one. The loop folds it into
	// the run's accounting and stamps it on the durable completion entry, so
	// the audit trail shows what staying inside the window cost.
	Usage Usage
}

// summaryCompactor is agentcore's built-in strategy: summarize the older span
// into a structured checkpoint with a model call, degrading to a deterministic
// elide when the call is unavailable or fails. It exists as a named type so the
// default is a provider like any other, not a hardcoded call.
type summaryCompactor struct{}

func (summaryCompactor) Name() string { return "summary" }

func (summaryCompactor) ShouldCompact(messages []Message, budget int) bool {
	return shouldCompact(messages, budget)
}

func (summaryCompactor) PruneContext(req ContextPruneRequest) ([]Message, bool) {
	return pruneContextRequest(req)
}

func (summaryCompactor) Compact(ctx context.Context, req CompactionRequest) (CompactionResult, error) {
	msgs, u := compactWithSummary(ctx, req.Provider, req.Model, req.Messages, req.Settings)
	return CompactionResult{Messages: msgs, Usage: u}, nil
}

// DefaultCompactor returns the built-in summarizing compactor. Wrap or replace
// it in a custom compaction plugin to change the strategy without touching the
// package.
func DefaultCompactor() Compactor { return summaryCompactor{} }
