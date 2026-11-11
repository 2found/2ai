package host

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/ai/protocol"
	"github.com/google/uuid"
)

// CompactionPolicy configures the consumer's request-only transform. The
// original Agent retains its full transcript, including every tool result.
type CompactionPolicy struct {
	// Approximate token counts. A nonpositive Budget disables new summaries;
	// ForWindow can derive or cap it from a model window. KeepRecent is bounded
	// to at least one token and at most half the effective budget.
	Budget, KeepRecent int
	// MaxInputTokens is the model's reserve-aware input ceiling. Budget remains
	// a compaction trigger/target; immutable host tools can exceed a very low
	// target while still fitting the model. Without a known ceiling, Budget is
	// also the fit bound.
	MaxInputTokens int
	// Force permits summarizing a completed latest turn after a confirmed
	// pre-output context rejection. It never replays tools or changes history.
	Force bool
	// Archive saves a full tool result before a request-only preview replaces
	// it. The returned locator must be readable by this run's retrieval tool.
	Archive func(context.Context, string, string, string) (string, error)
	// Task is the consumer's current user request, retained verbatim alongside
	// summaries. It is data, never a system instruction or extra permission.
	Task string
	// Summarize writes the summary of a native prefix, given this summary's
	// Instructions (a manual compaction adds its own focus to them).
	Summarize func(ctx context.Context, prefix json.RawMessage, instructions string) (string, protocol.Usage, error)
	// SummarizeView, when set, is tried before Summarize. It receives the whole
	// request view, so a host can ask for the summary as a continuation of the
	// live conversation and read that conversation from the provider's prompt
	// cache instead of paying for it again. An error or an unusable reply falls
	// back to Summarize.
	SummarizeView func(context.Context, SummaryRequest) (string, protocol.Usage, error)
	// Instructions is the host's or the user's focus for summaries, such as
	// "keep every SQL statement". Summarize and SummarizeView receive it.
	Instructions string
	// Allow is consulted before a summary is paid for. Returning false leaves
	// the view unsummarized for this request and is not counted as a failure.
	// The prefix is the span the summary would replace.
	Allow func(ctx context.Context, prefix json.RawMessage) bool
	// Observe receives one event per summary attempt, successful or not.
	Observe func(CompactionEvent)
	// Clearing clears old tool results from the view before a summary is
	// paid for. The zero value clears with the defaults.
	Clearing ToolResultClearing
	// Reattach, when set, is called after a new summary with every
	// authoritative message it stands for. What it returns, at most maxTokens,
	// is kept after the summary text: context the work still needs verbatim,
	// such as the files being edited, an active plan or a skill in use.
	Reattach func(ctx context.Context, dropped []json.RawMessage, maxTokens int) string
}

// MaxReattachTokens bounds what Reattach may add to one summary; a policy
// also caps it at an eighth of its Budget.
const MaxReattachTokens = 8192

// reattachMarker separates a summary from its reattached context.
const reattachMarker = "\n\n[Context reattached after the summary]\n"

// SummaryRequest is the request view a SummarizeView callback summarizes.
type SummaryRequest struct {
	// View is the JSON array of native messages the next provider request
	// would carry. Cut is the index at which the retained tail begins: the
	// summary replaces View[:Cut], system messages excepted.
	View         json.RawMessage
	Cut          int
	Revision     string
	Instructions string
}

// CompactionEvent reports one summary attempt. Token counts are the same
// estimates the compactor decides with.
type CompactionEvent struct {
	// Trigger is "auto" for the soft budget, "forced" for a hard input limit
	// or a provider overflow rejection, and "manual" for Compactor.Compact.
	Trigger string `json:"trigger"`
	// Mode is "shared" when SummarizeView produced the summary and
	// "standalone" when Summarize did.
	Mode       string `json:"mode"`
	PreTokens  int    `json:"pre_tokens"`
	PostTokens int    `json:"post_tokens,omitempty"`
	// PrefixMessages is how many authoritative messages the summary stands for.
	PrefixMessages int            `json:"prefix_messages,omitempty"`
	Failed         bool           `json:"failed,omitempty"`
	Usage          protocol.Usage `json:"usage"`
}

// MaxSummaryFailures is how many consecutive summary attempts may fail before
// the compactor stops trying at the soft budget. Without it a summarizer that
// keeps failing is called, and billed, on every request for the rest of the
// run. A forced compaction still tries: there the alternative is a request the
// provider rejects.
const MaxSummaryFailures = 3

// summaryLabel opens every summary message.
const summaryLabel = "[Earlier work summary]"

// SummaryText returns the handoff of a summary message's text, without its
// label and reattached context, and whether text is a summary at all.
func SummaryText(text string) (string, bool) {
	body, ok := strings.CutPrefix(strings.TrimSpace(text), summaryLabel)
	if !ok {
		return "", false
	}
	body, _, _ = strings.Cut(body, reattachMarker)
	return strings.TrimSpace(body), true
}

// Summary binds a persisted summary to an exact native transcript prefix.
type Summary struct {
	Revision     string          `json:"revision"`
	PrefixCount  int             `json:"prefix_count"`
	PrefixDigest string          `json:"prefix_digest"`
	Message      json.RawMessage `json:"message"`
}

// ParseSummary validates a persisted summary envelope.
func ParseSummary(raw string) (Summary, error) {
	var summary Summary
	var message struct {
		Role, Content, AgentrayContextSummary string
		Timestamp                             int64
	}
	if json.Unmarshal([]byte(raw), &summary) != nil || summary.Revision == "" || summary.PrefixCount <= 0 || len(summary.PrefixDigest) != 64 || json.Unmarshal(summary.Message, &message) != nil || message.Role != "user" || !validSummaryText(message.Content) || message.AgentrayContextSummary == "" || message.Timestamp <= 0 {
		return summary, errors.New("invalid native context summary")
	}
	if _, err := hex.DecodeString(summary.PrefixDigest); err != nil {
		return summary, errors.New("invalid native context summary digest")
	}
	return summary, nil
}

// PrefixDigest hashes semantic JSON identity without rounding opaque numbers.
// transcriptMemo.digest computes the same value incrementally.
func PrefixDigest(messages []json.RawMessage) string {
	raw, _ := json.Marshal(messages)
	canonical, err := CanonicalJSON(raw)
	if err != nil {
		return ""
	}
	digest := sha256.Sum256(canonical)
	return hex.EncodeToString(digest[:])
}

// Compactor owns one run's request view and persisted summary. Calls must be serialized.
type Compactor struct {
	config   CompactionPolicy
	revision string
	saved    *Summary
	cleared  ClearState
	failures int
	record   func(context.Context, json.RawMessage) error
	usage    func(protocol.Usage)
	memo     transcriptMemo
	rewrites map[rewriteKey]json.RawMessage
	previews map[previewKey]json.RawMessage
	now      func() time.Time
}

func (c *Compactor) clock() time.Time {
	if c.now != nil {
		return c.now()
	}
	return time.Now()
}

// Compact summarizes every settled message of raw now, whatever its size:
// the host's "/compact" with policy.Instructions as its focus. It does not
// consult Allow or the failure breaker, since someone asked for it.
func (c *Compactor) Compact(ctx context.Context, raw json.RawMessage, policy CompactionPolicy) error {
	policy.Force = true
	if _, _, summarized := c.transform(ctx, raw, policy, true); !summarized {
		if err := ctx.Err(); err != nil {
			return err
		}
		return errors.New("the conversation could not be compacted")
	}
	return nil
}

// transform returns the request view of raw under policy. A saved summary
// refers to a prefix of this exact view (after native host hooks), never to a
// Go text projection; on mismatch the full native context is the safe
// fallback. Calls are serialized by the host.
func (c *Compactor) transform(ctx context.Context, raw json.RawMessage, policy CompactionPolicy, manual bool) (messages []json.RawMessage, out json.RawMessage, summarized bool) {
	out = raw
	defer func() {
		if recover() != nil {
			messages, out, summarized = nil, raw, false
		}
	}()
	if ctx.Err() != nil {
		return
	}
	original, err := splitMessages(raw)
	if err != nil || validFacts(c.memo.of(original)) != nil {
		return
	}
	messages = original
	full, base, head := original, 0, 0
	if saved := c.saved; saved != nil && saved.Revision == c.revision && saved.PrefixCount <= len(original) && c.memo.digest(original, saved.PrefixCount) == saved.PrefixDigest {
		role := ""
		if saved.PrefixCount < len(original) {
			role = c.memo.of(original[saved.PrefixCount : saved.PrefixCount+1])[0].role
		}
		// A system boundary is the next run's opening message: a summary that
		// covered a whole earlier run still applies.
		if saved.PrefixCount == len(original) || role == "assistant" || role == "user" || role == "system" {
			full = summaryViewWithTask(original, c.memo.of(original), saved.PrefixCount, saved.Message, policy.Task)
			base = saved.PrefixCount
			head = len(full) - (len(original) - base)
		}
	}
	view := c.clear(ctx, full, head, base, policy, false)
	messages, out = view, joinMessages(view)
	facts := c.memo.of(view)
	cut := compactionCut(facts, policy.Budget, policy.KeepRecent)
	if policy.Force && len(view) > head {
		last := len(view) - 1
		// Host reminders may follow the current user/assistant turn. Preserve
		// those systems while choosing the same completed-turn boundary.
		for facts[last].role == "system" && last > head {
			last--
		}
		switch facts[last].role {
		case "assistant", "toolResult":
			cut = last + 1
		case "user":
			// When an oversized historical fact dominates the retained tail,
			// fold it as well while retaining the latest user message verbatim.
			// A lone initial request is never summarized away.
			for _, earlier := range facts[head:last] {
				if earlier.role == "user" || earlier.role == "assistant" {
					cut = last
					break
				}
			}
		}
	}
	if cut <= head || (policy.Summarize == nil && policy.SummarizeView == nil) {
		return
	}
	if !manual && c.failures >= MaxSummaryFailures && !policy.Force {
		return
	}
	// The summarizer reads the uncleared prefix: a cleared result is cheap to
	// leave out of a request, not something a handoff should forget.
	prefix := joinMessages(full[:cut])
	if !manual && policy.Allow != nil && !policy.Allow(ctx, prefix) {
		return
	}
	event := CompactionEvent{Trigger: "auto", PreTokens: factTokens(facts)}
	switch {
	case manual:
		event.Trigger = "manual"
	case policy.Force:
		event.Trigger = "forced"
	}
	// A cancelled run did not learn anything about the summarizer.
	failed := func() {
		if ctx.Err() != nil {
			return
		}
		c.failures++
		event.Failed = true
		if policy.Observe != nil {
			policy.Observe(event)
		}
	}
	account := func(usage protocol.Usage) {
		event.Usage = event.Usage.Add(usage)
		if c.usage != nil {
			c.usage(usage)
		}
	}
	text := ""
	if policy.SummarizeView != nil {
		reply, usage, err := policy.SummarizeView(ctx, SummaryRequest{View: out, Cut: cut, Revision: c.revision, Instructions: policy.Instructions})
		account(usage)
		if err == nil && ctx.Err() == nil && validSummaryText(reply) {
			text, event.Mode = reply, "shared"
		}
	}
	if text == "" && policy.Summarize != nil && ctx.Err() == nil {
		reply, usage, err := policy.Summarize(ctx, prefix, policy.Instructions)
		account(usage)
		if err == nil && ctx.Err() == nil && validSummaryText(reply) {
			text, event.Mode = reply, "standalone"
		}
	}
	if text == "" {
		failed()
		return
	}
	count := base + cut - head
	content := summaryLabel + "\n" + strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(text), summaryLabel))
	if policy.Reattach != nil {
		limit := min(MaxReattachTokens, max(0, policy.Budget/8))
		if extra := strings.TrimSpace(policy.Reattach(ctx, original[:count], limit)); extra != "" && limit > 0 && validSummaryText(extra) {
			content += reattachMarker + truncateMiddle(extra, limit*4)
		}
	}
	message, _ := json.Marshal(map[string]any{"role": "user", "content": content, "agentrayContextSummary": uuid.NewString(), "timestamp": c.clock().UnixMilli()})
	next := summaryViewWithTask(original, c.memo.of(original), count, message, policy.Task)
	// The new summary rewrites the prefix, so the cache cannot be read past it
	// anyway: the cheapest moment to clear the tail's old results too.
	next = c.clear(ctx, next, len(next)-(len(original)-count), count, policy, true)
	nextRaw := joinMessages(next)
	// A summary which grows the request is not compaction. Keep the last useful
	// view while still accounting for the auxiliary model work already performed.
	if len(nextRaw) >= len(out) {
		failed()
		return
	}
	saved := Summary{Revision: c.revision, PrefixCount: count, PrefixDigest: c.memo.digest(original, count), Message: message}
	payload, _ := json.Marshal(saved)
	if c.record != nil && c.record(ctx, payload) != nil {
		return
	}
	c.saved = &saved
	c.failures = 0
	if policy.Observe != nil {
		event.PostTokens, event.PrefixMessages = ContextTokens(nextRaw), count
		policy.Observe(event)
	}
	return next, nextRaw, true
}

// summaryView replaces a validated prefix while preserving its system messages.
// The caller must supply a cut within the message slice and outside a tool batch.
func summaryView(messages []json.RawMessage, facts []*messageFacts, cut int, summary json.RawMessage) []json.RawMessage {
	kept := []json.RawMessage{}
	for i, raw := range messages[:cut] {
		if facts[i].role == "system" {
			kept = append(kept, raw)
		}
	}
	kept = append(kept, summary)
	return append(kept, messages[cut:]...)
}

// summaryViewWithTask pins the current request only while the summary has
// absorbed it. A request still verbatim in the retained tail needs no copy, and
// the pin would sit between the summary and the tail: every new run brings a
// new task, so the provider's prompt cache would miss the whole tail each run.
func summaryViewWithTask(messages []json.RawMessage, facts []*messageFacts, cut int, summary json.RawMessage, task string) []json.RawMessage {
	view := summaryView(messages, facts, cut, summary)
	if task == "" || taskVisible(messages[cut:], facts[cut:], task) {
		return view
	}
	// Insert before the retained tail so later user messages keep their order.
	index := len(view) - (len(messages) - cut)
	pinned, _ := json.Marshal(map[string]any{"role": "user", "content": "[Current request — verbatim]\n" + task})
	view = append(view[:index:index], append([]json.RawMessage{pinned}, view[index:]...)...)
	return view
}

// taskVisible reports whether a retained user message still carries task,
// verbatim or as a JSON string value (consumers may wrap input in an envelope).
func taskVisible(tail []json.RawMessage, facts []*messageFacts, task string) bool {
	forms := []string{task}
	for _, escapeHTML := range []bool{true, false} {
		var b bytes.Buffer
		encoder := json.NewEncoder(&b)
		encoder.SetEscapeHTML(escapeHTML)
		if encoder.Encode(task) == nil {
			quoted := strings.TrimSpace(b.String())
			forms = append(forms, quoted[1:len(quoted)-1])
		}
	}
	for i, raw := range tail {
		var message struct{ Content json.RawMessage }
		if facts[i].role != "user" || json.Unmarshal(raw, &message) != nil {
			continue
		}
		text, _, _, err := ai.ProjectContent(message.Content)
		if err != nil {
			continue
		}
		for _, form := range forms {
			if strings.Contains(text, form) {
				return true
			}
		}
	}
	return false
}

// SummaryAllowance reserves the immutable host systems/tools and exact task
// before assigning output to the summary. Small budgets cannot spend a fixed
// fraction on summary text when tool declarations already fill most of them.
func SummaryAllowance(prefix json.RawMessage, task string, budget int) (int, error) {
	var messages []json.RawMessage
	if err := json.Unmarshal(prefix, &messages); err != nil {
		return 0, err
	}
	message := json.RawMessage(`{"role":"user","content":"[Earlier work summary]","agentrayContextSummary":"summary","timestamp":1}`)
	base, _ := json.Marshal(summaryViewWithTask(messages, new(transcriptMemo).of(messages), len(messages), message, task))
	available := budget - ContextTokens(base) - 64
	if available < 128 {
		return 0, &ContextBudgetError{Estimated: ContextTokens(base) + 128, Budget: budget}
	}
	return min(MaxSummaryTokens, budget/4, available), nil
}

func validSummaryText(text string) bool {
	if strings.TrimSpace(strings.TrimPrefix(text, summaryLabel)) == "" {
		return false
	}
	for _, marker := range []string{"<|open|>", "<|close|>", "<|sep|>", "<tool_call>", "<function_calls>"} {
		if strings.Contains(text, marker) {
			return false
		}
	}
	return true
}

// A long autonomous run may contain only one user prompt. Cut at a completed
// assistant/tool batch boundary as well as a user turn, never inside the batch.
// Several host-authored historical facts can also be compacted before the first
// assistant. Keep a lone initial task intact until it has produced a response.
func compactionCut(facts []*messageFacts, budget, keep int) int {
	if budget <= 0 || factTokens(facts) <= budget || validFacts(facts) != nil {
		return 0
	}
	keep = min(max(1, keep), max(1, budget/2))
	priorAssistant := make([]bool, len(facts))
	priorUsers := make([]int, len(facts))
	users, seen := 0, false
	for i, f := range facts {
		priorAssistant[i], priorUsers[i] = seen, users
		seen = seen || f.role == "assistant"
		if f.role == "user" {
			users++
		}
	}
	recent := 0
	for i := len(facts) - 1; i > 0; i-- {
		recent += facts[i].tokens
		role := facts[i].role
		if recent >= keep && ((priorAssistant[i] && (role == "assistant" || role == "user")) || (role == "user" && priorUsers[i] > 0)) {
			return i
		}
	}
	return 0
}

// CompactorOptions binds the host's persistence and usage accounting. Record
// must succeed before a new summary becomes visible to the provider.
type CompactorOptions struct {
	Revision string
	Policy   CompactionPolicy
	Record   func(context.Context, json.RawMessage) error
	Usage    func(protocol.Usage)
}

// NewCompactor creates a run-owned compactor. Calls are serialized by the host.
func NewCompactor(options CompactorOptions) *Compactor {
	return &Compactor{revision: options.Revision, config: options.Policy, record: options.Record, usage: options.Usage}
}

// Checkpoint returns an independent copy of the latest persisted summary.
// A nil result means no summary has been accepted.
func (c *Compactor) Checkpoint() json.RawMessage {
	if c.saved == nil {
		return nil
	}
	raw, _ := json.Marshal(c.saved)
	return raw
}

// Restore validates a saved summary before replacing the current checkpoint.
// The next Transform additionally verifies its digest against the native input.
func (c *Compactor) Restore(raw json.RawMessage) error {
	saved, err := ParseSummary(string(raw))
	if err != nil {
		return err
	}
	if saved.Revision != c.revision {
		return errors.New("native context summary revision mismatch")
	}
	c.saved = &saved
	return nil
}

// DefaultKeepRecentTokens is the host's default recent-context budget.
const DefaultKeepRecentTokens = 20000

// ReserveTokens leaves room for the next response when deriving a model budget.
const ReserveTokens = 16384

// ForWindow derives a request policy without mutating the original budget.
func (p CompactionPolicy) ForWindow(window int) CompactionPolicy {
	if window > 0 {
		limit := window - min(ReserveTokens, max(1, window/4))
		if p.MaxInputTokens <= 0 || p.MaxInputTokens > limit {
			p.MaxInputTokens = limit
		}
		if p.Budget <= 0 || p.Budget > limit {
			p.Budget = limit
		}
	}
	return p
}

// estimateTokens is ai.EstimateTextTokens as an int: the module has one
// estimator, and the host's counts are integers.
func estimateTokens(s string) int { return int(ai.EstimateTextTokens(s)) }

// tokenBytes is the longest text, in bytes, that estimateTokens prices at no
// more than tokens: a byte is at most one character.
func tokenBytes(tokens int) int { return int(float64(tokens) * ai.EstimatedCharsPerToken) }
