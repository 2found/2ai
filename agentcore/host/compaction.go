package host

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/2found/2ai/ai/protocol"
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
	Task      string
	Summarize func(context.Context, json.RawMessage, string) (string, protocol.Usage, error)
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
	record   func(context.Context, json.RawMessage) error
	usage    func(protocol.Usage)
}

// Request transforms are called serially by the original loop. A saved summary
// refers to a prefix of this exact view (after native host hooks), never to a Go
// text projection. On mismatch the full native context is the safe fallback.
func (c *Compactor) Transform(ctx context.Context, raw json.RawMessage) (out json.RawMessage) {
	return c.TransformWithPolicy(ctx, raw, c.config)
}

// TransformWithPolicy applies the effective policy for this request, retaining
// the compactor's checkpoint and original policy for subsequent model changes.
func (c *Compactor) TransformWithPolicy(ctx context.Context, raw json.RawMessage, policy CompactionPolicy) (out json.RawMessage) {
	out = raw
	defer func() {
		if recover() != nil {
			out = raw
		}
	}()
	if ctx.Err() != nil || ValidateMessages(raw) != nil {
		return
	}
	var original []json.RawMessage
	if json.Unmarshal(raw, &original) != nil {
		return
	}
	view := original
	baseCount, headCount := 0, 0
	if saved := c.saved; saved != nil && saved.Revision == c.revision && saved.PrefixCount <= len(original) && PrefixDigest(original[:saved.PrefixCount]) == saved.PrefixDigest {
		var boundary struct{ Role string }
		if saved.PrefixCount < len(original) {
			_ = json.Unmarshal(original[saved.PrefixCount], &boundary)
		}
		if saved.PrefixCount == len(original) || boundary.Role == "assistant" || boundary.Role == "user" {
			view = summaryViewWithTask(original, saved.PrefixCount, saved.Message, policy.Task)
			baseCount = saved.PrefixCount
			headCount = len(view) - (len(original) - baseCount)
		}
	}
	out, _ = json.Marshal(view)
	cut := compactionCut(view, policy.Budget, policy.KeepRecent)
	if policy.Force && len(view) > headCount {
		lastIndex := len(view) - 1
		var last struct{ Role string }
		_ = json.Unmarshal(view[lastIndex], &last)
		// Host reminders may follow the current user/assistant turn. Preserve
		// those systems while choosing the same completed-turn boundary.
		for last.Role == "system" && lastIndex > headCount {
			lastIndex--
			_ = json.Unmarshal(view[lastIndex], &last)
		}
		if last.Role == "assistant" || last.Role == "toolResult" {
			cut = lastIndex + 1
		} else if last.Role == "user" {
			// When an oversized historical fact dominates the retained tail,
			// fold it as well while retaining the latest user message verbatim.
			// A lone initial request is never summarized away.
			for _, raw := range view[headCount:lastIndex] {
				var earlier struct{ Role string }
				_ = json.Unmarshal(raw, &earlier)
				if earlier.Role == "user" || earlier.Role == "assistant" {
					cut = lastIndex
					break
				}
			}
		}
	}
	if cut <= headCount || policy.Summarize == nil {
		return
	}
	prefix, _ := json.Marshal(view[:cut])
	text, usage, err := policy.Summarize(ctx, prefix, c.revision)
	if c.usage != nil {
		c.usage(usage)
	}
	if err != nil || ctx.Err() != nil || !validSummaryText(text) {
		return
	}
	count := baseCount + cut - headCount
	message, _ := json.Marshal(map[string]any{"role": "user", "content": "[Earlier work summary]\n" + strings.TrimSpace(text), "agentrayContextSummary": uuid.NewString(), "timestamp": time.Now().UnixMilli()})
	next := summaryViewWithTask(original, count, message, policy.Task)
	nextRaw, _ := json.Marshal(next)
	// A summary which grows the request is not compaction. Keep the last useful
	// view while still accounting for the auxiliary model work already performed.
	if len(nextRaw) >= len(out) {
		return
	}
	saved := Summary{Revision: c.revision, PrefixCount: count, PrefixDigest: PrefixDigest(original[:count]), Message: message}
	payload, _ := json.Marshal(saved)
	if c.record != nil && c.record(ctx, payload) != nil {
		return
	}
	c.saved = &saved
	return nextRaw
}

// SummaryView replaces a validated prefix while preserving its system messages.
// The caller must supply a cut within the message slice and outside a tool batch.
func SummaryView(messages []json.RawMessage, cut int, summary json.RawMessage) []json.RawMessage {
	kept := []json.RawMessage{}
	for _, raw := range messages[:cut] {
		var header struct{ Role string }
		_ = json.Unmarshal(raw, &header)
		if header.Role == "system" {
			kept = append(kept, raw)
		}
	}
	kept = append(kept, summary)
	return append(kept, messages[cut:]...)
}

func summaryViewWithTask(messages []json.RawMessage, cut int, summary json.RawMessage, task string) []json.RawMessage {
	view := SummaryView(messages, cut, summary)
	if task == "" {
		return view
	}
	// Insert before the retained tail so later user messages keep their order.
	index := len(view) - (len(messages) - cut)
	pinned, _ := json.Marshal(map[string]any{"role": "user", "content": "[Current request — verbatim]\n" + task})
	view = append(view[:index:index], append([]json.RawMessage{pinned}, view[index:]...)...)
	return view
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
	base, _ := json.Marshal(summaryViewWithTask(messages, len(messages), message, task))
	available := budget - ContextTokens(base) - 64
	if available < 128 {
		return 0, &ContextBudgetError{Estimated: ContextTokens(base) + 128, Budget: budget}
	}
	return min(4096, budget/4, available), nil
}

func validSummaryText(text string) bool {
	if strings.TrimSpace(strings.TrimPrefix(text, "[Earlier work summary]")) == "" {
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
func compactionCut(messages []json.RawMessage, budget, keep int) int {
	if budget <= 0 {
		return 0
	}
	raw, _ := json.Marshal(messages)
	if ContextTokens(raw) <= budget || ValidateMessages(raw) != nil {
		return 0
	}
	keep = min(max(1, keep), max(1, budget/2))
	roles := make([]string, len(messages))
	priorAssistant := make([]bool, len(messages))
	priorUsers := make([]int, len(messages))
	users := 0
	seen := false
	for i, raw := range messages {
		var m struct{ Role string }
		_ = json.Unmarshal(raw, &m)
		roles[i] = m.Role
		priorAssistant[i] = seen
		priorUsers[i] = users
		seen = seen || m.Role == "assistant"
		if m.Role == "user" {
			users++
		}
	}
	recent := 0
	for i := len(messages) - 1; i > 0; i-- {
		recent += estimateTokens(string(messages[i]))
		if recent >= keep && ((priorAssistant[i] && (roles[i] == "assistant" || roles[i] == "user")) || (roles[i] == "user" && priorUsers[i] > 0)) {
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

func estimateTokens(s string) int { return (len(s) + 3) / 4 }
