package advisor

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// emission.go — the policy gate between what the reviewer model SAYS and what
// the primary agent actually reads.
//
// It exists because reviewer models do not obey the prose rules in their own
// system prompt. oh-my-pi issue #3520 recorded one session where the advisor
// made 309 `advise` calls covering 92 unique notes — 114x "Stop.", 52x "No
// issue; continue.", 41x "Done." — flooding the primary transcript with
// content-free advisories after the task was already complete. The fix is to
// make the rules load-bearing in code instead of prose: drop duplicates,
// content-free self-talk, and over-budget notes at the acceptance boundary.
//
// The gate is deliberately INVISIBLE to the reviewer. Telling a model its note
// was suppressed teaches it to rephrase the same useless note to get past the
// filter ("Stop." then "Halt." then "Stop now."), which defeats the dedupe and
// costs a round trip to do it.

// noteCapacity bounds the dedupe history. The pathological session above held
// 92 unique notes; 4096 leaves headroom while staying tiny.
const noteCapacity = 4096

// MaxNoteRunes clamps one note's length. The reviewer reads a transcript that
// may contain attacker-controlled text and its note becomes a user message in
// the primary conversation, so an unbounded note is an unbounded injection.
const MaxNoteRunes = 2000

// suppressedPhrases are normalized notes that carry no actionable content.
// Each key must be the output of NormalizeNote so one membership check covers
// every casing and punctuation variant ("Stop.", "stop", "STOP!").
//
// The list is deliberately conservative — only short, content-free filler. A
// genuine blocker like "Stop: the revenue figure sums a filtered and an
// unfiltered query" does not match, because normalization does not truncate.
var suppressedPhrases = map[string]bool{
	// Self-stop noise — "stop" with no reason is not advice.
	"stop":      true,
	"stop here": true,
	"stop now":  true,
	"halt":      true,
	"abort":     true,
	// Completion self-talk — the agent already knows it finished.
	"done":          true,
	"task done":     true,
	"task complete": true,
	"complete":      true,
	"finished":      true,
	"ok":            true,
	"okay":          true,
	"ok done":       true,
	// "Nothing to flag" — silence is the correct way to say that.
	"no issue":                 true,
	"no issues":                true,
	"no issue continue":        true,
	"no concerns":              true,
	"no concern":               true,
	"nothing to add":           true,
	"nothing to flag":          true,
	"nothing to report":        true,
	"no notes":                 true,
	"no further input":         true,
	"no further input needed":  true,
	"no further advice":        true,
	"no further advice needed": true,
	// Endorsements — equivalent to silence.
	"lgtm":              true,
	"looks good":        true,
	"all good":          true,
	"on track":          true,
	"agent on track":    true,
	"agent is on track": true,
	"continue":          true,
	"carry on":          true,
	"no action needed":  true,
	"no changes needed": true,
}

// NormalizeNote folds a note to its identity key: lowercase, NFKC-normalized,
// every run of non-letter/non-digit characters collapsed to a single space,
// trimmed. "Stop.", "*Stop*", and "  stop  " all key to "stop", while
// "No issue; continue." keys to "no issue continue".
//
// Exported because the dedupe key is part of the plugin's observable contract:
// a caller recording notes on a run needs the same identity the guard used.
func NormalizeNote(note string) string {
	var b strings.Builder
	b.Grow(len(note))
	space := false
	for _, r := range norm.NFKC.String(strings.ToLower(note)) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			if space && b.Len() > 0 {
				b.WriteByte(' ')
			}
			space = false
			b.WriteRune(r)
			continue
		}
		space = true
	}
	return b.String()
}

// severityRank orders the severities so the guard can tell a real escalation
// (nit -> concern -> blocker) from a verbatim repeat. An unset severity ranks
// as a nit, matching how the reviewer's schema treats an omitted field.
func severityRank(s Severity) int {
	switch s {
	case SeverityBlocker:
		return 3
	case SeverityConcern:
		return 2
	default:
		return 1
	}
}

// EmissionGuard decides which reviewer notes reach the primary agent.
//
// It enforces, in order: the length clamp, the content-free phrase filter,
// run-scoped dedupe by normalized text (FIFO-evicted at noteCapacity),
// escalation-rank dedupe (a repeat passes only when its severity strictly
// exceeds the rank already delivered for that text), and a per-review breadth
// budget. Suppressed noise never consumes the budget — a junk note must not
// burn the slot for a real concern behind it in the same review.
//
// The zero value is not usable; construct with NewEmissionGuard.
type EmissionGuard struct {
	seen         map[string]int // normalized text -> highest delivered severity rank
	order        []string       // insertion order, for FIFO eviction
	perReview    int            // notes accepted so far in this review
	maxPerReview int
}

// NewEmissionGuard builds a guard accepting at most maxPerReview notes per
// review round. A non-positive maxPerReview means unbounded breadth, which is
// only ever right for a test.
func NewEmissionGuard(maxPerReview int) *EmissionGuard {
	return &EmissionGuard{seen: make(map[string]int), maxPerReview: maxPerReview}
}

// BeginReview clears the per-review budget. Called once before each reviewer
// consultation; the dedupe history deliberately survives, because "you already
// said that last round" is the whole point of a second round.
func (g *EmissionGuard) BeginReview() { g.perReview = 0 }

// Reset drops all state. Called when the conversation the notes were about is
// rewritten (compaction, a rebase), so a re-primed reviewer may re-raise an
// issue it already raised against a transcript that no longer exists.
func (g *EmissionGuard) Reset() {
	g.seen = make(map[string]int)
	g.order = g.order[:0]
	g.perReview = 0
}

// Accept reports whether a note should reach the primary, and returns the note
// as it should be delivered (length-clamped). On true the guard has already
// recorded it: the budget is spent and the text is in the dedupe history.
func (g *EmissionGuard) Accept(n Note) (Note, bool) {
	n.Text = clampRunes(strings.TrimSpace(n.Text), MaxNoteRunes)
	key := NormalizeNote(n.Text)
	if key == "" || suppressedPhrases[key] {
		return Note{}, false
	}
	rank := severityRank(n.Severity)
	if prev, ok := g.seen[key]; ok && rank <= prev {
		return Note{}, false
	}
	if g.maxPerReview > 0 && g.perReview >= g.maxPerReview {
		return Note{}, false
	}
	g.perReview++
	if _, ok := g.seen[key]; !ok {
		g.order = append(g.order, key)
		if len(g.order) > noteCapacity {
			delete(g.seen, g.order[0])
			g.order = g.order[1:]
		}
	}
	g.seen[key] = rank
	return n, true
}

// clampRunes truncates to at most max runes, counting runes rather than bytes
// so a clamp never splits a multi-byte character (the reviewer writes
// Vietnamese).
func clampRunes(s string, max int) string {
	if max <= 0 {
		return s
	}
	n := 0
	for i := range s {
		n++
		if n > max {
			return s[:i]
		}
	}
	return s
}
