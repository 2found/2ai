package ai

import "strings"

// Context windows, and why there is a table here at all.
//
// The compaction budget has to be a number before the first turn, and getting it
// wrong is not symmetric. Too LOW means the loop summarizes earlier than it had
// to — wasteful, slightly lossy, run survives. Too HIGH means the loop never
// compacts and the provider rejects the request outright, which no retry or
// escalation can rescue because the transcript is simply too big. So every
// unknown resolves downward.
//
// The number is sourced in this order, best first:
//
//  1. the workspace's per-tier override (an operator who knows their endpoint),
//  2. the vendor's own list-models response — Gemini's inputTokenLimit, an
//     OpenAI-compatible server's context_length / max_model_len,
//  3. this table, for the vendors whose API does not report one at all
//     (Anthropic's /v1/models and OpenAI's /models both omit it),
//  4. nothing, and the caller falls back to its own configured ceiling.
//
// This is deliberately NOT a model catalog: it never contributes an id, only
// annotates one the vendor already named, and a model missing from it degrades
// to (4) rather than to a wrong number. Keeping it here rather than in
// agentcore is the same boundary doc.go draws — vendor specifics live in this
// package, and the kernel stays free of them.
//
// Entries are matched longest-prefix, so a family default can sit alongside the
// specific members that differ from it.
var contextWindows = map[string]int{
	// Devin GetCliModelConfigs reports 262000 for swe-2-{medium,high,max}
	// (verified 2026-10-06). The CLI alias swe-2 resolves to that same family.
	"swe-2": 262_000,
	// Anthropic reports no window. Every currently-served Claude is at least
	// 200k; the 1M-token variants are beta and opt-in, so the family default
	// stays at the floor.
	"claude-":        200_000,
	"claude-2":       100_000,
	"claude-instant": 100_000,

	// OpenAI reports no window. gpt-4 is the trap: the bare model is 8k while
	// every later member of the family is 128k or more, which longest-prefix
	// matching resolves correctly only because the longer keys are present.
	"gpt-3.5":     16_385,
	"gpt-4":       8_192,
	"gpt-4-32k":   32_768,
	"gpt-4-turbo": 128_000,
	"gpt-4o":      128_000,
	"gpt-4.1":     1_047_576,
	"gpt-5":       400_000,
	"o1":          200_000,
	"o3":          200_000,
	"o4":          200_000,

	// Gemini reports inputTokenLimit live, so these are only reached through an
	// OpenAI-compatible proxy that strips it.
	"gemini-":        1_000_000,
	"gemini-1.5-pro": 2_000_000,
	"gemini-2.5-pro": 1_000_000,
	"gemini-1.0":     32_768,

	// xAI serves an OpenAI-compatible surface that does report context_length,
	// so these are a backstop for proxies that drop it.
	"grok-":  131_072,
	"grok-4": 256_000,
}

// ContextWindowFor returns the input context window for a model id, or 0 when
// this package has nothing better than a guess. Zero is a real answer — "no
// opinion" — and callers must treat it as such rather than as a window of size
// zero.
//
// Vendor is accepted for future per-vendor disambiguation and is currently only
// normalized; ids are matched on their own because a workspace reaches the same
// model through several vendor ids (openai, openai-compat, a router).
func ContextWindowFor(vendor, model string) int {
	id := normalizeModelID(model)
	if id == "" {
		return 0
	}
	best, bestLen := 0, -1
	for prefix, window := range contextWindows {
		if strings.HasPrefix(id, prefix) && len(prefix) > bestLen {
			best, bestLen = window, len(prefix)
		}
	}
	return best
}

// normalizeModelID strips the decorations a model id picks up in transit: the
// "models/" resource prefix Gemini returns, and the "vendor/" namespace routers
// like OpenRouter prepend ("anthropic/claude-sonnet-4"). Without this, a routed
// id matches nothing and silently loses its window.
func normalizeModelID(model string) string {
	id := strings.ToLower(strings.TrimSpace(model))
	id = strings.TrimPrefix(id, "models/")
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	return id
}
