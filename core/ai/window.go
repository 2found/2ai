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
	// Anthropic reports no window. Claude 4.5 and earlier are 200k; 4.6 onward
	// serve 1M without a beta header and are listed further down.
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

	// The entries below come from oh-my-pi's model catalog (2026-11-17): the
	// window the model's maker serves, the smaller where its own plans differ.
	// A vendor that serves a model with another window has its own table
	// (vendorContextWindows); a router reports its own through discovery.
	"claude-opus-4-6":   1_000_000,
	"claude-opus-4-7":   1_000_000,
	"claude-opus-4-8":   1_000_000,
	"claude-opus-5":     1_000_000,
	"claude-sonnet-4-6": 1_000_000,
	"claude-sonnet-5":   1_000_000,
	"claude-haiku-5":    1_000_000,
	"claude-fable-":     1_000_000,
	"claude-mythos-":    1_000_000,

	"gpt-5-chat":          128_000,
	"gpt-5.1-chat":        128_000,
	"gpt-5.2-chat":        128_000,
	"gpt-5.3-chat":        128_000,
	"gpt-5.3-codex":       272_000,
	"gpt-5.3-codex-spark": 128_000,
	"gpt-5.4":             1_000_000,
	"gpt-5.4-mini":        400_000,
	"gpt-5.4-nano":        400_000,
	"gpt-5.5":             1_050_000,
	"gpt-5.6":             1_050_000,
	"gpt-5.6-cyber":       400_000,
	"gpt-6":               1_050_000,

	"grok-4.20":     1_000_000,
	"grok-4.3":      1_000_000,
	"grok-4.5":      500_000,
	"grok-4.6":      500_000,
	"grok-4.7":      500_000,
	"grok-build":    256_000,
	"grok-code":     256_000,
	"grok-composer": 200_000,

	"deepseek-v4":              1_000_000,
	"deepseek-v4-flash-free":   200_000,
	"deepseek-v4.1-flash-free": 200_000,
	"deepseek-flash":           1_000_000,
	"deepseek-pro":             1_000_000,

	"kimi-k2":      262_144,
	"kimi-k2-0711": 131_072,
	"kimi-k3":      1_048_576,

	"glm-4.5":            131_072,
	"glm-4.5v":           64_000,
	"glm-4.6":            200_000,
	"glm-4.6v":           128_000,
	"glm-4.7":            200_000,
	"glm-5":              204_800,
	"glm-5.1":            200_000,
	"glm-5.2":            1_000_000,
	"glm-5.3":            1_000_000,
	"glm-5.3-flash-free": 200_000,

	"minimax-m2":      204_800,
	"minimax-m3":      512_000,
	"minimax-m3-free": 200_000,
	"minimax-m3.1":    1_000_000,

	"qwen3.5-plus":  1_000_000,
	"qwen3.6-flash": 1_000_000,
	"qwen3.6-plus":  1_000_000,
	"qwen3.7":       1_000_000,
	"qwen3.8-flash": 1_000_000,
	"qwen3.8-max":   983_616,

	"swe-1": 200_000,
}

// vendorContextWindows are the windows a vendor serves that differ from the
// model's own, matched before contextWindows. A Claude Code login never sends
// the 1M-context beta (a subscription has no long-context credit), so Claude
// 4.x keeps 200k there; the 5.x families are 1M without it.
var vendorContextWindows = map[string]map[string]int{
	VendorClaudeCode: {
		"claude-":         200_000,
		"claude-opus-5":   1_000_000,
		"claude-sonnet-5": 1_000_000,
		"claude-haiku-5":  1_000_000,
		"claude-fable-":   1_000_000,
		"claude-mythos-":  1_000_000,
	},
	VendorGoogleAntigravity: {
		"claude-":           200_000,
		"claude-opus-4-6":   250_000,
		"claude-sonnet-4-6": 250_000,
		"claude-opus-5":     1_000_000,
		"claude-sonnet-5":   1_000_000,
	},
	VendorOpenAICodex: {
		"gpt-5":        272_000,
		"gpt-5.6":      1_000_000,
		"gpt-6":        272_000,
		"gpt-6.1":      500_000,
		"gpt-daybreak": 272_000,
	},
}

// ContextWindowFor returns the input context window for a model id, or 0 when
// this package has nothing better than a guess. Zero is a real answer — "no
// opinion" — and callers must treat it as such rather than as a window of size
// zero.
//
// Vendor picks a vendor's own table first (vendorContextWindows); otherwise ids
// are matched on their own because a workspace reaches the same model through
// several vendor ids (openai, openai-compat, a router).
func ContextWindowFor(vendor, model string) int {
	id := normalizeModelID(model)
	if id == "" {
		return 0
	}
	if window := longestPrefix(vendorContextWindows[strings.ToLower(strings.TrimSpace(vendor))], id); window > 0 {
		return window
	}
	return longestPrefix(contextWindows, id)
}

// longestPrefix is the value of the longest key of table that id starts
// with, or 0.
func longestPrefix(table map[string]int, id string) int {
	best, bestLen := 0, -1
	for prefix, n := range table {
		if strings.HasPrefix(id, prefix) && len(prefix) > bestLen {
			best, bestLen = n, len(prefix)
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
