package ai

import (
	"regexp"
	"slices"
	"strings"
)

// This file and window.go are the model tables a host reads when a provider's
// own list says nothing: the effort levels each family serves and the input
// size past which it bills more. A vendor that reports either takes over (see
// Model). Sources: oh-my-pi's model catalog (packages/catalog/src/models.json,
// 2026-11-17), as the model's maker serves it, and the vendors' price lists. The
// model-catalog skill (.claude/skills/model-catalog) keeps them current.

// ReasoningEffortLevels are the reasoning efforts a host may ask for, weakest
// first. "none" turns thinking off.
var ReasoningEffortLevels = []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}

var (
	effortsMinimalToHigh  = []string{"minimal", "low", "medium", "high"}
	effortsMinimalToXHigh = []string{"minimal", "low", "medium", "high", "xhigh"}
	effortsLowToHigh      = []string{"low", "medium", "high"}
	effortsLowToXHigh     = []string{"low", "medium", "high", "xhigh"}
	effortsLowToMax       = []string{"low", "medium", "high", "xhigh", "max"}
	effortsLowHighMax     = []string{"low", "high", "max"}
	// A model that reasons with no level to choose refuses one.
	effortsNone = []string{}
)

// reasoningEfforts are the levels a model family serves, matched longest
// prefix first, so a family lists only the members that differ from it. A
// level a model does not serve is not always mapped by its provider: OpenAI
// refuses max for GPT-5.5, and the request then loses its effort.
var reasoningEfforts = map[string][]string{
	"claude-opus-4-5":   effortsMinimalToXHigh,
	"claude-sonnet-4-5": effortsMinimalToXHigh,
	"claude-haiku-4-5":  effortsMinimalToXHigh,
	"claude-opus-4-6":   {"low", "medium", "high", "max"},
	"claude-sonnet-4-6": effortsLowToHigh,
	"claude-opus-4-7":   effortsLowToMax,
	"claude-opus-4-8":   effortsLowToMax,
	"claude-opus-5":     effortsLowToMax,
	"claude-sonnet-5":   effortsLowToMax,
	"claude-haiku-5":    effortsLowToMax,
	"claude-fable-":     effortsLowToMax,
	"claude-mythos-":    effortsLowToMax,

	"gpt-5":              effortsMinimalToHigh,
	"gpt-5-chat":         effortsNone,
	"gpt-5.1-codex-mini": {"medium", "high"},
	"gpt-5.2":            effortsLowToXHigh,
	"gpt-5.3":            effortsLowToXHigh,
	"gpt-5.3-chat":       effortsNone,
	"gpt-5.4":            effortsLowToXHigh,
	"gpt-5.5":            effortsLowToXHigh,
	"gpt-5.6":            effortsLowToMax,
	"gpt-6":              effortsLowToMax,
	"daybreak-":          effortsLowToMax,
	"gpt-daybreak-":      effortsLowToMax,

	"gemini-2.5":        effortsMinimalToHigh,
	"gemini-3-flash":    effortsMinimalToHigh,
	"gemini-3-pro":      {"low", "high"},
	"gemini-3.1-flash":  effortsMinimalToHigh,
	"gemini-3.1-pro":    {"low", "high"},
	"gemini-3.5-flash":  effortsMinimalToHigh,
	"gemini-3.6-flash":  effortsMinimalToHigh,
	"gemini-3.7":        effortsLowToHigh,
	"gemini-3.8":        effortsLowToHigh,
	"gemini-4":          effortsLowToHigh,
	"gemini-4-hydrogen": effortsLowToXHigh,
	"gemini-flash-":     effortsMinimalToHigh,
	"gemini-pro-":       effortsMinimalToHigh,
	"gemma-4":           {"high"},

	"grok-3-mini":           effortsMinimalToHigh,
	"grok-4":                effortsNone,
	"grok-4.20":             effortsNone,
	"grok-4.20-multi-agent": effortsMinimalToXHigh,
	"grok-4.3":              effortsMinimalToHigh,
	"grok-4.5":              effortsMinimalToHigh,
	"grok-4.6":              effortsMinimalToXHigh,
	"grok-4.7":              effortsMinimalToXHigh,
	"grok-build":            effortsNone,
	"grok-code":             effortsNone,

	"deepseek-v4":    effortsLowHighMax,
	"deepseek-flash": effortsLowHighMax,
	"deepseek-pro":   effortsLowHighMax,

	// Kimi K2 itself (instruct, turbo) does not reason; its thinking members
	// and K2.5 onward do.
	"kimi-k2":          effortsNone,
	"kimi-k2-thinking": effortsMinimalToHigh,
	"kimi-k2.":         effortsMinimalToHigh,
	"kimi-k3":          effortsLowHighMax,

	"glm-4.5": effortsMinimalToXHigh,
	"glm-4.6": effortsMinimalToXHigh,
	"glm-4.7": effortsMinimalToXHigh,
	"glm-5":   effortsMinimalToXHigh,
	"glm-5.2": {"high", "max"},
	"glm-5.3": effortsLowHighMax,

	"minimax-m2":   effortsLowToHigh,
	"minimax-m3":   effortsLowToHigh,
	"minimax-m3.1": effortsLowToMax,

	"qwen3.5":       effortsMinimalToHigh,
	"qwen3.6":       effortsMinimalToHigh,
	"qwen3.7":       effortsMinimalToHigh,
	"qwen3.8-max":   effortsMinimalToHigh,
	"qwen3.8-flash": effortsMinimalToXHigh,
}

// ReasoningEffortsFor are the effort levels model serves, weakest first, and
// whether the table knows it; an empty list is a model that reasons without a
// level. "none" is not listed: whether a model can stop thinking is its
// provider's to answer. Vendor is reserved for a vendor that serves a model
// with other levels; none does yet, since a proxy maps a level it lacks.
func ReasoningEffortsFor(vendor, model string) ([]string, bool) {
	_ = vendor
	levels, ok := longestPrefixOf(reasoningEfforts, catalogModelID(model))
	if !ok {
		return nil, false
	}
	return slices.Clone(levels), true
}

// ClampReasoningEffort is the effort to send model for a requested one: the
// request when the model serves it or is unknown, else the strongest level it
// serves below the request, else its weakest. "", "none" and "off" pass
// unchanged; a model that reasons without a level is sent none (""). Send the
// result rather than the request: a refused level loses the effort altogether.
func ClampReasoningEffort(vendor, model, effort string) string {
	levels, ok := ReasoningEffortsFor(vendor, model)
	if !ok || effort == "" {
		return effort
	}
	if len(levels) == 0 {
		return ""
	}
	rank := slices.Index(ReasoningEffortLevels, effort)
	if effort == "none" || effort == "off" || rank < 0 {
		return effort
	}
	clamped := ""
	for _, level := range levels {
		if level == effort {
			return effort
		}
		if slices.Index(ReasoningEffortLevels, level) < rank {
			clamped = level
		}
	}
	if clamped == "" {
		return levels[0]
	}
	return clamped
}

// LongContextThresholdFor is the input size, in tokens, past which model
// bills every token of the request at a higher rate, or 0 for a flat price or
// an unknown model. A window says how much a model can hold; this says how much
// it holds at its list price, so a host that summarizes a long run should do
// so before it. Vendor is reserved as for ReasoningEffortsFor.
func LongContextThresholdFor(vendor, model string) int {
	_ = vendor
	id := catalogModelID(model)
	switch {
	case strings.HasPrefix(id, "claude-haiku-5"):
		// Input above 100k bills at five times the rate.
		return 100_000
	case strings.HasPrefix(id, "gpt-5"), strings.HasPrefix(id, "gpt-6"), strings.HasPrefix(id, "daybreak-blue"), strings.HasPrefix(id, "gpt-daybreak-blue"):
		// Twice the rate above 272k on the 1M-window members; the smaller
		// windows take at most 272k of input anyway.
		return 272_000
	case strings.HasPrefix(id, "gemini-1.5-pro"):
		return 128_000
	case strings.HasPrefix(id, "gemini-2.5-pro"), strings.HasPrefix(id, "gemini-3-pro"), strings.HasPrefix(id, "gemini-3.1-pro"):
		// Google's own price list; the catalog does not tier Gemini Pro.
		return 200_000
	case grokLongContext.MatchString(id), strings.HasPrefix(id, "muse-ember"):
		// Grok 4.20 and later, and Meta's Muse Ember: twice (three times)
		// the rate above 200k.
		return 200_000
	case strings.HasPrefix(id, "qwen3.7-plus"):
		return 256_000
	case strings.HasPrefix(id, "minimax-m3"):
		return 512_000
	}
	return 0
}

var grokLongContext = regexp.MustCompile(`^grok-(?:4\.[2-9]|build|composer)`)

// catalogModelID is a model id as these tables match it: normalizeModelID,
// and without a Bedrock "anthropic." (or "us.anthropic.") prefix.
func catalogModelID(model string) string {
	id := normalizeModelID(model)
	if i := strings.Index(id, "anthropic."); i >= 0 {
		id = id[i+len("anthropic."):]
	}
	return id
}

func longestPrefixOf[V any](table map[string]V, id string) (V, bool) {
	var best V
	bestLen := -1
	for prefix, v := range table {
		if strings.HasPrefix(id, prefix) && len(prefix) > bestLen {
			best, bestLen = v, len(prefix)
		}
	}
	return best, bestLen >= 0
}
