package sandbox

import (
	"context"
	"regexp"

	"github.com/2found/2ai/agentcore"
)

// InjectionGuard is a runtime prompt-injection / exfiltration filter installed
// as a BeforeToolCall hook. Where AGT's PromptDefense statically grades a system
// prompt *before* deployment, this scans the actual tool *arguments* the model
// emits *at* runtime — the live vector for indirect injection, where attacker
// text smuggled through retrieved data steers the agent into exfiltrating
// secrets or overriding its instructions.
//
// It is deterministic regex (zero LLM cost), default-deny on match, and feeds
// the block reason back to the model so the run continues safely rather than
// failing silently. It is a cheap second layer behind the Sandbox, not a
// replacement: the sandbox stops a malicious command from reaching anything;
// the guard stops many such commands from ever being issued.
type InjectionGuard struct {
	patterns []*regexp.Regexp
}

// injectionVectors are the runtime signatures, mapped loosely to the OWASP LLM
// Top-10 vectors AGT's promptdefense enumerates. Kept deliberately small and
// high-signal — each one is a phrase that has no legitimate place in a tool
// argument for an analytics/support agent.
var injectionVectors = []string{
	// instruction-override / role-escape
	`(?i)ignore\s+(?:all\s+)?(?:your\s+)?(?:previous|prior|above)\s+instructions`,
	`(?i)disregard\s+(?:all\s+)?(?:previous|prior|the\s+system)`,
	`(?i)you\s+are\s+now\s+(?:a|an|in)\b`,
	`(?i)\bact\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+)?(?:DAN|jailbroken|unrestricted)\b`,
	// system-prompt / instruction exfiltration
	`(?i)(?:reveal|print|repeat|show|dump|leak)\s+(?:me\s+)?(?:your\s+)?(?:system\s+prompt|instructions|the\s+prompt)`,
	// secret / credential exfiltration
	`(?i)(?:print|read|cat|reveal|exfiltrate|send|leak|dump)\b.{0,40}\b(?:env(?:ironment)?\s*(?:vars?|variables?)?|secrets?|api[_\s-]?keys?|credentials?|passwords?)\b`,
	`(?i)/proc/self/environ`,   // the canonical "read my process env" path — high-signal on its own
	`(?i)\bAKIA[0-9A-Z]{16}\b`, // literal AWS access key id
}

// NewInjectionGuard builds a guard over the built-in vectors.
func NewInjectionGuard() *InjectionGuard {
	g := &InjectionGuard{patterns: make([]*regexp.Regexp, 0, len(injectionVectors))}
	for _, v := range injectionVectors {
		g.patterns = append(g.patterns, regexp.MustCompile(v))
	}
	return g
}

// Match reports whether s trips any injection vector (exposed for tests).
func (g *InjectionGuard) Match(s string) bool {
	for _, re := range g.patterns {
		if re.MatchString(s) {
			return true
		}
	}
	return false
}

// Hook returns the BeforeToolCall hook. It blocks any call whose arguments trip
// a vector; the reason is returned to the model so it can correct course.
func (g *InjectionGuard) Hook() agentcore.BeforeToolCall {
	return func(_ context.Context, call agentcore.ToolCall) agentcore.Decision {
		if g.Match(call.Arguments) {
			return agentcore.Blocked(
				"injection guard: tool arguments match a known prompt-injection / " +
					"exfiltration pattern and were blocked")
		}
		return agentcore.Allowed()
	}
}
