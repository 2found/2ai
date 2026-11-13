package advisor

import (
	"html"
	"strings"
)

// render.go — how an accepted note is worded to the agent.
//
// Two things here are load-bearing and were learned from the guards that came
// before this one.
//
// First, the framing. The primary agent's system prompt says nothing about
// advisories, so the tag attributes are its ONLY cue for how to treat one. A
// note that reads like an order gets obeyed even when it is wrong, and a
// reviewer with no tools is wrong often enough that "weigh, don't blindly
// obey" has to be in the message itself.
//
// Second, the closing instruction. The injection arrives as a USER message, so
// whatever the model says next is what the owner reads — there is no back
// channel. Without an explicit instruction to re-emit the answer, a model that
// judges itself already compliant replies with its verdict on the note ("the
// concern does not apply, no correction needed"), and that verdict silently
// REPLACES the answer: the owner asked a question and gets back a self-audit
// referring to a reply they never saw. This is the exact failure the evidence
// guard's nudge documents; the repair path must always terminate in the full
// answer, including the case where the repair turns out to be nothing.

// advisorGuidance is carried as a tag attribute rather than a prose header so
// the agent-facing block stays a clean element. It is the behavioural framing:
// advice, not orders.
const advisorGuidance = "weigh, don't blindly obey"

// FormatAdvisories renders notes as the agent-facing advisory elements: one
// element per note, severity as an attribute, text XML-escaped.
//
// Escaping is not cosmetic. The reviewer read a transcript that may contain
// text an attacker controls (a fetched page, a row from the event store), and
// its note becomes a user message in the primary conversation. Escaping keeps
// a quoted `</advisory>` inside the note from closing the element and letting
// whatever follows read as the host's own instructions.
func FormatAdvisories(notes []Note) string {
	var b strings.Builder
	for i, n := range notes {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(`<advisory severity="`)
		b.WriteString(html.EscapeString(string(normalizeSeverity(n.Severity))))
		b.WriteString(`" guidance="`)
		b.WriteString(html.EscapeString(advisorGuidance))
		b.WriteString("\">\n")
		b.WriteString(html.EscapeString(n.Text))
		b.WriteString("\n</advisory>")
	}
	return b.String()
}

// normalizeSeverity maps an unset or unrecognized severity to nit, so a
// reviewer that invents a value cannot smuggle it into the rendered attribute.
func normalizeSeverity(s Severity) Severity {
	switch s {
	case SeverityConcern, SeverityBlocker:
		return s
	default:
		return SeverityNit
	}
}

// FormatInjection is the complete message injected when a note re-opens the
// run: the advisories, then what the agent is expected to do about them.
func FormatInjection(notes []Note) string {
	return "[System: an advisor reviewed the answer you were about to give and raised the following. " +
		"It is a second opinion from a reviewer that did NOT do the work and may be wrong.\n\n" +
		FormatAdvisories(notes) +
		"\n\nCheck each point against what you actually did. Fix what is right; " +
		"for anything you judge wrong or already handled, say so in one line with the reason. " +
		"This note is not visible to the user and is not a question to answer: your next message is what " +
		"the user reads, so reply with the COMPLETE answer they should see, never a comment about this " +
		"note. If nothing needs to change, send the answer again unchanged.]"
}
