package agentcore

import (
	"fmt"
	"slices"
)

const (
	budgetExhaustedSteer = "Your run budget for this period has been exhausted. Do not call any more tools. Summarize the progress you have made so far and any recommended next steps in a few sentences, then stop."
	maxTurnsSteer        = "You have reached this run's step limit and cannot do any more work. Do not call any more tools. Answer the original question as well as the evidence you already have allows: state what you found, say how confident you are, and name what you would check next. If what you have is not enough to answer, say that plainly and say what is missing."
	maxToolCallsSteer    = "You have used every tool call this run is allowed. Do not call any more tools. Answer the original question from the results you already have: state what you found, say how confident you are, and name what you would check next. If what you have is not enough to answer, say that plainly and say what is missing."
)

// budgetNoticeAt is how few charged calls a run has left when each tool result
// starts saying so. The model cannot see the ceiling otherwise: a support
// agent with seven calls spent them all on lookups and, at the wrap-up, could
// no longer file the lead the customer had asked it to.
const budgetNoticeAt = 2

func withBudgetNotice(m Message, left int64) Message {
	notice := fmt.Sprintf("[Tool calls left in this run: %d. After that you answer without tools, so spend what is left on what only a tool can do.]", left)
	if m.Content == "" && len(m.ContentParts) > 0 {
		m.ContentParts = append(slices.Clip(m.ContentParts), ContentPart{Type: ContentPartText, Text: notice})
		return m
	}
	if m.Content != "" {
		m.Content += "\n\n"
	}
	m.Content += notice
	return m
}

// finalizeSteer is the wrap-up instruction for the ceiling that tripped. The
// reason doubles as the run's StopReason, so a caller reading the trace sees the
// same cause the model was told about.
func finalizeSteer(reason string) string {
	switch reason {
	case "max_turns":
		return maxTurnsSteer
	case "max_tool_calls":
		return maxToolCallsSteer
	default:
		return budgetExhaustedSteer
	}
}

// finalizeNote is the progress line a watching viewer sees when the wrap-up
// starts. It names the ceiling, because "summarizing" on its own reads as the
// agent choosing to stop rather than being stopped.
func finalizeNote(reason string) string {
	switch reason {
	case "max_turns":
		return "Step limit reached — writing up what I found so far."
	case "max_tool_calls":
		return "Tool-call limit reached — writing up what I found so far."
	default:
		return "Budget reached — summarizing and stopping."
	}
}

func isBookkeeping(tools *ToolSet, name string) bool {
	if tools == nil {
		return false
	}
	t, ok := tools.Get(name)
	if !ok {
		return false
	}
	bk, ok := t.(BookkeepingTool)
	return ok && bk.Bookkeeping()
}
