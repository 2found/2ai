package agentcore

import (
	"strings"
	"testing"
)

func TestStopGuardNudgeAsksForTheAnswerAgain(t *testing.T) {
	original := []Message{{Role: RoleUser, Content: "Finish the plan."}}
	d := withAnswerReminder(StopDecision{Continue: true, Inject: original})
	if !strings.HasPrefix(d.Inject[0].Content, "Finish the plan.") || !strings.HasSuffix(d.Inject[0].Content, answerReminder) {
		t.Fatalf("nudge = %q", d.Inject[0].Content)
	}
	if original[0].Content != "Finish the plan." {
		t.Fatal("the extension's own message was modified")
	}
	again := withAnswerReminder(d)
	if strings.Count(again.Inject[0].Content, answerReminder) != 1 {
		t.Fatal("reminder appended twice")
	}
}
