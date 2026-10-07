package todo_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/todo"
	"github.com/2found/2ai/ai"
)

const (
	planStepA = "Enumerate the shards in the ledger corpus"
	planStepB = "Reconcile the CLEARING-7742 discrepancy"
	planStepC = "File one report per region"
)

const resumePlanArgs = `{"items":[` +
	`{"content":"` + planStepA + `","status":"completed"},` +
	`{"content":"` + planStepB + `","status":"in_progress"},` +
	`{"content":"` + planStepC + `","status":"pending"}]}`

// A failed native request still returns a checkpoint. A fresh agent and plan
// store must recover the accepted update without replaying its tool effect.
func TestPlanComesBackOnResume(t *testing.T) {
	ctx := context.Background()
	first := todo.NewStore()
	calls := 0
	stream := func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		calls++
		if calls == 1 {
			return ai.ScriptedStream(nativePlanCall("p1", todo.ToolName, resumePlanArgs))(ctx, model, view, options)
		}
		return nil, errors.New("provider died mid-task")
	}
	result, err := newPlanAgent(t, first, stream).Prompt(ctx, "audit the corpus")
	if err == nil || len(result.NativeState) == 0 || !strings.Contains(first.Render(), planStepB) {
		t.Fatalf("failed run did not checkpoint its accepted plan: calls=%d plan=%q err=%v", calls, first.Render(), err)
	}
	second := todo.NewStore()
	requests := 0
	resumed := newPlanAgent(t, second, func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		requests++
		prompt := ai.GetCurrentSystemPrompt(view.Messages())
		for _, want := range []string{"[x] " + planStepA, "[~] " + planStepB, "[ ] " + planStepC} {
			if !strings.Contains(prompt, want) {
				t.Errorf("restored request lost %q", want)
			}
		}
		return ai.ScriptedStream(nativePlanAnswer("carrying on"))(ctx, model, view, options)
	})
	continued, err := resumed.RunNative(ctx, agentcore.NativeRun{State: result.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "carry on"}}})
	if err != nil || requests != 1 || len(continued.Tools) != 0 {
		t.Fatalf("resume replayed work: requests=%d tools=%+v err=%v", requests, continued.Tools, err)
	}
	items := second.List()
	if len(items) != 3 || items[0].Status != todo.StatusCompleted || items[1].Status != todo.StatusInProgress || items[2].Status != todo.StatusPending {
		t.Fatalf("restored plan lost its position: %+v", items)
	}
}

// The consumer chooses whether to continue a checkpoint or start a new task.
// A new task has a new plan store and no inherited checkpoint.
func TestFinishedRunsPlanIsNotInherited(t *testing.T) {
	ctx := context.Background()
	first := todo.NewStore()
	result, err := newPlanAgent(t, first, ai.ScriptedStream(nativePlanCall("p1", todo.ToolName, resumePlanArgs), nativePlanAnswer("audit filed"))).Prompt(ctx, "audit the corpus")
	if err != nil || len(first.List()) != 3 || len(result.NativeState) == 0 {
		t.Fatalf("first run: plan=%+v err=%v", first.List(), err)
	}
	next := todo.NewStore()
	chained := newPlanAgent(t, next, func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		if strings.Contains(ai.GetCurrentSystemPrompt(view.Messages()), todo.ContextPrefix) {
			t.Error("a new task inherited the previous checklist")
		}
		return ai.ScriptedStream(nativePlanAnswer("on it"))(ctx, model, view, options)
	})
	if _, err := chained.Prompt(ctx, "now do something else entirely"); err != nil {
		t.Fatal(err)
	}
	if got := next.Render(); got != "" {
		t.Fatalf("new task inherited plan: %s", got)
	}
}
