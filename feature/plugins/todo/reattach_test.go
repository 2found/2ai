package todo_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/core/agentcore/host"
	"github.com/2found/2ai/core/ai"
	"github.com/2found/2ai/feature/plugins/todo"
)

// A summary that folds away the turn which showed the plan carries the plan
// after its text, so the compacted request itself has it — not a reminder one
// step later.
func TestSummaryReattachesThePlanItFoldedAway(t *testing.T) {
	store := todo.NewStore()
	first := ai.ScriptedStream(
		nativePlanCall("p1", todo.ToolName, `{"items":[{"content":"read the schema","status":"completed"},{"content":"write the query","status":"in_progress"}]}`),
		nativePlanAnswer("planned"),
	)
	a := newPlanAgent(t, store, first)
	planned, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "do the work"}}})
	if err != nil {
		t.Fatal(err)
	}
	var views [][]ai.Message
	b := newPlanAgent(t, store, func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		views = append(views, view.Messages())
		return ai.ScriptedStream(nativePlanAnswer("continuing"))(ctx, model, view, options)
	})
	input := []agentcore.Message{}
	for i := 0; i < 10; i++ {
		input = append(input, agentcore.Message{Role: agentcore.RoleUser, Content: strings.Repeat("old source evidence ", 100)})
	}
	input = append(input, agentcore.Message{Role: agentcore.RoleUser, Content: "continue"})
	summaries := 0
	_, err = b.RunNative(context.Background(), agentcore.NativeRun{State: planned.NativeState, Input: input, Compaction: &host.CompactionPolicy{Budget: 1200, KeepRecent: 100, Summarize: func(context.Context, json.RawMessage, string) (string, agentcore.Usage, error) {
		summaries++
		return "Earlier evidence.", agentcore.Usage{}, nil
	}}})
	if err != nil || summaries != 1 || len(views) != 1 {
		t.Fatalf("requests=%d summaries=%d err=%v", len(views), summaries, err)
	}
	carriers := 0
	for _, message := range views[0] {
		raw, _ := json.Marshal(message)
		if text := string(raw); strings.Contains(text, todo.ContextPrefix) {
			carriers++
			if !strings.Contains(text, "agentrayContextSummary") || !strings.Contains(text, "[~] write the query") {
				t.Fatalf("the plan rides outside the summary: %s", text)
			}
		}
	}
	if carriers != 1 {
		t.Fatalf("the compacted request carries the plan %d times, want once in the summary", carriers)
	}
}

// A plan the summarized messages never showed is not reattached (the retained
// tail or the next reminder has it), and one too large for the room is left to
// the reminder rather than cut.
func TestPlanReattachNeedsTheFoldedPlanAndRoom(t *testing.T) {
	store := todo.NewStore()
	store.Set([]todo.Item{{Content: "write the query", Status: todo.StatusInProgress}})
	ext, err := todo.With(store).BeginRun(context.Background(), agentcore.RunInfo{})
	if err != nil {
		t.Fatal(err)
	}
	reattacher, ok := ext.(agentcore.CompactionReattacher)
	if !ok {
		t.Fatal("the plan does not ride the compaction reattach seam")
	}
	showing, _ := json.Marshal(map[string]any{"role": "toolResult", "toolCallId": "p", "toolName": todo.ToolName, "content": []any{map[string]any{"type": "text", "text": store.Render()}}})
	unrelated, _ := json.Marshal(map[string]any{"role": "user", "content": "old evidence"})
	if got := reattacher.ReattachAfterCompaction(context.Background(), []json.RawMessage{showing}, 1000); !strings.HasPrefix(got, todo.ContextPrefix) {
		t.Fatalf("folded plan not reattached: %q", got)
	}
	if got := reattacher.ReattachAfterCompaction(context.Background(), []json.RawMessage{unrelated}, 1000); got != "" {
		t.Fatalf("a plan the summary did not fold was reattached: %q", got)
	}
	if got := reattacher.ReattachAfterCompaction(context.Background(), []json.RawMessage{showing}, 4); got != "" {
		t.Fatalf("a plan that does not fit was reattached: %q", got)
	}
}
