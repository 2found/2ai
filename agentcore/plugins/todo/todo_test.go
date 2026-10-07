package todo_test

import (
	"context"
	"encoding/json"
	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/host"
	"github.com/2found/2ai/agentcore/plugins/todo"
	"github.com/2found/2ai/ai"
	"strings"
	"testing"
)

func TestTodoToolSetsAndRenders(t *testing.T) {
	store := todo.NewStore()
	tool := todo.NewTool(store)

	out, err := tool.Run(context.Background(), `{"items":[
		{"content":"Read the schema","status":"completed"},
		{"content":"Write the migration","status":"in_progress"},
		{"content":"Run the tests","status":"pending"}]}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	for _, want := range []string{"[x] Read the schema", "[~] Write the migration", "[ ] Run the tests"} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered plan missing %q in:\n%s", want, out)
		}
	}
	if got := store.List(); len(got) != 3 {
		t.Fatalf("store should hold 3 items, got %d", len(got))
	}
}

func TestTodoToolRejectsMultipleInProgress(t *testing.T) {
	tool := todo.NewTool(todo.NewStore())
	_, err := tool.Run(context.Background(), `{"items":[
		{"content":"a","status":"in_progress"},
		{"content":"b","status":"in_progress"}]}`)
	if err == nil || !strings.Contains(err.Error(), "in_progress") {
		t.Fatalf("expected rejection of two in_progress items, got %v", err)
	}
}

func TestTodoToolRejectsBadStatusAndEmpty(t *testing.T) {
	tool := todo.NewTool(todo.NewStore())
	if _, err := tool.Run(context.Background(), `{"items":[{"content":"a","status":"doing"}]}`); err == nil {
		t.Fatal("expected rejection of invalid status")
	}
	if _, err := tool.Run(context.Background(), `{"items":[{"content":"   ","status":"pending"}]}`); err == nil {
		t.Fatal("expected rejection of empty content")
	}
}

func TestContextHookInjectsLivePlan(t *testing.T) {
	store := todo.NewStore()
	hook := todo.PiContextHook(store)
	base := []json.RawMessage{json.RawMessage(`{"role":"system","content":"persona"}`), json.RawMessage(`{"role":"user","content":"go"}`)}
	before, _ := json.Marshal(base)
	if got, err := hook(context.Background(), base); err != nil || len(got) != len(base) {
		t.Fatalf("empty plan changed native view: %s %v", got, err)
	}
	store.Set([]todo.Item{{Content: "ship it", Status: todo.StatusInProgress}})
	out, err := hook(context.Background(), base)
	if err != nil || len(out) != len(base)+1 {
		t.Fatalf("missing native reminder: %s %v", out, err)
	}
	var last struct{ Role, Content string }
	if err := json.Unmarshal(out[len(out)-1], &last); err != nil {
		t.Fatal(err)
	}
	if last.Role != "system" || !strings.HasPrefix(last.Content, todo.ContextPrefix) || !strings.Contains(last.Content, "ship it") {
		t.Fatalf("wrong native reminder: %+v", last)
	}
	after, _ := json.Marshal(base)
	if string(after) != string(before) {
		t.Fatal("native hook mutated caller history")
	}
}

// Exercise the shipping request transform inside the engine. The live plan
// survives real compaction without accumulating reminders in the checkpoint.
func TestTodoSurvivesCompaction(t *testing.T) {
	store := todo.NewStore()
	store.Set([]todo.Item{{Content: "phase 1", Status: todo.StatusCompleted}, {Content: "phase 2", Status: todo.StatusInProgress}})
	requests := 0
	a := newPlanAgent(t, store, func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		requests++
		prompt := ai.GetCurrentSystemPrompt(view.Messages())
		if !strings.Contains(prompt, "[~] phase 2") || strings.Count(prompt, todo.ContextPrefix) != 1 {
			t.Errorf("live plan lost or duplicated after compaction: %s", prompt)
		}
		raw, _ := json.Marshal(view)
		if host.ContextTokens(raw) > (host.CompactionPolicy{Budget: 1200}).ForWindow(128000).MaxInputTokens {
			t.Error("compacted request still exceeds the model input ceiling")
		}
		if !strings.Contains(string(raw), "[Earlier work summary]") {
			t.Error("native request was not compacted")
		}
		return ai.ScriptedStream(nativePlanAnswer("continuing phase 2"))(ctx, model, view, options)
	})
	input := []agentcore.Message{}
	for i := 0; i < 10; i++ {
		input = append(input, agentcore.Message{Role: agentcore.RoleUser, Content: strings.Repeat("old source evidence ", 100)})
	}
	input = append(input, agentcore.Message{Role: agentcore.RoleUser, Content: "continue"})
	summaries := 0
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: input, Compaction: &host.CompactionPolicy{Budget: 1200, KeepRecent: 100, Summarize: func(context.Context, json.RawMessage, string) (string, agentcore.Usage, error) {
		summaries++
		return "Earlier evidence.", agentcore.Usage{}, nil
	}}})
	if err != nil || requests != 1 || summaries != 1 {
		t.Fatalf("native compaction: requests=%d summaries=%d err=%v", requests, summaries, err)
	}
	var checkpoint struct{ Messages, Summary json.RawMessage }
	if err := json.Unmarshal(result.NativeState, &checkpoint); err != nil {
		t.Fatal(err)
	}
	if len(checkpoint.Summary) == 0 || strings.Contains(string(checkpoint.Messages), todo.ContextPrefix) {
		t.Fatal("summary missing or request-only plan leaked into persisted history")
	}
}
