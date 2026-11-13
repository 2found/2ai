package todo_test

import (
	"bytes"
	"context"
	"encoding/json"
	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/core/agentcore/host"
	"github.com/2found/2ai/core/ai"
	"github.com/2found/2ai/feature/plugins/todo"
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

func planExtension(t *testing.T, store *todo.Store) (agentcore.StepInterceptor, agentcore.RunObserver) {
	t.Helper()
	ext, err := todo.With(store).BeginRun(context.Background(), agentcore.RunInfo{})
	if err != nil {
		t.Fatal(err)
	}
	return ext.(agentcore.StepInterceptor), ext.(agentcore.RunObserver)
}

// The reminder is appended only when the model cannot see the current plan,
// and only once per disappearance.
func TestPlanReminderOnlyWhenPlanIsNotVisible(t *testing.T) {
	ctx := context.Background()
	store := todo.NewStore()
	step, observer := planExtension(t, store)
	request := func(msgs ...agentcore.Message) { observer.ObserveMessages(ctx, agentcore.PhaseRequest, 1, msgs) }
	reminders := func() []agentcore.Message { return step.BeforeStep(ctx, agentcore.StepInfo{}).AdditionalContexts }

	request(agentcore.Message{Role: agentcore.RoleUser, Content: "go"})
	if got := reminders(); len(got) != 0 {
		t.Fatalf("empty plan injected a reminder: %+v", got)
	}

	store.Set([]todo.Item{{Content: "ship it", Status: todo.StatusInProgress}})
	got := reminders()
	if len(got) != 1 || got[0].Role != agentcore.RoleUser || !strings.HasPrefix(got[0].Content, todo.ContextPrefix+"\n") || !strings.Contains(got[0].Content, "[~] ship it") {
		t.Fatalf("invisible plan was not appended as a user message: %+v", got)
	}
	if again := reminders(); len(again) != 0 {
		t.Fatalf("reminder repeated before the model saw a new view: %+v", again)
	}

	// The update_plan result carries the rendering: no reminder needed.
	request(agentcore.Message{Role: agentcore.RoleTool, Content: "Plan updated.\n" + store.Render()})
	if got := reminders(); len(got) != 0 {
		t.Fatalf("visible plan injected a reminder: %+v", got)
	}

	// A stale rendering in view does not count as the current plan.
	store.Set([]todo.Item{{Content: "ship it", Status: todo.StatusCompleted}, {Content: "announce", Status: todo.StatusInProgress}})
	if got := reminders(); len(got) != 1 || !strings.Contains(got[0].Content, "[~] announce") {
		t.Fatalf("stale plan in view suppressed the reminder: %+v", got)
	}

	// An appended message showing the plan counts; a rebase that drops it resets.
	observer.ObserveMessages(ctx, agentcore.PhaseAppend, 2, []agentcore.Message{{Role: agentcore.RoleTool, Content: store.Render()}})
	if got := reminders(); len(got) != 0 {
		t.Fatalf("appended plan injected a reminder: %+v", got)
	}
	observer.ObserveMessages(ctx, agentcore.PhaseRebase, 2, []agentcore.Message{{Role: agentcore.RoleUser, Content: "[Earlier work summary]"}})
	if got := reminders(); len(got) != 1 {
		t.Fatalf("compaction dropped the plan without a reminder: %+v", got)
	}
}

// A plan update must not rewrite anything the provider has already cached. On a
// provider without mid-conversation system messages every earlier request must
// be a strict byte prefix of the next one.
func TestPlanUpdatesKeepRequestsPrefixStable(t *testing.T) {
	store := todo.NewStore()
	var requests [][]byte
	script := ai.ScriptedStream(
		nativePlanCall("p1", todo.ToolName, `{"items":[{"content":"read","status":"in_progress"},{"content":"write","status":"pending"}]}`),
		nativePlanCall("p2", todo.ToolName, `{"items":[{"content":"read","status":"completed"},{"content":"write","status":"in_progress"}]}`),
		nativePlanAnswer("done"),
	)
	a := newPlanAgent(t, store, func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		raw, err := json.Marshal(ai.ResolveTranscript(view, false).Messages())
		if err != nil {
			t.Fatal(err)
		}
		requests = append(requests, raw)
		return script(ctx, model, view, options)
	})
	if _, err := a.Prompt(context.Background(), "do the work"); err != nil {
		t.Fatal(err)
	}
	if len(requests) != 3 {
		t.Fatalf("requests = %d, want 3", len(requests))
	}
	for i := 1; i < len(requests); i++ {
		earlier := bytes.TrimSuffix(requests[i-1], []byte("]"))
		if len(requests[i]) <= len(requests[i-1]) || !bytes.HasPrefix(requests[i], earlier) {
			t.Fatalf("request %d rewrote request %d's prefix:\nbefore: %s\nafter:  %s", i+1, i, requests[i-1], requests[i])
		}
	}
	if strings.Contains(string(requests[2]), todo.ContextPrefix) {
		t.Fatal("a plan the model just wrote was re-sent as a reminder")
	}
}

// A plan the model cannot see (here: set before the run, then compaction) is
// appended once as a persisted user message, never as a system message.
func TestTodoSurvivesCompaction(t *testing.T) {
	store := todo.NewStore()
	store.Set([]todo.Item{{Content: "phase 1", Status: todo.StatusCompleted}, {Content: "phase 2", Status: todo.StatusInProgress}})
	requests := 0
	a := newPlanAgent(t, store, func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		requests++
		raw, _ := json.Marshal(view)
		if prompt := ai.GetCurrentSystemPrompt(view.Messages()); strings.Contains(prompt, todo.ContextPrefix) {
			t.Errorf("plan rewrote the system prompt: %s", prompt)
		}
		if !strings.Contains(string(raw), "[~] phase 2") || strings.Count(string(raw), todo.ContextPrefix) != 1 {
			t.Errorf("live plan lost or duplicated after compaction: %s", raw)
		}
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
	if len(checkpoint.Summary) == 0 || strings.Count(string(checkpoint.Messages), todo.ContextPrefix) != 1 {
		t.Fatalf("summary missing or plan reminder not persisted exactly once: %s", checkpoint.Messages)
	}
}
