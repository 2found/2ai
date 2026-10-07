package todo_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/todo"
	"github.com/2found/2ai/ai"
)

func TestCompletionUsesLiveRestoredAndChildPlans(t *testing.T) {
	ctx := context.Background()
	store := todo.NewStore()
	store.Set([]todo.Item{{Content: "verify", Status: todo.StatusPending}})
	p := todo.Plugin{Store: store, CheckCompletion: true, MaxCompletionNudges: 2}
	root, _ := p.BeginRun(ctx, agentcore.RunInfo{})
	guard := root.(agentcore.StopInterceptor)
	if !guard.TurnStopping(ctx, agentcore.StopInfo{}).Continue {
		t.Fatal("accepted unfinished plan")
	}
	if d := guard.TurnStopping(ctx, agentcore.StopInfo{Attempt: 2}); d.Continue || d.StopReason != "todo_incomplete" {
		t.Fatalf("ceiling accepted false success: %+v", d)
	}
	child, _ := p.BeginRun(ctx, agentcore.RunInfo{Depth: 1})
	if child.(agentcore.StopInterceptor).TurnStopping(ctx, agentcore.StopInfo{}).Continue {
		t.Fatal("child inherited parent checklist")
	}
	raw, _ := root.(agentcore.NativeStateContributor).NativeState()
	if err := child.(agentcore.NativeStateContributor).RestoreNativeState(raw); err != nil {
		t.Fatal(err)
	}
	if !child.(agentcore.StopInterceptor).TurnStopping(ctx, agentcore.StopInfo{}).Continue {
		t.Fatal("restored checklist not checked")
	}
	for _, status := range []string{todo.StatusCompleted, todo.StatusBlocked, todo.StatusAbandoned} {
		store.Set([]todo.Item{{Content: "verify", Status: status}})
		if d := guard.TurnStopping(ctx, agentcore.StopInfo{}); d.Continue || d.StopReason != "" {
			t.Fatalf("resolved status %s rejected: %+v", status, d)
		}
	}
}

func TestNativeCompletionRepairsUnfinishedPlan(t *testing.T) {
	store := todo.NewStore()
	store.Set([]todo.Item{{Content: "verify", Status: todo.StatusInProgress}})
	requests := 0
	stream := func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, opts map[string]any) (*ai.AssistantMessageEventStream, error) {
		requests++
		msg := nativePlanAnswer("done")
		if requests == 2 {
			msg = nativePlanCall("resolve", todo.ToolName, `{"items":[{"content":"verify","status":"completed"}]}`)
		}
		return ai.ScriptedStream(msg)(ctx, model, view, opts)
	}
	a, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{Model: "test", NativeProvider: nativePlanProvider(stream), Policy: agentcore.NewAllowList(todo.ToolName)}), todo.Plugin{Store: store, CheckCompletion: true})
	if err != nil {
		t.Fatal(err)
	}
	r, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "verify"}}})
	if err != nil || requests != 3 || r.StopReason != "stop" || store.List()[0].Status != todo.StatusCompleted {
		t.Fatalf("completion guard: requests=%d stop=%s err=%v", requests, r.StopReason, err)
	}
}

func TestNativeCompletionCeilingDoesNotReportSuccess(t *testing.T) {
	store := todo.NewStore()
	store.Set([]todo.Item{{Content: "verify", Status: todo.StatusPending}})
	a, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{Model: "test", NativeProvider: nativePlanProvider(ai.ScriptedStream(nativePlanAnswer("done"), nativePlanAnswer("done"))), Policy: agentcore.NewAllowList(todo.ToolName)}), todo.Plugin{Store: store, CheckCompletion: true, MaxCompletionNudges: 1})
	if err != nil {
		t.Fatal(err)
	}
	r, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "verify"}}})
	if err != nil || r.StopReason != "todo_incomplete" || r.Turns != 2 {
		t.Fatalf("accepted incomplete work: %+v %v", r, err)
	}
}
