package todo_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/todo"
)

func TestPiPlanRecoveryRequiresSuccessfulExecution(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []string{"denied", "not-executed", "error", "invalid-arguments", "nested-denied", "nested-not-executed", "nested-error", "nested-success"} {
		t.Run(kind, func(t *testing.T) {
			log := agentcore.NewMemorySessionStore()
			args := func(text string) string {
				b, _ := json.Marshal(map[string]any{"items": []any{map[string]string{"content": text, "status": "in_progress"}}})
				return string(b)
			}
			appendOutcome := func(audit agentcore.PiToolOutcome) {
				payload, _ := json.Marshal(map[string]any{"effectId": "effect", "result": map[string]any{"details": audit}})
				if err := log.Append(ctx, "plan", agentcore.SessionEntry{Kind: agentcore.EntryPiEffectDone, CallID: "plan-call", Content: string(payload)}); err != nil {
					t.Fatal(err)
				}
			}
			good := agentcore.PiToolOutcome{Executed: true, Trace: agentcore.ToolTrace{Tool: todo.ToolName, Allowed: true, Args: args("original plan")}}
			appendOutcome(good)
			next := good
			next.Trace.Args = args("replacement plan")
			switch kind {
			case "denied", "nested-denied":
				next.Trace.Allowed = false
			case "not-executed", "nested-not-executed":
				next.Executed = false
			case "error", "nested-error":
				next.Trace.Error = "failed"
			case "invalid-arguments":
				next.Trace.Args = `{"items":[{"content":"","status":"pending"}]}`
			}
			if strings.HasPrefix(kind, "nested-") {
				next = agentcore.PiToolOutcome{Executed: true, Trace: agentcore.ToolTrace{Tool: "eval", Allowed: true}, Invocations: []agentcore.ToolInvocation{{Executed: next.Executed, Trace: next.Trace}}}
			}
			appendOutcome(next)
			plan := todo.NewStore()
			if _, err := todo.With(plan).BeginRun(ctx, agentcore.RunInfo{Session: log, SessionID: "plan"}); err != nil {
				t.Fatal(err)
			}
			want := "original plan"
			if kind == "nested-success" {
				want = "replacement plan"
			}
			if len(plan.List()) != 1 || plan.List()[0].Content != want {
				t.Fatalf("recovered unexecuted plan: %+v", plan.List())
			}
		})
	}
}

func TestPiPlanIgnoresRequestedCallsAndClearsOnLeaf(t *testing.T) {
	ctx := context.Background()
	log := agentcore.NewMemorySessionStore()
	accepted := agentcore.PiToolOutcome{Executed: true, Trace: agentcore.ToolTrace{Tool: todo.ToolName, Allowed: true, Args: `{"items":[{"content":"accepted plan","status":"pending"}]}`}}
	payload, _ := json.Marshal(map[string]any{"result": map[string]any{"details": accepted}})
	for _, entry := range []agentcore.SessionEntry{
		{Kind: agentcore.EntryPiEffectDone, Content: string(payload)},
		{Kind: agentcore.EntryMessage, Message: &agentcore.Message{Role: agentcore.RoleAssistant, ToolCalls: []agentcore.ToolCall{{ID: "requested", Name: todo.ToolName, Arguments: resumePlanArgs}}}},
	} {
		if err := log.Append(ctx, "plan", entry); err != nil {
			t.Fatal(err)
		}
	}
	plan := todo.NewStore()
	if _, err := todo.With(plan).BeginRun(ctx, agentcore.RunInfo{Session: log, SessionID: "plan"}); err != nil {
		t.Fatal(err)
	}
	if got := plan.List(); len(got) != 1 || got[0].Content != "accepted plan" {
		t.Fatalf("requested call replaced accepted plan: %+v", got)
	}
	if err := log.Append(ctx, "plan", agentcore.SessionEntry{Kind: agentcore.EntryLeaf}); err != nil {
		t.Fatal(err)
	}
	fresh := todo.NewStore()
	if _, err := todo.With(fresh).BeginRun(ctx, agentcore.RunInfo{Session: log, SessionID: "plan"}); err != nil {
		t.Fatal(err)
	}
	if len(fresh.List()) != 0 {
		t.Fatal("finished session plan inherited by a new task")
	}
}
