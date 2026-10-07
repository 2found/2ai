package goal

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

func TestRevisionCommitPrecedesStoreMutation(t *testing.T) {
	store := NewStore("original")
	tool := updateGoalTool{store: store}
	failure := errors.New("persistence unavailable")
	ctx := agentcore.WithGoalRevisionRecorder(context.Background(), func(context.Context, agentcore.GoalRevision) error { return failure })
	if _, err := tool.Run(ctx, `{"goal":"next","reason":"finding"}`); !errors.Is(err, failure) {
		t.Fatalf("lost commit error: %v", err)
	}
	if store.Goal() != "original" || len(store.Revisions()) != 1 {
		t.Fatal("failed commit changed the completion contract")
	}
	if _, pending := store.takePending(); pending {
		t.Fatal("failed commit was queued for prompt publication")
	}
	var recorded []agentcore.GoalRevision
	ctx = agentcore.WithGoalRevisionRecorder(context.Background(), func(_ context.Context, revision agentcore.GoalRevision) error {
		recorded = append(recorded, revision)
		return nil
	})
	for _, args := range []string{`{"goal":" next ","reason":" finding "}`, `{"goal":"next","reason":"duplicate"}`} {
		if _, err := tool.Run(ctx, args); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(recorded, []agentcore.GoalRevision{{Previous: "original", Goal: "next", Reason: "finding"}}) || len(store.Revisions()) != 2 {
		t.Fatalf("incorrect normalized commit/no-op: %+v", recorded)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := tool.Run(cancelled, `{"goal":"never","reason":"cancelled"}`); !errors.Is(err, context.Canceled) || store.Goal() != "next" {
		t.Fatalf("cancelled revision changed contract: %v", err)
	}
}

func TestNativeGoalHostRejectsMissingRecorderAndUncommittedDrain(t *testing.T) {
	ctx := context.Background()
	store := NewStore("original")
	a, err := agentcore.New(agentcore.Config{
		NativeProvider: &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: ai.ScriptedStream()}}}, Model: "test", Goal: "original",
		Policy: agentcore.NewAllowList(ToolName), Extensions: []agentcore.ExtensionFactory{Plugin{Goal: "original", Revisable: true, Store: store}},
	})
	if err != nil {
		t.Fatal(err)
	}
	host, err := a.OpenPiTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	if _, err := host.StartPiRun(ctx, "work"); err != nil {
		t.Fatal(err)
	}
	_, audit, err := host.Execute(ctx, []byte(`{"toolCallId":"call","toolName":"update_goal","args":{"goal":"next","reason":"finding"}}`), nil)
	if err != nil || !strings.Contains(audit.Trace.Error, "session recorder") || store.Goal() != "original" {
		t.Fatalf("native tool bypassed commit boundary: %+v %v goal=%s", audit, err, store.Goal())
	}
	store.Update("uncommitted", "external change")
	if _, _, err := host.RefreshPiGoal(); err == nil || !strings.Contains(err.Error(), "committed condition") || host.PiGoal() != "original" {
		t.Fatalf("uncommitted extension change reached native prompt: %v", err)
	}
}
