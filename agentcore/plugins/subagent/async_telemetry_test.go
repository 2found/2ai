package subagent_test

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/jobs"
	"github.com/2found/2ai/agentcore/plugins/subagent"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry"
)

// An async child outlives the spawn call that started it. Its spans record
// under the parent's run, which joins the job, never under the spawn call's
// span: a child opening its run after that span ended would be a late no-op
// and vanish from the run's telemetry.
func TestAsyncChildRecordsUnderTheRun(t *testing.T) {
	var parentCalls atomic.Int32
	spawn := ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "spawn", Name: subagent.ToolSpawnSubagent, Arguments: json.RawMessage(`{"task":"async child work","async":true}`)})}
	provider := nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, opts map[string]any) (*ai.AssistantMessageEventStream, error) {
		for _, m := range view.Messages() {
			if m.Role == "user" && m.Content.Text != nil && strings.Contains(*m.Content.Text, "async child work") {
				return ai.ScriptedStream(nativeAnswer("child finished"))(ctx, model, view, opts)
			}
		}
		if parentCalls.Add(1) == 1 {
			return ai.ScriptedStream(spawn)(ctx, model, view, opts)
		}
		return ai.ScriptedStream(nativeAnswer("parent finished"))(ctx, model, view, opts)
	})
	a, err := agentcore.New(agentcore.Config{Model: "test", NativeProvider: provider, Policy: agentcore.NewAllowList(subagent.ToolSpawnSubagent, "job_wait", "job_status", "job_list", "job_cancel"),
		Extensions: []agentcore.ExtensionFactory{&subagent.Plugin{AllowAsync: true}, jobs.Plugin{}}})
	if err != nil {
		t.Fatal(err)
	}
	private := telemetry.NewInMemory()
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Telemetry: private.Context, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "parent work"}}})
	if err != nil || result.StopReason != "stop" {
		t.Fatalf("run: %+v %v", result, err)
	}
	var runs []telemetry.RecordedSpan
	for _, s := range private.GetSpans() {
		if s.Name == "agentray.agent.run" {
			runs = append(runs, s)
		}
	}
	if len(runs) != 2 || runs[0].ParentID != nil || runs[1].ParentID == nil || *runs[1].ParentID != runs[0].ID {
		t.Fatalf("child run must be a child of the parent run: %+v", runs)
	}
}
