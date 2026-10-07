package todo

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
)

func TestPhasedPatchAtomicAndCheckpointed(t *testing.T) {
	s := NewStore()
	full := NewTool(s)
	patch := &patchTool{s}
	ctx := context.Background()
	if _, err := full.Run(ctx, `{"items":[{"id":"a","phase":"investigate","content":"inspect","status":"in_progress"},{"id":"b","phase":"verify","content":"test","status":"pending"}]}`); err != nil {
		t.Fatal(err)
	}
	if _, err := patch.Run(ctx, `{"operations":[{"action":"replace","id":"a","item":{"content":"inspect","phase":"investigate","status":"completed"}},{"action":"replace","id":"b","item":{"content":"test","phase":"verify","status":"blocked"}}]}`); err != nil {
		t.Fatal(err)
	}
	before, _ := json.Marshal(s.List())
	if _, err := patch.Run(ctx, `{"operations":[{"action":"remove","id":"a"},{"action":"remove","id":"missing"}]}`); err == nil {
		t.Fatal("invalid batch accepted")
	}
	after, _ := json.Marshal(s.List())
	if string(before) != string(after) {
		t.Fatal("failed batch partially mutated")
	}
	r := &runPlan{store: s}
	raw, _ := r.NativeState()
	restored := &runPlan{store: NewStore()}
	if err := restored.RestoreNativeState(raw); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(restored.store.Render(), "[verify]") || !strings.Contains(restored.store.Render(), "[!]") {
		t.Fatal(restored.store.Render())
	}
	// Full replacements and delta batches share the one-active-step invariant.
	if _, err := patch.Run(ctx, `{"operations":[{"action":"add","id":"c","item":{"content":"one","status":"in_progress"}},{"action":"add","id":"d","item":{"content":"two","status":"in_progress"}}]}`); err == nil {
		t.Fatal("multiple active steps")
	}
	s.Set([]Item{{Content: strings.Repeat("ế", 100), Status: StatusPending}})
	if !utf8.ValidString(s.Render()) {
		t.Fatal("split UTF-8")
	}
}
func TestPatchReceiptRecoveryIgnoresRejectedDeltas(t *testing.T) {
	ctx := context.Background()
	log := agentcore.NewMemorySessionStore()
	for _, trace := range []agentcore.ToolTrace{
		{Tool: ToolName, Allowed: true, Args: `{"items":[{"id":"a","content":"inspect","status":"pending"}]}`},
		{Tool: PatchToolName, Allowed: true, Args: `{"operations":[{"action":"replace","id":"a","item":{"content":"verified","phase":"verify","status":"completed"}}]}`},
		{Tool: PatchToolName, Allowed: false, Args: `{"operations":[{"action":"remove","id":"a"}]}`},
	} {
		raw, _ := json.Marshal(map[string]any{"result": map[string]any{"details": agentcore.PiToolOutcome{Executed: true, Trace: trace}}})
		if err := log.Append(ctx, "plan", agentcore.SessionEntry{Kind: agentcore.EntryPiEffectDone, Content: string(raw)}); err != nil {
			t.Fatal(err)
		}
	}
	s := NewStore()
	if _, err := With(s).BeginRun(ctx, agentcore.RunInfo{Session: log, SessionID: "plan"}); err != nil {
		t.Fatal(err)
	}
	if len(s.List()) != 1 || s.List()[0].Status != StatusCompleted || s.List()[0].Phase != "verify" {
		t.Fatal(s.List())
	}
}
