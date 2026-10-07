package agentcore

import (
	"reflect"
	"strings"
	"testing"
)

func pruneFixture() []Message {
	big := strings.Repeat("x", 4000)
	return []Message{
		{Role: RoleSystem, Content: "system prompt"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "read_file", Arguments: `{"path":"a.txt"}`}}},
		{Role: RoleTool, ToolCallID: "c1", Name: "read_file", Content: "a-v1 " + big},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c2", Name: "read_file", Arguments: `{"path":"b.txt"}`}}},
		{Role: RoleTool, ToolCallID: "c2", Name: "read_file", Content: "b-v1 " + big},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c3", Name: "read_file", Arguments: `{"path":"c.txt"}`}}},
		{Role: RoleTool, ToolCallID: "c3", Name: "read_file", Content: "c-v1 " + big},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c4", Name: "run_shell", Arguments: `{"command":"true"}`}}},
		{Role: RoleTool, ToolCallID: "c4", Name: "run_shell", Content: "exit_code: 0"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c5", Name: "edit_file", Arguments: `{"path":"b.txt","old_string":"x","new_string":"y"}`}}},
		{Role: RoleTool, ToolCallID: "c5", Name: "edit_file", Content: "path: b.txt\nreplacements: 1"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c6", Name: "read_file", Arguments: `{"path":"a.txt"}`}}},
		{Role: RoleTool, ToolCallID: "c6", Name: "read_file", Content: "a-v2 " + big},
		{Role: RoleUser, Content: strings.Repeat("recent tail ", 400)},
		{Role: RoleAssistant, Content: "working on it", Usage: &Usage{InputTokens: 6000}},
	}
}

func TestPruneContextBelowThresholdIsOriginalSlice(t *testing.T) {
	msgs := pruneFixture()
	out, changed := pruneContextRequest(ContextPruneRequest{Messages: msgs, Budget: 10_000_000, Settings: DefaultCompactionSettings()})
	if changed {
		t.Fatal("must not prune below the soft threshold")
	}
	if &out[0] != &msgs[0] {
		t.Fatal("no-op must return the original slice")
	}
}

func TestPruneContextClearsSupersededStaleAndOldResults(t *testing.T) {
	msgs := pruneFixture()
	out, changed := pruneContextRequest(ContextPruneRequest{Messages: msgs, Budget: 1000, Settings: CompactionSettings{KeepRecentTokens: 500}})
	if !changed {
		t.Fatal("expected pruning")
	}

	byID := make(map[string]Message)
	for _, m := range out {
		if m.Role == RoleTool {
			byID[m.ToolCallID] = m
		}
	}
	if !strings.Contains(byID["c1"].Content, "superseded by a newer identical read_file") {
		t.Fatalf("c1 = %q, want superseded placeholder", byID["c1"].Content)
	}
	if !strings.Contains(byID["c2"].Content, "b.txt was modified later") {
		t.Fatalf("c2 = %q, want stale-read placeholder", byID["c2"].Content)
	}
	if !strings.Contains(byID["c3"].Content, "result cleared after it was consumed") {
		t.Fatalf("c3 = %q, want old-result placeholder", byID["c3"].Content)
	}
	if byID["c4"].Content != "exit_code: 0" {
		t.Fatalf("small result was pruned: %q", byID["c4"].Content)
	}
	for id, m := range byID {
		if m.ToolCallID == "" || m.Name == "" {
			t.Fatalf("tool linkage lost for %s: %+v", id, m)
		}
	}
	// The provider observation is retained so system/tool-schema overhead is
	// not lost, but removed message bytes are carried as a separate adjustment.
	last := out[len(out)-1]
	if last.Usage == nil || last.ContextTokenAdjustment >= 0 {
		t.Fatalf("usage should be retained with a negative context adjustment: %+v", last)
	}
}

func TestPruneContextClearsSupersededRichResultWithoutMutatingSource(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "old", Name: "eval", Arguments: `{"code":"plot()"}`}}},
		{Role: RoleTool, ToolCallID: "old", Name: "eval", Content: "plot ready", ContentParts: []ContentPart{{Type: ContentPartImage, MIMEType: "image/png", Data: "aW1hZ2U="}}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "new", Name: "eval", Arguments: `{"code":"plot()"}`}}},
		{Role: RoleTool, ToolCallID: "new", Name: "eval", Content: "new plot"},
		{Role: RoleUser, Content: strings.Repeat("recent ", 500)},
	}
	out, changed := pruneContextRequest(ContextPruneRequest{Messages: msgs, Budget: 1000, Settings: CompactionSettings{KeepRecentTokens: 100}})
	if !changed {
		t.Fatal("expected superseded rich result to be pruned")
	}
	if len(out[2].ContentParts) != 0 || !strings.Contains(out[2].Content, "superseded") {
		t.Fatalf("rich result was not safely replaced: %+v", out[2])
	}
	if len(msgs[2].ContentParts) != 1 {
		t.Fatal("pruning mutated the source transcript")
	}
}

func TestPruneContextRetainsUnseenProviderOverhead(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "small visible system"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "r", Name: "read_file", Arguments: `{"path":"a"}`}}},
		{Role: RoleTool, ToolCallID: "r", Name: "read_file", Content: strings.Repeat("x", 9000)},
		{Role: RoleUser, Content: strings.Repeat("recent ", 350)},
		// This includes several thousand tokens of provider tool schemas that
		// estimateBytesTokens(messages) cannot observe.
		{Role: RoleAssistant, Content: "continue", Usage: &Usage{InputTokens: 6000, CostUSD: 1.25}},
	}
	out, changed := pruneContextRequest(ContextPruneRequest{Messages: msgs, Budget: 3000, Settings: CompactionSettings{KeepRecentTokens: 500}})
	if !changed {
		t.Fatal("expected pruning")
	}
	last := out[len(out)-1]
	if last.Usage == nil || last.Usage.InputTokens != 6000 || last.Usage.CostUSD != 1.25 {
		t.Fatalf("billable usage was changed: %+v", last.Usage)
	}
	if got := estimateContextTokens(out); got <= 3000 {
		t.Fatalf("provider-only schema overhead was lost; estimate=%d", got)
	}
}

func TestPruneContextPreservesRecoverableResultReference(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "q", Name: "run_query", Arguments: `{}`}}},
		{Role: RoleTool, ToolCallID: "q", Name: "run_query", Content: strings.Repeat("x", 5000), ResultRef: "spill_abc123"},
		{Role: RoleUser, Content: strings.Repeat("recent ", 500)},
	}
	out, changed := pruneContextRequest(ContextPruneRequest{Messages: msgs, Budget: 1000, Settings: CompactionSettings{KeepRecentTokens: 500}})
	if !changed {
		t.Fatal("expected old preview to be pruned")
	}
	got := out[2]
	if got.ResultRef != "spill_abc123" || !strings.Contains(got.Content, "spill_abc123") {
		t.Fatalf("recoverable reference was lost: %+v", got)
	}
}

func TestPruneContextProtectsDeepWarmCachePrefix(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "old", Name: "read_file", Arguments: `{"path":"a"}`}}},
		{Role: RoleTool, ToolCallID: "old", Name: "read_file", Content: strings.Repeat("o", 4000)},
		{Role: RoleUser, Content: strings.Repeat("gap ", 1000)},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "new", Name: "read_file", Arguments: `{"path":"a"}`}}},
		{Role: RoleTool, ToolCallID: "new", Name: "read_file", Content: strings.Repeat("n", 400)},
		{Role: RoleUser, Content: strings.Repeat("recent ", 100)},
	}
	settings := CompactionSettings{
		KeepRecentTokens:           100,
		PruneCacheWarmSuffixTokens: 200,
		PruneMinimumSavingsTokens:  1,
	}
	uncached, changed := pruneContextRequest(ContextPruneRequest{
		Messages: msgs, Budget: 1000, Settings: settings,
	})
	if !changed || strings.Contains(uncached[2].Content, strings.Repeat("o", 200)) {
		t.Fatal("uncached superseded result should be pruned")
	}
	cached, changed := pruneContextRequest(ContextPruneRequest{
		Messages: msgs, Budget: 1000, Settings: settings, PromptCacheActive: true,
	})
	if changed || !reflect.DeepEqual(cached, msgs) {
		t.Fatal("deep warm-prefix result must wait for full compaction")
	}
}

func TestPruneContextStillPrunesCheapCacheTail(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "old", Name: "read_file", Arguments: `{"path":"a"}`}}},
		{Role: RoleTool, ToolCallID: "old", Name: "read_file", Content: strings.Repeat("o", 4000)},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "new", Name: "read_file", Arguments: `{"path":"a"}`}}},
		{Role: RoleTool, ToolCallID: "new", Name: "read_file", Content: strings.Repeat("n", 400)},
		{Role: RoleUser, Content: strings.Repeat("recent ", 50)},
	}
	out, changed := pruneContextRequest(ContextPruneRequest{
		Messages: msgs, Budget: 1000,
		Settings: CompactionSettings{
			KeepRecentTokens:           100,
			PruneCacheWarmSuffixTokens: 500,
			PruneMinimumSavingsTokens:  1,
		},
		PromptCacheActive: true,
	})
	if !changed || !strings.Contains(out[2].Content, "superseded") {
		t.Fatalf("cheap cache-tail result was not pruned: %+v", out[2])
	}
}

func TestPruneContextBatchesGenericResultsBehindSavingsFloor(t *testing.T) {
	build := func(two bool) []Message {
		msgs := []Message{
			{Role: RoleSystem, Content: "sys"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "a", Name: "query_a", Arguments: `{}`}}},
			{Role: RoleTool, ToolCallID: "a", Name: "query_a", Content: strings.Repeat("a", 1200)},
		}
		if two {
			msgs = append(msgs,
				Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "b", Name: "query_b", Arguments: `{}`}}},
				Message{Role: RoleTool, ToolCallID: "b", Name: "query_b", Content: strings.Repeat("b", 1200)},
			)
		}
		return append(msgs, Message{Role: RoleUser, Content: strings.Repeat("recent ", 1000)})
	}
	settings := CompactionSettings{KeepRecentTokens: 500}
	one := build(false)
	if out, changed := pruneContextRequest(ContextPruneRequest{Messages: one, Budget: 4000, Settings: settings}); changed || !reflect.DeepEqual(out, one) {
		t.Fatal("a sub-floor generic saving should not churn the prefix")
	}
	two := build(true)
	if _, changed := pruneContextRequest(ContextPruneRequest{Messages: two, Budget: 4000, Settings: settings}); !changed {
		t.Fatal("batched generic savings above the adaptive floor should prune")
	}
}

func TestPruneContextProtectsErrorsSkillsAndConfiguredTools(t *testing.T) {
	big := strings.Repeat("e", 4000)
	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "err", Name: "run_shell", Arguments: `{}`},
			{ID: "skill", Name: readSkillToolName, Arguments: `{"id":"safe"}`},
			{ID: "artifact", Name: "capture_artifact", Arguments: `{}`},
		}},
		{Role: RoleTool, ToolCallID: "err", Name: "run_shell", Content: "error: " + big},
		{Role: RoleTool, ToolCallID: "skill", Name: readSkillToolName, Content: "instructions " + big},
		{Role: RoleTool, ToolCallID: "artifact", Name: "capture_artifact", Content: "one-off " + big},
		{Role: RoleUser, Content: strings.Repeat("recent ", 500)},
	}
	settings := CompactionSettings{
		KeepRecentTokens:    500,
		PruneProtectedTools: []string{readSkillToolName, "capture_artifact"},
	}
	out, changed := pruneContextRequest(ContextPruneRequest{Messages: msgs, Budget: 1000, Settings: settings})
	if changed || !reflect.DeepEqual(out, msgs) {
		t.Fatal("protected and error results must remain verbatim")
	}
}

func TestPruneContextIsCopyOnWriteAndIdempotent(t *testing.T) {
	msgs := pruneFixture()
	want := append([]Message(nil), msgs...)
	out, changed := pruneContextRequest(ContextPruneRequest{Messages: msgs, Budget: 1000, Settings: CompactionSettings{KeepRecentTokens: 500}})
	if !changed {
		t.Fatal("first pass must prune")
	}
	if !reflect.DeepEqual(msgs, want) {
		t.Fatal("pruning mutated its input")
	}
	again, changedAgain := pruneContextRequest(ContextPruneRequest{Messages: out, Budget: 1000, Settings: CompactionSettings{KeepRecentTokens: 500}})
	if changedAgain || !reflect.DeepEqual(again, out) {
		t.Fatal("second pass must be an exact no-op")
	}
}

func TestPruneContextKeepsRecentToolResults(t *testing.T) {
	big := strings.Repeat("y", 8000)
	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: strings.Repeat("old padding ", 500)},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "r1", Name: "read_file", Arguments: `{"path":"z.txt"}`}}},
		{Role: RoleTool, ToolCallID: "r1", Name: "read_file", Content: big},
	}
	out, _ := pruneContextRequest(ContextPruneRequest{Messages: msgs, Budget: 1000, Settings: CompactionSettings{KeepRecentTokens: 4000}})
	if got := out[len(out)-1].Content; got != big {
		t.Fatalf("recent tool result changed: %.60q", got)
	}
}
