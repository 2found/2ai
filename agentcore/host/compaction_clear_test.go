package host

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

// clearRequest is a tool loop of read calls answered by size-byte results.
// stamp, when nonzero, timestamps every message at that Unix millisecond.
func clearRequest(results, size int, stamp int64, extra ...json.RawMessage) []json.RawMessage {
	messages := []json.RawMessage{json.RawMessage(`{"role":"system","content":"policy"}`), json.RawMessage(`{"role":"user","content":"complete the task"}`)}
	for i := 0; i < results; i++ {
		id := fmt.Sprintf("call-%d", i)
		call := map[string]any{"role": "assistant", "stopReason": "toolUse", "content": []any{map[string]any{"type": "toolCall", "id": id, "name": "read", "arguments": map[string]any{"path": fmt.Sprintf("f%d.go", i)}}}}
		result := map[string]any{"role": "toolResult", "toolCallId": id, "toolName": "read", "content": []any{map[string]any{"type": "text", "text": fmt.Sprintf("file %d: %s", i, strings.Repeat("x", size))}}}
		if stamp != 0 {
			call["timestamp"], result["timestamp"] = stamp, stamp
		}
		messages = append(messages, nativeRequestJSON(call), nativeRequestJSON(result))
	}
	return append(messages, extra...)
}

func viewOf(t *testing.T, c *Compactor, messages []json.RawMessage, policy CompactionPolicy) []json.RawMessage {
	t.Helper()
	var view []json.RawMessage
	if err := json.Unmarshal(c.testTransformWith(context.Background(), nativeRequestJSON(messages), policy), &view); err != nil {
		t.Fatal(err)
	}
	return view
}

func clearedCount(view []json.RawMessage) int {
	n := 0
	for _, raw := range view {
		if strings.Contains(string(raw), "cleared to save context") {
			n++
		}
	}
	return n
}

func TestClearingKeepsRecentResultsAndRepeatsTheSameBytes(t *testing.T) {
	policy := CompactionPolicy{Budget: 100_000, KeepRecent: 2000}
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	messages := clearRequest(60, 4000, 0)
	view := viewOf(t, c, messages, policy)
	if got := clearedCount(view); got != 55 {
		t.Fatalf("cleared %d results, want all but the %d most recent", got, DefaultKeepToolResults)
	}
	if !strings.Contains(string(view[3]), `read({\"path\":\"f0.go\"})`) && !strings.Contains(string(view[3]), "read(path=") {
		t.Fatalf("placeholder does not name the call: %s", view[3])
	}
	if !strings.Contains(string(view[3]), "Call it again") || !strings.Contains(string(view[len(view)-1]), "file 59:") {
		t.Fatal("placeholder gives no recovery or a recent result was cleared")
	}
	if ValidateMessages(nativeRequestJSON(view)) != nil || !SameJSON(view[2], messages[2]) {
		t.Fatal("clearing broke a tool batch or rewrote an assistant message")
	}
	// Two more small turns: below ClearAtLeast the boundary stays, so every
	// cleared message is byte-identical and the cached prefix still reads.
	messages = clearRequest(62, 4000, 0)
	next := viewOf(t, c, messages, policy)
	for i := range view {
		if string(next[i]) != string(view[i]) {
			t.Fatalf("message %d changed between requests", i)
		}
	}
	if clearedCount(next) != 55 {
		t.Fatal("the boundary moved for a saving below ClearAtLeast")
	}
	// The trigger measures the cleared view: it moves again only once the
	// view is back past the trigger.
	if got := clearedCount(viewOf(t, c, clearRequest(80, 4000, 0), policy)); got != 55 {
		t.Fatalf("cleared %d under the trigger, want 55", got)
	}
	if got := clearedCount(viewOf(t, c, clearRequest(110, 4000, 0), policy)); got != 105 {
		t.Fatalf("cleared %d past the trigger, want 105", got)
	}
}

func TestClearingIsPersistedAcrossCompactors(t *testing.T) {
	policy := CompactionPolicy{Budget: 100_000, KeepRecent: 2000}
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	messages := clearRequest(60, 4000, 0)
	want := c.testTransformWith(context.Background(), nativeRequestJSON(messages), policy)
	state := c.Cleared()
	if state == nil {
		t.Fatal("no clearing boundary to persist")
	}
	// A resumed run under the trigger (fewer results visible to a smaller
	// tail) still applies the persisted boundary rather than restoring bulk.
	restored := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	if err := restored.RestoreCleared(state); err != nil {
		t.Fatal(err)
	}
	if got := restored.testTransformWith(context.Background(), nativeRequestJSON(messages), policy); string(got) != string(want) {
		t.Fatal("a restored boundary produced different bytes")
	}
	if NewCompactor(CompactorOptions{}).RestoreCleared(json.RawMessage(`{"boundary":-1}`)) == nil {
		t.Fatal("an invalid boundary was accepted")
	}
	disabled := policy
	disabled.Clearing.Disabled = true
	if got := restored.testTransformWith(context.Background(), nativeRequestJSON(messages), disabled); !SameJSON(got, nativeRequestJSON(messages)) {
		t.Fatal("disabled clearing still applied the persisted boundary")
	}
}

func TestClearingWaitsForTheTriggerWhileTheCacheIsWarm(t *testing.T) {
	now := time.UnixMilli(10_000_000)
	policy := CompactionPolicy{Budget: 100_000, KeepRecent: 2000, Clearing: ToolResultClearing{CacheTTL: time.Minute}}
	// 20 results: ~20k tokens, under the 50k trigger, a 15k saving.
	warm := clearRequest(20, 4000, now.Add(-30*time.Second).UnixMilli())
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	c.now = func() time.Time { return now }
	if got := clearedCount(viewOf(t, c, warm, policy)); got != 0 {
		t.Fatalf("cleared %d results under the trigger with a warm cache", got)
	}
	cold := clearRequest(20, 4000, now.Add(-2*time.Minute).UnixMilli())
	if got := clearedCount(viewOf(t, c, cold, policy)); got != 15 {
		t.Fatalf("cleared %d results with a cold cache, want 15", got)
	}
	// Without timestamps the cache is never assumed cold.
	c = NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	if got := clearedCount(viewOf(t, c, clearRequest(20, 4000, 0), policy)); got != 0 {
		t.Fatal("an untimestamped transcript was assumed cold")
	}
}

func TestClearingExcludesToolsShortensErrorsAndArchives(t *testing.T) {
	archived := map[string]string{}
	policy := CompactionPolicy{Budget: 100_000, KeepRecent: 2000, Clearing: ToolResultClearing{ExcludeTools: []string{"read_skill"}},
		Archive: func(_ context.Context, name, id, text string) (string, error) {
			archived[id] = text
			return "spill/" + id, nil
		}}
	messages := clearRequest(60, 4000, 0)
	skill := map[string]any{"role": "toolResult", "toolCallId": "call-1", "toolName": "read_skill", "content": []any{map[string]any{"type": "text", "text": strings.Repeat("rule ", 1000)}}}
	messages[5] = nativeRequestJSON(skill)
	failure := map[string]any{"role": "toolResult", "toolCallId": "call-2", "toolName": "read", "isError": true, "content": []any{map[string]any{"type": "text", "text": "permission denied " + strings.Repeat("e", 4000)}}}
	messages[7] = nativeRequestJSON(failure)
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	view := viewOf(t, c, messages, policy)
	if !SameJSON(view[5], messages[5]) {
		t.Fatal("an excluded tool was cleared")
	}
	if text := string(view[7]); !strings.Contains(text, "failed: permission denied") || len(text) > 800 {
		t.Fatalf("a cleared error is not short: %d bytes", len(text))
	}
	if !strings.Contains(string(view[3]), `locator \"spill/call-0\"`) || archived["call-0"] != resultText(messages[3]) {
		t.Fatalf("a large cleared result was not archived: %s", view[3])
	}
	if _, ok := archived["call-2"]; ok {
		t.Fatal("an error result was archived")
	}
}

// The shared summarizer reads the view the provider sees, cleared results
// included, so its request reuses that view's cache; the standalone one reads
// the full prefix, so a handoff does not forget a cleared result.
func TestSummarizersReadTheClearedViewAndTheFullPrefix(t *testing.T) {
	var shared, standalone json.RawMessage
	policy := CompactionPolicy{Budget: 20_000, KeepRecent: 2000, Clearing: ToolResultClearing{Keep: 25},
		SummarizeView: func(_ context.Context, request SummaryRequest) (string, protocol.Usage, error) {
			shared = request.View
			return "", protocol.Usage{}, fmt.Errorf("fall back")
		},
		Summarize: func(_ context.Context, prefix json.RawMessage, _ string) (string, protocol.Usage, error) {
			standalone = prefix
			return "Goal and evidence.", protocol.Usage{}, nil
		}}
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	view := c.testTransformWith(context.Background(), nativeRequestJSON(clearRequest(60, 4000, 0)), policy)
	if !strings.Contains(string(shared), "cleared to save context") || strings.Contains(string(standalone), "cleared to save context") || !strings.Contains(string(standalone), "file 0: xxx") {
		t.Fatal("summarizers read the wrong view")
	}
	if !strings.Contains(string(view), "Goal and evidence.") {
		t.Fatal("no summary")
	}
}

// A previewed result is the same bytes on every request, however long the
// transcript grows, and the archive is asked for it once.
func TestPreviewOfAnOversizedResultIsStable(t *testing.T) {
	archived := 0
	policy := CompactionPolicy{Budget: 4096, KeepRecent: 1024, Clearing: ToolResultClearing{Disabled: true}, Archive: func(context.Context, string, string, string) (string, error) {
		archived++
		return "spill/big", nil
	}}
	messages := []json.RawMessage{json.RawMessage(`{"role":"user","content":"task"}`)}
	for i := 0; i < 5; i++ {
		messages = append(messages, nativeRequestJSON(map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": fmt.Sprintf("step %d", i)}}}), nativeRequestJSON(map[string]any{"role": "user", "content": "go on"}))
	}
	messages = append(messages,
		nativeRequestJSON(map[string]any{"role": "assistant", "stopReason": "toolUse", "content": []any{map[string]any{"type": "toolCall", "id": "big", "name": "read", "arguments": map[string]any{}}}}),
		nativeRequestJSON(map[string]any{"role": "toolResult", "toolCallId": "big", "toolName": "read", "content": []any{map[string]any{"type": "text", "text": strings.Repeat("evidence ", 4000)}}}))
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	c.now = func() time.Time { return time.UnixMilli(1000) }
	prepare := func(messages []json.RawMessage) []json.RawMessage {
		raw, err := c.Prepare(context.Background(), nativeRequestJSON(messages), policy)
		if err != nil {
			t.Fatal(err)
		}
		var view []json.RawMessage
		_ = json.Unmarshal(raw, &view)
		return view
	}
	first := prepare(messages)
	index := len(messages) - 1
	c.now = func() time.Time { return time.UnixMilli(2000) }
	messages = append(messages, nativeRequestJSON(map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "read it"}}}), nativeRequestJSON(map[string]any{"role": "user", "content": "next"}), nativeRequestJSON(map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "ok"}}}), nativeRequestJSON(map[string]any{"role": "user", "content": "again"}))
	next := prepare(messages)
	if !strings.Contains(string(first[index]), "spill/big") || string(first[index]) != string(next[index]) {
		t.Fatalf("the preview changed between requests:\n%.200s\n%.200s", first[index], next[index])
	}
	if archived != 1 {
		t.Fatalf("archive called %d times for one result", archived)
	}
}

func TestManualCompactSummarizesWithItsFocus(t *testing.T) {
	var instructions []string
	var events []CompactionEvent
	policy := CompactionPolicy{Budget: 1 << 20, KeepRecent: 300, Observe: func(e CompactionEvent) { events = append(events, e) },
		Allow: func(context.Context, json.RawMessage) bool { return false },
		SummarizeView: func(_ context.Context, request SummaryRequest) (string, protocol.Usage, error) {
			instructions = append(instructions, request.Instructions)
			return "Focused handoff.", protocol.Usage{}, nil
		}}
	raw := summaryTestRequest(4, 400)
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	// Under the budget and with Allow refusing, nothing compacts on its own.
	if got := c.testTransform(context.Background(), raw); !SameJSON(got, raw) || len(instructions) != 0 {
		t.Fatal("an automatic compaction ran under the budget")
	}
	focused := policy
	focused.Instructions = "keep the SQL"
	c.failures = MaxSummaryFailures
	if err := c.Compact(context.Background(), raw, focused); err != nil {
		t.Fatal(err)
	}
	if len(instructions) != 1 || instructions[0] != "keep the SQL" || c.saved == nil || events[0].Trigger != "manual" {
		t.Fatalf("manual compaction: instructions=%q events=%+v", instructions, events)
	}
	if view := c.testTransform(context.Background(), raw); !strings.Contains(string(view), "Focused handoff.") {
		t.Fatal("the manual summary does not apply to the next request")
	}
	// Nothing settled to summarize is an error, not a silent no-op.
	lone := nativeRequestJSON([]any{map[string]any{"role": "user", "content": "task"}})
	if err := NewCompactor(CompactorOptions{Revision: "r", Policy: policy}).Compact(context.Background(), lone, focused); err == nil {
		t.Fatal("compacting a lone request reported success")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := NewCompactor(CompactorOptions{Revision: "r", Policy: policy}).Compact(ctx, raw, focused); err != context.Canceled {
		t.Fatalf("a cancelled compaction returned %v", err)
	}
}

func TestReattachFollowsTheSummaryWithinItsCap(t *testing.T) {
	var dropped int
	var limit int
	policy := CompactionPolicy{Budget: 8000, KeepRecent: 300, Clearing: ToolResultClearing{Disabled: true},
		Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
			return "Goal and evidence.", protocol.Usage{}, nil
		},
		Reattach: func(_ context.Context, messages []json.RawMessage, maxTokens int) string {
			dropped, limit = len(messages), maxTokens
			return "plan: step 3 of 5\n" + strings.Repeat("p", 20_000)
		}}
	raw := summaryTestRequest(12, 4000)
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	view := c.testTransform(context.Background(), raw)
	var message struct{ Content string }
	_ = json.Unmarshal(c.saved.Message, &message)
	if dropped != c.saved.PrefixCount || limit != 1000 {
		t.Fatalf("reattach saw %d messages with a %d-token cap, want %d and 1000", dropped, limit, c.saved.PrefixCount)
	}
	summary, extra, ok := strings.Cut(message.Content, reattachMarker)
	if !ok || !strings.Contains(summary, "Goal and evidence.") || !strings.HasPrefix(extra, "plan: step 3 of 5") || len(extra) > limit*4 {
		t.Fatalf("reattached context missing or over its cap: %d bytes", len(extra))
	}
	if !strings.Contains(string(view), "plan: step 3 of 5") {
		t.Fatal("reattached context is not in the request")
	}
	// The next summary renders the earlier one without its reattached context,
	// which Reattach rebuilds.
	rendered := strings.Join(RenderTranscript([]json.RawMessage{c.saved.Message}), "\n")
	if !strings.Contains(rendered, "Goal and evidence.") || strings.Contains(rendered, "plan: step 3") {
		t.Fatal("a previous summary rendered with its reattached context")
	}
}

// A second summary is given the first in full, user corrections included,
// and is asked to carry them forward.
func TestSummaryOfSummaryKeepsUserCorrections(t *testing.T) {
	var inputs []string
	policy := CompactionPolicy{Budget: 3000, KeepRecent: 300, Clearing: ToolResultClearing{Disabled: true},
		Summarize: func(_ context.Context, prefix json.RawMessage, _ string) (string, protocol.Usage, error) {
			var messages []json.RawMessage
			_ = json.Unmarshal(prefix, &messages)
			inputs = append(inputs, strings.Join(RenderTranscript(messages), "\n"))
			return fmt.Sprintf("User messages: \"use tabs, never spaces\" (correction %d).", len(inputs)), protocol.Usage{}, nil
		}}
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	var messages []json.RawMessage
	_ = json.Unmarshal(summaryTestRequest(8, 2000), &messages)
	c.testTransform(context.Background(), nativeRequestJSON(messages))
	_ = json.Unmarshal(summaryTestRequest(16, 2000), &messages)
	c.testTransform(context.Background(), nativeRequestJSON(messages))
	if len(inputs) != 2 || !strings.Contains(inputs[1], "PREVIOUS SUMMARY") || !strings.Contains(inputs[1], `"use tabs, never spaces" (correction 1)`) {
		t.Fatalf("the second summary did not read the first in full: %d inputs", len(inputs))
	}
	if !strings.Contains(SummaryInstructions(MaxSummaryTokens, "", false), "every correction and preference word for word") {
		t.Fatal("the summarizer is not asked to carry corrections forward")
	}
}
