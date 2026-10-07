package host

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func TestSummaryAllowanceReservesHostToolsAndVerbatimTask(t *testing.T) {
	prefix := piRequestJSON([]any{map[string]any{"role": "system", "content": strings.Repeat("host tools ", 1900)}, map[string]any{"role": "user", "content": "original input"}})
	task := "Write exact.json and verify the schema"
	allowance, err := SummaryAllowance(prefix, task, 6144)
	if err != nil || allowance >= 6144/4 || allowance < 128 {
		t.Fatal("summary ignored host overhead", allowance, err)
	}
	var messages []json.RawMessage
	_ = json.Unmarshal(prefix, &messages)
	summary := piRequestJSON(map[string]any{"role": "user", "content": strings.Repeat("x", allowance*4)})
	view := piRequestJSON(summaryViewWithTask(messages, len(messages), summary, task))
	if ContextTokens(view) > 6144 {
		t.Fatal("summary allowance does not fit", ContextTokens(view))
	}
	if _, err := SummaryAllowance(prefix, task, 1024); err == nil {
		t.Fatal("impossible host head allowed paid summary")
	}
}

func TestCompactionTargetIsDistinctFromModelInputCeiling(t *testing.T) {
	raw := piRequestJSON([]any{map[string]any{"role": "system", "content": strings.Repeat("immutable tools ", 1800)}, map[string]any{"role": "user", "content": "current task"}})
	policy := CompactionPolicy{Budget: 1024}.ForWindow(32000)
	view, err := NewCompactor(CompactorOptions{Revision: "test", Policy: policy}).Prepare(context.Background(), raw, policy)
	if err != nil || !SameJSON(raw, view) || ContextTokens(view) <= policy.Budget || ContextTokens(view) > policy.MaxInputTokens {
		t.Fatal("soft compaction trigger became an impossible input cap", err)
	}
	policy = policy.ForWindow(4000)
	if _, err := NewCompactor(CompactorOptions{Revision: "test", Policy: policy}).Prepare(context.Background(), raw, policy); err == nil {
		t.Fatal("model input ceiling was ignored")
	}
}

func TestSoftTargetDoesNotForceCompactARecentToolBatchEveryTurn(t *testing.T) {
	raw := piRequestJSON([]any{map[string]any{"role": "system", "content": strings.Repeat("immutable tools ", 1200)}, map[string]any{"role": "user", "content": "current task"}, map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "toolCall", "id": "one", "name": "read", "arguments": map[string]any{}}}}, map[string]any{"role": "toolResult", "toolCallId": "one", "toolName": "read", "content": []any{map[string]any{"type": "text", "text": "recent exact evidence"}}}})
	summaries := 0
	policy := CompactionPolicy{Budget: 1024, KeepRecent: 512, Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
		summaries++
		return "Summary", protocol.Usage{}, nil
	}}.ForWindow(32000)
	view, err := NewCompactor(CompactorOptions{Revision: "test", Policy: policy}).Prepare(context.Background(), raw, policy)
	if err != nil || summaries != 0 || !SameJSON(raw, view) {
		t.Fatal("recent evidence was folded solely to meet an impossible soft target", summaries, err)
	}
}

func TestPrepareRecoversOversizedFirstToolBatch(t *testing.T) {
	messages := piLongRequest()[:4]
	var result map[string]json.RawMessage
	_ = json.Unmarshal(messages[3], &result)
	full := strings.Repeat("bằng chứng ", 10000)
	result["content"] = piRequestJSON([]any{map[string]any{"type": "text", "text": full}})
	messages[3] = piRequestJSON(result)
	original := piRequestJSON(messages)
	archives := map[string]string{}
	policy := CompactionPolicy{Budget: 4096, KeepRecent: 2048, Task: "Write exact.json", Archive: func(_ context.Context, name, id, text string) (string, error) {
		archives[id] = text
		return "saved-result", nil
	}}
	c := NewCompactor(CompactorOptions{Revision: "test", Policy: policy})
	view, err := c.Prepare(context.Background(), original, policy)
	if err != nil || ContextTokens(view) > policy.Budget || ValidateMessages(view) != nil {
		t.Fatal("oversized batch not rescued", len(view), err)
	}
	if archives["reused"] != full || !strings.Contains(string(view), "read_spill") {
		t.Fatal("full evidence not recoverable")
	}
	var after []json.RawMessage
	_ = json.Unmarshal(view, &after)
	if !SameJSON(messages[2], after[2]) {
		t.Fatal("opaque assistant/signature changed")
	}
	if !SameJSON(original, piRequestJSON(messages)) {
		t.Fatal("source transcript rewritten")
	}
	policy.Archive = func(context.Context, string, string, string) (string, error) {
		return "", errors.New("disk unavailable")
	}
	if _, err = c.Prepare(context.Background(), original, policy); err == nil {
		t.Fatal("lost evidence presented as compacted")
	}
}

func TestContextTokensUsageAndRewriteAnchors(t *testing.T) {
	for _, tc := range []struct {
		name       string
		role, stop string
		rewrite    bool
		wantHigh   bool
	}{{"settled", "assistant", "stop", false, true}, {"error", "assistant", "error", false, false}, {"aborted", "assistant", "aborted", false, false}, {"rewritten", "assistant", "stop", true, false}} {
		t.Run(tc.name, func(t *testing.T) {
			m := []any{}
			if tc.rewrite {
				m = append(m, map[string]any{"role": "user", "content": "summary", "agentrayContextSummary": "id", "timestamp": 2})
			}
			m = append(m, map[string]any{"role": tc.role, "stopReason": tc.stop, "timestamp": 1, "content": []any{map[string]any{"type": "text", "text": "short"}}, "usage": map[string]any{"input": 5000, "cacheRead": 200, "output": 10}}, map[string]any{"role": "user", "content": "tail"})
			n := ContextTokens(piRequestJSON(m))
			if (n >= 5210) != tc.wantHigh {
				t.Fatal("incorrect usage anchor", n)
			}
		})
	}
}

func TestSummaryFoldsInsideInputBudget(t *testing.T) {
	prefix := piRequestJSON(strings.Repeat("evidence🙂 ", 2000))
	calls := 0
	text, usage, err := FoldSummary(context.Background(), prefix, 512, "r", func(_ context.Context, raw json.RawMessage, _ string) (string, protocol.Usage, error) {
		calls++
		if estimateTokens(string(raw)) > 512 || !json.Valid(raw) {
			t.Fatal("summary input overflow", len(raw))
		}
		if calls > 1 && !strings.Contains(string(raw), "earlier facts") {
			t.Fatal("previous fold missing")
		}
		return "earlier facts", protocol.Usage{InputTokens: 7}, nil
	})
	if err != nil || calls < 2 || text != "earlier facts" || usage.InputTokens != calls*7 {
		t.Fatal("fold failed", calls, usage, err)
	}
}
