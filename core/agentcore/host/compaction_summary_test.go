package host

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/2found/2ai/core/ai"
	"github.com/2found/2ai/core/ai/protocol"
)

func summaryTestRequest(batches, resultBytes int) json.RawMessage {
	messages := []json.RawMessage{json.RawMessage(`{"role":"system","content":"policy"}`), json.RawMessage(`{"role":"user","content":"complete the task"}`)}
	for i := 0; i < batches; i++ {
		id := fmt.Sprintf("call-%d", i)
		messages = append(messages, nativeRequestJSON(map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "toolCall", "id": id, "name": "read", "arguments": map[string]any{}}}}))
		messages = append(messages, nativeRequestJSON(map[string]any{"role": "toolResult", "toolCallId": id, "toolName": "read", "content": []any{map[string]any{"type": "text", "text": strings.Repeat("x", resultBytes)}}}))
	}
	return nativeRequestJSON(messages)
}

func TestRenderTranscriptKeepsFactsAndDropsOpaqueBytes(t *testing.T) {
	image := strings.Repeat("A", 40_000)
	blocks := RenderTranscript([]json.RawMessage{
		json.RawMessage(`{"role":"system","content":"policy"}`),
		nativeRequestJSON(map[string]any{"role": "user", "content": "[Earlier work summary]\nGoal: ship", "agentrayContextSummary": "id", "timestamp": 1}),
		nativeRequestJSON(map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "look at this"}, map[string]any{"type": "image", "data": image, "mimeType": "image/png"}}}),
		nativeRequestJSON(map[string]any{"role": "assistant", "usage": map[string]any{"input": 99}, "content": []any{
			map[string]any{"type": "thinking", "thinking": "private", "thinkingSignature": "SIGNATURE"},
			map[string]any{"type": "text", "text": "Writing the file."},
			map[string]any{"type": "toolCall", "id": "c1", "name": "write_file", "arguments": map[string]any{"path": "a.go", "content": strings.Repeat("y", 5000)}}}}),
		nativeRequestJSON(map[string]any{"role": "toolResult", "toolCallId": "c1", "toolName": "write_file", "isError": true, "content": []any{map[string]any{"type": "text", "text": "HEAD " + strings.Repeat("z", 9000) + " permission denied"}}}),
	})
	text := strings.Join(blocks, "\n")
	for _, want := range []string{"PREVIOUS SUMMARY", "Goal: ship", "USER: look at this [1 image(s) omitted]", "ASSISTANT: Writing the file.", "ASSISTANT called write_file(content=", "path=\"a.go\"", "TOOL write_file (error) -> HEAD", "permission denied"} {
		if !strings.Contains(text, want) {
			t.Errorf("rendering lost %q", want)
		}
	}
	for _, unwanted := range []string{"SIGNATURE", "private", image[:64], "policy", `"usage"`} {
		if strings.Contains(text, unwanted) {
			t.Errorf("rendering kept %q", unwanted)
		}
	}
	if len(blocks) != 4 || len(text) > 6000 {
		t.Fatalf("rendering blocks=%d bytes=%d", len(blocks), len(text))
	}
}

func TestFoldRenderedSplitsOnBlockBoundariesAndCarriesTheSummary(t *testing.T) {
	blocks := []string{}
	for i := 0; i < 40; i++ {
		blocks = append(blocks, fmt.Sprintf("USER: fact-%02d %s", i, strings.Repeat("w", 400)))
	}
	var segments, carried []string
	summarize := func(_ context.Context, previous, segment string) (string, protocol.Usage, error) {
		segments, carried = append(segments, segment), append(carried, previous)
		return fmt.Sprintf("summary through %d", len(segments)), protocol.Usage{InputTokens: 7}, nil
	}
	text, usage, err := FoldRendered(context.Background(), blocks, 1200, summarize)
	if err != nil || len(segments) < 2 || text != fmt.Sprintf("summary through %d", len(segments)) || usage.InputTokens != 7*len(segments) {
		t.Fatalf("fold text=%q calls=%d usage=%+v err=%v", text, len(segments), usage, err)
	}
	if joined := strings.Join(segments, "\n"); joined != strings.Join(blocks, "\n") {
		t.Fatal("a fold dropped, repeated or split a block")
	}
	for i, previous := range carried {
		if (i == 0) != (previous == "") {
			t.Fatalf("call %d carried %q", i, previous)
		}
	}
	segments = nil
	if _, _, err := FoldRendered(context.Background(), blocks[:2], 4000, summarize); err != nil || len(segments) != 1 {
		t.Fatalf("a transcript that fits took %d calls err=%v", len(segments), err)
	}
}

func TestExtractAndClampSummary(t *testing.T) {
	for reply, want := range map[string]string{
		"<analysis>scratch</analysis>\n<summary>\n1. Goal: ship\n</summary>": "1. Goal: ship",
		"<analysis>scratch</analysis>\nGoal: ship":                           "Goal: ship",
		"Goal: ship":                             "Goal: ship",
		"<summary>Goal: cut off at the limit":    "Goal: cut off at the limit",
		"<analysis>the reply ended in its notes": "",
	} {
		if got := ExtractSummary(reply); got != want {
			t.Errorf("ExtractSummary(%q) = %q, want %q", reply, got, want)
		}
	}
	lines := []string{"1. Goal: keep FINDING-4=SHARD002OK"}
	for i := 0; i < 200; i++ {
		lines = append(lines, fmt.Sprintf("- done item %03d with some detail", i))
	}
	lines = append(lines, "9. Next step: run the migration")
	clamped := ClampSummary(strings.Join(lines, "\n"), 256)
	if len(clamped) > 256*4 || !strings.HasPrefix(clamped, lines[0]) || !strings.HasSuffix(clamped, lines[len(lines)-1]) || !strings.Contains(clamped, "lines omitted") {
		t.Fatalf("clamp lost an end or the marker: %d bytes", len(clamped))
	}
	for _, line := range strings.Split(clamped, "\n") {
		if strings.HasPrefix(line, "- done item") && len(line) != len(lines[1]) {
			t.Fatalf("clamp cut a line in half: %q", line)
		}
	}
}

func TestSummaryFailuresStopAtTheSoftBudgetButNotWhenForced(t *testing.T) {
	calls, fail := 0, true
	var events []CompactionEvent
	policy := CompactionPolicy{Budget: 1500, KeepRecent: 300, Observe: func(e CompactionEvent) { events = append(events, e) }, Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
		calls++
		if fail {
			return "", protocol.Usage{InputTokens: 100}, errors.New("length")
		}
		return "Goal and earlier evidence.", protocol.Usage{InputTokens: 100}, nil
	}}
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	raw := summaryTestRequest(12, 800)
	for i := 0; i < 10; i++ {
		c.testTransform(context.Background(), raw)
	}
	if calls != MaxSummaryFailures || len(events) != MaxSummaryFailures || !events[0].Failed || events[0].Usage.InputTokens != 100 {
		t.Fatalf("a failing summarizer was called %d times over 10 requests (events=%d)", calls, len(events))
	}
	forced := policy
	forced.Force = true
	c.testTransformWith(context.Background(), raw, forced)
	if calls != MaxSummaryFailures+1 {
		t.Fatal("a forced compaction did not try the summarizer")
	}
	fail = false
	view := c.testTransformWith(context.Background(), raw, forced)
	last := events[len(events)-1]
	if c.saved == nil || c.failures != 0 || len(view) >= len(raw) || last.Failed || last.Trigger != "forced" || last.Mode != "standalone" || last.PostTokens >= last.PreTokens || last.PrefixMessages != c.saved.PrefixCount {
		t.Fatalf("success did not reset the breaker or report: failures=%d event=%+v", c.failures, last)
	}
}

func TestSummarizeViewIsPreferredAndFallsBack(t *testing.T) {
	raw := summaryTestRequest(12, 800)
	for _, sharedFails := range []bool{false, true} {
		shared, standalone := 0, 0
		var event CompactionEvent
		c := NewCompactor(CompactorOptions{Revision: "r", Policy: CompactionPolicy{Budget: 1500, KeepRecent: 300, Instructions: "keep SQL", Clearing: ToolResultClearing{Disabled: true}, Observe: func(e CompactionEvent) { event = e },
			SummarizeView: func(_ context.Context, request SummaryRequest) (string, protocol.Usage, error) {
				shared++
				var view []json.RawMessage
				if json.Unmarshal(request.View, &view) != nil || request.Cut <= 0 || request.Cut >= len(view) || request.Instructions != "keep SQL" || !SameJSON(request.View, raw) {
					t.Errorf("shared request cut=%d of %d instructions=%q", request.Cut, len(view), request.Instructions)
				}
				if sharedFails {
					return "", protocol.Usage{InputTokens: 5}, errors.New("called a tool")
				}
				return "Shared handoff.", protocol.Usage{CacheReadTokens: 900}, nil
			},
			Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
				standalone++
				return "Standalone handoff.", protocol.Usage{InputTokens: 400}, nil
			}}})
		view := string(c.testTransform(context.Background(), raw))
		want, mode, fallbacks := "Shared handoff.", "shared", 0
		if sharedFails {
			want, mode, fallbacks = "Standalone handoff.", "standalone", 1
		}
		if shared != 1 || standalone != fallbacks || !strings.Contains(view, want) || event.Mode != mode || event.Failed {
			t.Fatalf("sharedFails=%v shared=%d standalone=%d mode=%q", sharedFails, shared, standalone, event.Mode)
		}
		if sharedFails && (event.Usage.InputTokens != 405 || event.Usage.CacheReadTokens != 0) {
			t.Fatalf("event usage must include the failed shared attempt: %+v", event.Usage)
		}
	}
}

func TestAllowDefersASummaryWithoutCountingAFailure(t *testing.T) {
	calls, allow := 0, false
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: CompactionPolicy{Budget: 1500, KeepRecent: 300, Clearing: ToolResultClearing{Disabled: true},
		Allow: func(_ context.Context, prefix json.RawMessage) bool {
			if ValidateMessages(prefix) != nil {
				t.Error("the hook saw a split tool batch")
			}
			return allow
		},
		Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
			calls++
			return "Goal and earlier evidence.", protocol.Usage{}, nil
		}}})
	raw := summaryTestRequest(12, 800)
	for i := 0; i < 5; i++ {
		if view := c.testTransform(context.Background(), raw); !SameJSON(view, raw) {
			t.Fatal("a deferred compaction changed the request")
		}
	}
	if calls != 0 || c.failures != 0 {
		t.Fatalf("deferred compaction calls=%d failures=%d", calls, c.failures)
	}
	allow = true
	if c.testTransform(context.Background(), raw); calls != 1 || c.saved == nil {
		t.Fatal("compaction did not resume once allowed")
	}
}

func TestInlineImagesAreNotCountedAsText(t *testing.T) {
	imageTokens := int(math.Ceil(ai.EstimatedImageTokens))
	image := map[string]any{"type": "image", "data": strings.Repeat("A", 1_000_000), "mimeType": "image/png"}
	raw := nativeRequestJSON([]any{
		map[string]any{"role": "user", "timestamp": 1, "content": []any{map[string]any{"type": "text", "text": "what is this"}, image}},
		map[string]any{"role": "assistant", "stopReason": "stop", "timestamp": 2, "usage": map[string]any{"input": 1500, "output": 20}, "content": []any{map[string]any{"type": "text", "text": "a chart"}}},
	})
	// The floor charges the image its fixed estimate, so it may sit a little
	// above what the provider reported, never near the 250k its bytes suggest.
	if got := ContextTokens(raw); got < 1520 || got > 2000 {
		t.Fatalf("provider reported 1520 tokens; estimate is %d", got)
	}
	// Before any usage exists the image still costs a fixed charge, not its bytes.
	unanswered := nativeRequestJSON([]any{map[string]any{"role": "user", "timestamp": 1, "content": []any{image}}})
	if got := ContextTokens(unanswered); got < imageTokens || got > imageTokens+64 {
		t.Fatalf("one unanswered image estimated at %d tokens", got)
	}
	// A screenshot in the recent tail must not consume the whole keep budget.
	messages := []json.RawMessage{json.RawMessage(`{"role":"user","content":"task"}`)}
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("shot-%d", i)
		messages = append(messages, nativeRequestJSON(map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "toolCall", "id": id, "name": "screenshot", "arguments": map[string]any{}}}}))
		messages = append(messages, nativeRequestJSON(map[string]any{"role": "toolResult", "toolCallId": id, "toolName": "screenshot", "content": []any{image}}))
	}
	if cut := compactionCut(new(transcriptMemo).of(messages), 4*imageTokens, 2*imageTokens); cut != len(messages)-4 {
		t.Fatalf("cut=%d of %d: the tail should keep two screenshots", cut, len(messages))
	}
}

// The host prices text with ai's estimator (characters, not bytes), and one
// inline image with ai's flat image charge, whatever its data length.
func TestHostEstimateIsTheAIEstimate(t *testing.T) {
	text := strings.Repeat("Tiếng Việt có dấu ", 100)
	if got, want := estimateTokens(text), int(ai.EstimateTextTokens(text)); got != want || got >= len(text)/3 {
		t.Fatalf("estimateTokens=%d ai=%d bytes=%d", got, want, len(text))
	}
	small := nativeRequestJSON(map[string]any{"role": "user", "content": []any{map[string]any{"type": "image", "data": strings.Repeat("A", 10_000), "mimeType": "image/png"}}})
	large := nativeRequestJSON(map[string]any{"role": "user", "content": []any{map[string]any{"type": "image", "data": strings.Repeat("A", 1_000_000), "mimeType": "image/png"}}})
	if a, b := decodeFacts(small).tokens, decodeFacts(large).tokens; a != b || a > int(math.Ceil(ai.EstimatedImageTokens))+64 {
		t.Fatalf("image priced by its data: small=%d large=%d", a, b)
	}
}
