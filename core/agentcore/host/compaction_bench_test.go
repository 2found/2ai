package host

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/2found/2ai/core/ai/protocol"
)

// benchTranscript is a long tool loop of roughly size bytes: one system
// message, the task, then assistant calls each answered by a 4 KB result.
func benchTranscript(size int) []json.RawMessage {
	messages := []json.RawMessage{json.RawMessage(`{"role":"system","content":"policy"}`), json.RawMessage(`{"role":"user","content":"complete the task","timestamp":1}`)}
	total := 0
	for i := 0; total < size; i++ {
		call, _ := json.Marshal(map[string]any{"role": "assistant", "stopReason": "toolUse", "timestamp": 2 + i, "usage": map[string]any{"input": 1000 + i, "output": 10}, "content": []any{map[string]any{"type": "thinking", "thinking": "reason", "thinkingSignature": strings.Repeat("s", 200)}, map[string]any{"type": "toolCall", "id": fmt.Sprintf("c%d", i), "name": "read", "arguments": map[string]any{"path": fmt.Sprintf("f%d.go", i), "n": 1.50}}}})
		result, _ := json.Marshal(map[string]any{"role": "toolResult", "toolCallId": fmt.Sprintf("c%d", i), "toolName": "read", "timestamp": 2 + i, "content": []any{map[string]any{"type": "text", "text": strings.Repeat("evidence ", 450)}}})
		messages = append(messages, call, result)
		total += len(call) + len(result)
	}
	return messages
}

func BenchmarkPrepareLongTranscript(b *testing.B) {
	for _, size := range []int{1 << 20, 11 << 20} {
		b.Run(fmt.Sprintf("%dMB", size>>20), func(b *testing.B) {
			messages := benchTranscript(size)
			policy := CompactionPolicy{Budget: 1 << 30, KeepRecent: 2000, Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
				return "summary", protocol.Usage{}, nil
			}}
			c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
			count := len(messages) - 10
			message, _ := json.Marshal(map[string]any{"role": "user", "content": summaryLabel + "\nearlier", "agentrayContextSummary": "id", "timestamp": 1})
			c.saved = &Summary{Revision: "r", PrefixCount: count, PrefixDigest: PrefixDigest(messages[:count]), Message: message}
			raw, _ := json.Marshal(messages)
			b.SetBytes(int64(len(raw)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := c.Prepare(context.Background(), raw, policy); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// A request after the first must not re-canonicalize the prefix an earlier
// request hashed: per-request cost would grow with the whole transcript.
func TestPrepareDoesNotRehashTheSettledPrefix(t *testing.T) {
	messages := benchTranscript(1 << 20)
	policy := CompactionPolicy{Budget: 1 << 30, KeepRecent: 2000}
	c := NewCompactor(CompactorOptions{Revision: "r", Policy: policy})
	count := len(messages) - 10
	message, _ := json.Marshal(map[string]any{"role": "user", "content": summaryLabel + "\nearlier", "agentrayContextSummary": "id", "timestamp": 1})
	c.saved = &Summary{Revision: "r", PrefixCount: count, PrefixDigest: PrefixDigest(messages[:count]), Message: message}
	for turn := 0; turn < 3; turn++ {
		messages = append(messages, nativeRequestJSON(map[string]any{"role": "user", "content": fmt.Sprintf("next %d", turn)}))
		view, err := c.Prepare(context.Background(), nativeRequestJSON(messages), policy)
		if err != nil || !strings.Contains(string(view), "earlier") {
			t.Fatal("summary did not apply", err)
		}
	}
	if c.memo.canonicalized != count {
		t.Fatalf("canonicalized %d messages over three requests, want the %d-message prefix once", c.memo.canonicalized, count)
	}
	// An edited prefix still invalidates the summary.
	messages[3] = nativeRequestJSON(map[string]any{"role": "toolResult", "toolCallId": "c0", "toolName": "read", "content": "changed"})
	if view, _ := c.Prepare(context.Background(), nativeRequestJSON(messages), policy); strings.Contains(string(view), "earlier") {
		t.Fatal("a summary applied over an edited prefix")
	}
}

// The incremental digest is the one-shot digest, including numbers and
// characters that canonical JSON rewrites.
func TestIncrementalDigestMatchesPrefixDigest(t *testing.T) {
	messages := []json.RawMessage{
		json.RawMessage(`{"role":"system","content":"<b>&amp;</b> é"}`),
		json.RawMessage(`{ "role" : "user", "content":"x", "timestamp": 1.50e3 }`),
		json.RawMessage(`{"role":"assistant","content":[],"usage":{"input":12345678901234567890}}`),
	}
	var memo transcriptMemo
	for count := 0; count <= len(messages); count++ {
		if got, want := memo.digest(messages, count), PrefixDigest(messages[:count]); got != want {
			t.Fatalf("count %d: incremental %s, one-shot %s", count, got, want)
		}
	}
	if got := memo.digest(messages, 1); got != PrefixDigest(messages[:1]) {
		t.Fatal("a shorter prefix after a longer one differs")
	}
	edited := slices.Clone(messages)
	edited[0] = json.RawMessage(`{"role":"system","content":"other"}`)
	if got := memo.digest(edited, 3); got != PrefixDigest(edited) || got == PrefixDigest(messages) {
		t.Fatal("an edited message did not invalidate the digest state")
	}
}
