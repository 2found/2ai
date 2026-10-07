package host

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func piLongRequest() []json.RawMessage {
	result := []json.RawMessage{json.RawMessage(`{"role":"system","content":"policy","toolsAdded":[{"name":"read","parameters":{"type":"object"}}]}`), json.RawMessage(`{"role":"user","content":"complete the task"}`)}
	for i := 0; i < 4; i++ {
		result = append(result, json.RawMessage(`{"role":"assistant","content":[{"type":"thinking","thinking":"reason","thinkingSignature":"opaque"},{"type":"toolCall","id":"reused","name":"read","arguments":{}}]}`))
		raw, _ := json.Marshal(map[string]any{"role": "toolResult", "toolCallId": "reused", "toolName": "read", "content": []any{map[string]any{"type": "text", "text": strings.Repeat("source ", 500)}}, "details": map[string]any{"opaque": i}})
		result = append(result, raw)
	}
	return result
}

func piRequestJSON(value any) json.RawMessage { raw, _ := json.Marshal(value); return raw }

func TestPiRequestCompactionPreservesTranscriptAndReusesExactPrefix(t *testing.T) {
	messages := piLongRequest()
	raw := piRequestJSON(messages)
	before := string(raw)
	calls, writes, total := 0, 0, 0
	c := Compactor{revision: "r", config: CompactionPolicy{Budget: 1500, KeepRecent: 500, Summarize: func(_ context.Context, raw json.RawMessage, revision string) (string, protocol.Usage, error) {
		calls++
		if ValidateMessages(raw) != nil || revision != "r" {
			t.Fatal("summary split tool batch")
		}
		return "Task and earlier evidence.", protocol.Usage{InputTokens: 17}, nil
	}}, record: func(context.Context, json.RawMessage) error { writes++; return nil }, usage: func(u protocol.Usage) { total += u.InputTokens }}
	view := c.Transform(context.Background(), raw)
	if c.saved == nil || calls != 1 || writes != 1 || total != 17 || len(view) >= len(raw) {
		t.Fatalf("compaction did not persist/save bytes: calls=%d writes=%d usage=%d", calls, writes, total)
	}
	if string(raw) != before {
		t.Fatal("request transform rewrote original transcript")
	}
	var got []json.RawMessage
	_ = json.Unmarshal(view, &got)
	if !SameJSON(got[0], messages[0]) {
		t.Fatal("native system/tool state changed")
	}
	for i, tail := range messages[c.saved.PrefixCount:] {
		if !SameJSON(tail, got[i+2]) {
			t.Fatal("retained block/signature/result changed")
		}
	}
	// A fresh compactor loads only its durable summary. No paid replay or usage
	// charge occurs when the same full native transcript is presented again.
	persisted, err := ParseSummary(string(piRequestJSON(c.saved)))
	if err != nil {
		t.Fatal(err)
	}
	restored := Compactor{revision: "r", saved: &persisted, config: c.config, record: c.record, usage: c.usage}
	again := restored.Transform(context.Background(), raw)
	if !SameJSON(view, again) || calls != 1 || total != 17 {
		t.Fatal("restored summary reran or charged historical work")
	}
	// An opaque edit invalidates the summary even when rendered text is equal.
	messages[2] = json.RawMessage(strings.Replace(string(messages[2]), "opaque", "different-signature", 1))
	changed := restored.Transform(context.Background(), piRequestJSON(messages))
	if calls != 2 || SameJSON(view, changed) {
		t.Fatal("summary survived a changed native prefix")
	}
}

func TestPiRequestCompactionIteratesWithoutLosingEarlierSummary(t *testing.T) {
	messages := piLongRequest()
	calls := 0
	c := Compactor{revision: "r", config: CompactionPolicy{Budget: 1500, KeepRecent: 500, Summarize: func(_ context.Context, raw json.RawMessage, _ string) (string, protocol.Usage, error) {
		calls++
		if calls > 1 && !strings.Contains(string(raw), "prior facts") {
			t.Fatal("iterative summary lost earlier facts")
		}
		return "prior facts and more evidence", protocol.Usage{}, nil
	}}}
	c.Transform(context.Background(), piRequestJSON(messages))
	first := c.saved.PrefixCount
	messages = append(messages, piLongRequest()[2:]...)
	view := c.Transform(context.Background(), piRequestJSON(messages))
	if calls != 2 || c.saved.PrefixCount <= first || c.saved.PrefixDigest != PrefixDigest(messages[:c.saved.PrefixCount]) || ValidateMessages(view) != nil {
		t.Fatal("iterative compaction lost original prefix indexing")
	}
}

func TestPiRequestCompactionFailuresUseSafeNativeView(t *testing.T) {
	raw := piRequestJSON(piLongRequest())
	for _, kind := range []string{"error", "panic", "empty", "tool markup", "too long", "storage", "cancel"} {
		t.Run(kind, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			charged := 0
			c := Compactor{revision: "r", config: CompactionPolicy{Budget: 1500, KeepRecent: 500, Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
				u := protocol.Usage{InputTokens: 9}
				switch kind {
				case "error":
					return "", u, errors.New("summary failed")
				case "panic":
					panic("broken observer")
				case "empty":
					return " ", u, nil
				case "tool markup":
					return `<|open|>tools<|sep|>call tool="session_query"<|close|>tools`, u, nil
				case "too long":
					return strings.Repeat("larger ", 5000), u, nil
				case "cancel":
					cancel()
				}
				return "summary", u, nil
			}}, record: func(context.Context, json.RawMessage) error {
				if kind == "storage" {
					return errors.New("write failed")
				}
				t.Fatal("failed summary published")
				return nil
			}, usage: func(u protocol.Usage) { charged += u.InputTokens }}
			if got := c.Transform(ctx, raw); !SameJSON(got, raw) || c.saved != nil {
				t.Fatal("failed summary changed view")
			}
			if kind != "panic" && charged != 9 {
				t.Fatal("failed model work was counted as free")
			}
		})
	}
}

func TestCompactionKeepsCurrentRequestDespiteLossySummary(t *testing.T) {
	raw := piRequestJSON(piLongRequest())
	task := "Write synthesis.json with release_code, budget, sites and total_cost; do not edit policy.json."
	c := Compactor{revision: "r", config: CompactionPolicy{Budget: 1500, KeepRecent: 500, Task: task, Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
		return "facts only", protocol.Usage{}, nil
	}}}
	view := c.Transform(context.Background(), raw)
	if c.saved == nil || !strings.Contains(string(view), task) {
		t.Fatal("exact current request missing", string(view))
	}
	if !SameJSON(view, c.Transform(context.Background(), raw)) {
		t.Fatal("restored view dropped or duplicated task")
	}
	bad := *c.saved
	bad.Message = piRequestJSON(map[string]any{"role": "user", "content": "[Earlier work summary]\n<|open|>tools<|sep|>session_query", "agentrayContextSummary": "id", "timestamp": 1})
	if _, err := ParseSummary(string(piRequestJSON(bad))); err == nil {
		t.Fatal("persisted wire output accepted as summary")
	}
}

func TestPiRequestCompactionDigestAndCorruptRecords(t *testing.T) {
	a := []json.RawMessage{json.RawMessage(`{"role":"user","extension":9007199254740992}`)}
	b := []json.RawMessage{json.RawMessage(`{"extension":9007199254740992,"role":"user"}`)}
	c := []json.RawMessage{json.RawMessage(`{"role":"user","extension":9007199254740993}`)}
	if PrefixDigest(a) != PrefixDigest(b) || PrefixDigest(a) == PrefixDigest(c) {
		t.Fatal("digest lost JSON identity or numeric precision")
	}
	broken := []json.RawMessage{json.RawMessage(`{"role":"user","content":"task"}`), json.RawMessage(`{"role":"assistant","content":[{"type":"toolCall","id":"c","name":"write"}]}`), json.RawMessage(`{"role":"user","content":"later"}`)}
	if compactionCut(broken, 1, 1) != 0 {
		t.Fatal("compaction split an unfinished effect batch")
	}
}

func TestCompactionCutsHistoricalUserFactsButKeepsLoneTask(t *testing.T) {
	fact := piRequestJSON(map[string]string{"role": "user", "content": strings.Repeat("historical evidence ", 1000)})
	system := json.RawMessage(`{"role":"system","content":"policy"}`)
	question := json.RawMessage(`{"role":"user","content":"Explain the findings."}`)
	for _, tc := range []struct {
		name     string
		messages []json.RawMessage
		wantCut  int
	}{
		{"lone task", []json.RawMessage{system, fact}, 0},
		{"two facts and question", []json.RawMessage{system, fact, fact, question}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := compactionCut(tc.messages, 4096, 2048); got != tc.wantCut {
				t.Fatalf("cut = %d, want %d", got, tc.wantCut)
			}
		})
	}
}
