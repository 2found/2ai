package host

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/ai/protocol"
)

// collapsedHead is the system prompt a provider without mid-conversation system
// messages receives for a request: the head of its collapsed transcript.
func collapsedHead(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var messages []ai.Message
	if err := json.Unmarshal(raw, &messages); err != nil {
		t.Fatal(err)
	}
	collapsed := ai.CollapseSystemMessages(ai.NormalizeContext(ai.Context{Messages: messages})).Messages()
	if len(collapsed) == 0 || collapsed[0].Role != "system" {
		t.Fatalf("collapsed transcript has no system head: %s", raw)
	}
	return ai.ContentText(collapsed[0].Content)
}

// A system message stated mid-transcript (the goal plugin's "[run goal]"
// statement) stays behind the summary when compaction folds its position, so
// the system+tools head a provider caches is the same before and after every
// compaction.
func TestCompactionKeepsALaterSystemMessageBehindTheSummary(t *testing.T) {
	const goal = "[run goal] This run finishes when:\nthe code is recalled"
	messages := nativeLongRequest()
	statement := nativeRequestJSON(map[string]string{"role": "system", "content": goal})
	messages = append(messages[:2:2], append([]json.RawMessage{statement}, messages[2:]...)...)
	raw := nativeRequestJSON(messages)
	c := Compactor{revision: "r", config: CompactionPolicy{Budget: 1500, KeepRecent: 500, Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
		return "Task and earlier evidence.", protocol.Usage{}, nil
	}}}
	view := c.testTransform(context.Background(), raw)
	if c.saved == nil || c.saved.PrefixCount <= 3 {
		t.Fatalf("compaction did not fold the statement's position: %+v", c.saved)
	}
	var got []json.RawMessage
	if err := json.Unmarshal(view, &got); err != nil {
		t.Fatal(err)
	}
	if !SameJSON(got[0], messages[0]) {
		t.Fatal("system prompt changed")
	}
	if !strings.Contains(string(got[1]), "agentrayContextSummary") {
		t.Fatalf("summary does not directly follow the system prompt: %s", got[1])
	}
	if !SameJSON(got[2], statement) {
		t.Fatalf("goal statement is not kept right behind the summary: %s", got[2])
	}
	before, after := collapsedHead(t, raw), collapsedHead(t, view)
	if before != after || strings.Contains(after, "[run goal]") {
		t.Fatalf("compaction rewrote the cached system head:\nbefore %q\nafter  %q", before, after)
	}
	// A forced second compaction over the summarized view keeps the head too.
	forced := c.config
	forced.Force = true
	prior := c.saved
	again := c.testTransformWith(context.Background(), view, forced)
	if c.saved == prior {
		t.Fatal("the forced second compaction did not run")
	}
	if head := collapsedHead(t, again); head != before {
		t.Fatalf("second compaction rewrote the system head: %q", head)
	}
	if !strings.Contains(string(again), "run goal") {
		t.Fatal("second compaction dropped the goal statement")
	}
}
