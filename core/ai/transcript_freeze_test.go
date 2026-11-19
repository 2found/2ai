package ai

import (
	"encoding/json"
	"strings"
	"testing"
)

// The JSON round trip freezeTranscript replaced; kept here as the reference
// the typed copy must match on the wire.
func roundTripTranscript(t testing.TB, transcript TranscriptContext) TranscriptContext {
	t.Helper()
	encoded, err := json.Marshal(transcript)
	if err != nil {
		t.Fatal(err)
	}
	var frozen Context
	if err := json.Unmarshal(encoded, &frozen); err != nil {
		t.Fatal(err)
	}
	return NormalizeContext(frozen)
}

func freezeFixture(t testing.TB) TranscriptContext {
	t.Helper()
	// Decoded records carry wire-shape metadata: unknown fields, explicit nulls,
	// a non-string model identity, a lone surrogate, null content entries.
	wire := `[
		{"role":"user","content":"hi \ud800 there","timestamp":1,"custom":{"a":[1,2]}},
		{"role":"user","content":[{"type":"text","text":"x","textSignature":null},null,{"type":"image","data":"QUJD","mimeType":"image/png","extra":true}],"timestamp":2},
		{"role":"assistant","content":[{"type":"thinking","thinking":"hm","thinkingSignature":"sig"},{"type":"toolCall","id":"c1","name":"read","arguments":{"path":"a"}}],"api":"anthropic-messages","provider":"anthropic","model":5,"usage":{"input":1,"output":2,"cacheRead":0,"cacheWrite":0,"totalTokens":3,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"toolUse","timestamp":3,"responseId":null},
		{"role":"toolResult","toolCallId":"c1","toolName":"read","content":[{"type":"text","text":"body"}],"details":{"lines":[1,2,3]},"isError":false,"timestamp":4},
		{"role":"system","content":"sys","sections":{"b":"two","1":"one","a":null},"toolsAdded":[{"name":"t","description":"d","parameters":{"type":"object"},"x":1}],"toolsRemoved":[{"name":"old","why":"gone"}],"timestamp":5},
		{"role":"user","timestamp":6}
	]`
	var messages []Message
	if err := json.Unmarshal([]byte(wire), &messages); err != nil {
		t.Fatal(err)
	}
	// Natively built records: live Details, a zero usage, nil content.
	details := NewObject()
	details.Set("n", 1.0)
	messages = append(messages,
		Message{Role: "toolResult", ToolCallID: "c2", ToolName: "bash", Content: TextContent("out"), Details: details},
		Message{Role: "assistant", Content: BlockContent(ContentBlock{Type: "text", Text: "done"}), StopReason: "stop"},
		Message{Role: "user"},
	)
	return NormalizeContext(Context{Messages: messages})
}

func mustMarshal(t testing.TB, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestFreezeTranscriptMatchesJSONRoundTrip(t *testing.T) {
	transcript := freezeFixture(t)
	want := mustMarshal(t, roundTripTranscript(t, transcript))
	if got := mustMarshal(t, freezeTranscript(transcript)); got != want {
		t.Fatalf("frozen transcript differs from the round trip\n got: %s\nwant: %s", got, want)
	}
	if got := mustMarshal(t, transcript); got != want {
		t.Fatalf("round trip is not lossless for this fixture\n got: %s\nwant: %s", got, want)
	}
	// Every provider resolves the frozen copy next; the resolved transcript
	// must also be the same, with and without mid-conversation system messages.
	for _, mid := range []bool{false, true} {
		want := mustMarshal(t, ResolveTranscript(roundTripTranscript(t, transcript), mid))
		if got := mustMarshal(t, ResolveTranscript(freezeTranscript(transcript), mid)); got != want {
			t.Fatalf("resolved (mid=%v) differs\n got: %s\nwant: %s", mid, got, want)
		}
	}
	if got := mustMarshal(t, freezeTranscript(TranscriptContext{})); got != `{"messages":[]}` {
		t.Fatalf("empty transcript froze to %s", got)
	}
}

func TestFreezeTranscriptDetachesFromCaller(t *testing.T) {
	transcript := freezeFixture(t)
	frozen := freezeTranscript(transcript)
	want := mustMarshal(t, frozen)

	messages := transcript.Messages()
	*messages[0].Content.Text = "changed"
	messages[0].Extra["custom"][2] = '9'
	messages[1].Content.Blocks.Get(0).Text = "changed"
	messages[1].Content.Blocks.Append(&ContentBlock{Type: "text", Text: "appended"})
	call := messages[2].Content.Blocks.Get(1)
	call.Arguments[2] = 'X'
	messages[2].Usage.Output = 99
	messages[4].Sections[0].Value = nil
	messages[4].ToolsAdded[0].Parameters[2] = 'X'
	messages[4].ToolsRemoved[0].Extra["why"] = json.RawMessage(`"changed"`)
	messages[6].Details.(*Object).Set("n", 2.0)
	messages[7].StopReason = "error"

	if got := mustMarshal(t, frozen); got != want {
		t.Fatalf("caller mutation reached the frozen transcript\n got: %s\nwant: %s", got, want)
	}
}

func TestFreezeTranscriptResetsFieldsJSONDoesNotCarry(t *testing.T) {
	message := Message{Role: "assistant", StopReason: "stop", reminder: true, Usage: &Usage{Observation: UsageObservation{UsageObserved: true, UsageSource: "provider"}}}
	frozen := freezeTranscript(NormalizeContext(Context{Messages: []Message{message}})).Messages()[0]
	if frozen.reminder || frozen.Usage.Observation != (UsageObservation{}) {
		t.Fatalf("reminder %v or usage observation %+v survived the freeze", frozen.reminder, frozen.Usage.Observation)
	}
}

// A long run's transcript: the freeze runs once per model call on all of it.
func benchmarkTranscript(b *testing.B) TranscriptContext {
	body := strings.Repeat("tool output line\n", 240)
	messages := []Message{{Role: "user", Content: TextContent("task"), Timestamp: 1}}
	for i := range 100 {
		messages = append(messages,
			Message{Role: "assistant", API: "anthropic-messages", Provider: "anthropic", Model: "m", StopReason: "toolUse", Usage: &Usage{Input: 1}, Content: BlockContent(
				ContentBlock{Type: "text", Text: "reading"},
				ContentBlock{Type: "toolCall", ID: "c", Name: "read", Arguments: json.RawMessage(`{"path":"file.go","offset":10}`)},
			), Timestamp: int64(i)},
			Message{Role: "toolResult", ToolCallID: "c", ToolName: "read", Content: TextContent(body), Timestamp: int64(i)},
		)
	}
	return NormalizeContext(Context{Messages: messages})
}

func BenchmarkFreezeTranscript(b *testing.B) {
	transcript := benchmarkTranscript(b)
	b.Run("typed", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			freezeTranscript(transcript)
		}
	})
	b.Run("json-roundtrip", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			roundTripTranscript(b, transcript)
		}
	})
}
