package agentcore_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/2found/2ai/core/agentcore"
	nativehost "github.com/2found/2ai/core/agentcore/host"
	"github.com/2found/2ai/core/ai"
)

// storedTranscriptCase is one stored message. written marks bytes 2ai itself
// writes (json.Marshal of an ai.Message); the others are what a consumer or an
// older writer may hand back.
type storedTranscriptCase struct {
	name    string
	raw     string
	written bool
}

func storedTranscriptCases(tb testing.TB) []storedTranscriptCase {
	tb.Helper()
	write := func(name string, m ai.Message) storedTranscriptCase {
		raw, err := json.Marshal(m)
		if err != nil {
			tb.Fatal(err)
		}
		return storedTranscriptCase{name: name, raw: string(raw), written: true}
	}
	usage := &ai.Usage{Input: 120, Output: 7, CacheRead: 3, TotalTokens: 130}
	return []storedTranscriptCase{
		write("user text with unicode, HTML and control characters", ai.Message{Role: "user", Content: ai.TextContent("xin chào 😀 <b>&amp;</b> \u2028 \x01 \"q\" \\ end"), Timestamp: 10}),
		write("assistant tool calls", ai.Message{Role: "assistant", API: "scripted", Provider: "scripted", Model: "bench", StopReason: "toolUse", Usage: usage, Timestamp: 11, Content: ai.BlockContent(
			ai.ContentBlock{Type: "text", Text: "reading"},
			ai.ContentBlock{Type: "toolCall", ID: "call-1", Name: "read", Arguments: json.RawMessage(`{"z":1,"a":[null,{"y":"\ud83d\ude00"}],"m":{}}`)},
			ai.ContentBlock{Type: "toolCall", ID: "call-2", Name: "read", Arguments: json.RawMessage(`null`)},
		)}),
		write("tool result", ai.Message{Role: "toolResult", ToolCallID: "call-1", ToolName: "read", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "line\n\tend"}), Timestamp: 12}),
		write("tool error result", ai.Message{Role: "toolResult", ToolCallID: "call-2", ToolName: "read", IsError: true, Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "no such file"}), Timestamp: 13}),
		write("assistant reply", ai.Message{Role: "assistant", API: "scripted", Provider: "scripted", Model: "bench", StopReason: "stop", Usage: usage, Timestamp: 14, Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})}),
		{name: "unusual key order", raw: `{"timestamp":20,"content":[{"text":"key order","type":"text"}],"role":"user"}`},
		{name: "escapes and a lone surrogate", raw: `{"role":"user","content":"lone \ud800 surrogate, \u00e9 escaped, \/ slash","timestamp":21}`},
		{name: "explicit nulls beside absent fields", raw: `{"role":"assistant","content":[{"type":"text","text":"raw <b> & </b>","textSignature":null}],"api":"scripted","stopReason":"stop","errorMessage":null,"responseId":null,"timestamp":22}`},
		{name: "whitespace inside a message", raw: "{ \"role\" : \"user\" ,\n \"content\" : \"spaced\" , \"timestamp\" : 23 }"},
		{name: "tool batch in unusual key order", raw: `{"stopReason":"toolUse","role":"assistant","content":[{"arguments":{"b":2,"a":1},"name":"read","id":"call-3","type":"toolCall"}],"timestamp":24}`},
		{name: "its result in unusual key order", raw: `{"toolName":"read","toolCallId":"call-3","role":"toolResult","content":[{"type":"text","text":"ok"}],"isError":false,"timestamp":25}`},
		{name: "unknown member", raw: `{"role":"assistant","content":[{"type":"text","text":"fin"}],"stopReason":"stop","futureField":{"kept":[1,2]},"timestamp":26}`},
	}
}

// spliceTranscript is state with its transcript replaced by messages, laid
// out with whitespace between elements, without re-encoding any message.
func spliceTranscript(tb testing.TB, state json.RawMessage, messages []string) json.RawMessage {
	tb.Helper()
	var checkpoint map[string]json.RawMessage
	if err := json.Unmarshal(state, &checkpoint); err != nil {
		tb.Fatal(err)
	}
	delete(checkpoint, "messages")
	rest, err := json.Marshal(checkpoint)
	if err != nil {
		tb.Fatal(err)
	}
	return json.RawMessage(string(rest[:len(rest)-1]) + `, "messages" : [` + "\n  " + strings.Join(messages, ",\n  ") + "\n]}")
}

// storedMessages are the exact bytes of every message in a state.
func storedMessages(tb testing.TB, state json.RawMessage) []json.RawMessage {
	tb.Helper()
	var checkpoint struct{ Messages json.RawMessage }
	if err := json.Unmarshal(state, &checkpoint); err != nil {
		tb.Fatal(err)
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(checkpoint.Messages, &messages); err != nil {
		tb.Fatal(err)
	}
	return messages
}

// rewrittenByDecode is how a run before the transcript was read in place
// wrote a stored message back: decoded into an ai.Message, then encoded.
func rewrittenByDecode(tb testing.TB, raw json.RawMessage) json.RawMessage {
	tb.Helper()
	var message ai.Message
	if err := json.Unmarshal(raw, &message); err != nil {
		tb.Fatal(err)
	}
	out, err := json.Marshal(&message)
	if err != nil {
		tb.Fatal(err)
	}
	return out
}

func TestNativeRunWritesStoredMessagesBackByteForByte(t *testing.T) {
	a := transcriptBenchAgent(t)
	head := transcriptBenchState(t, a, 3)
	var messages []string
	for _, raw := range storedMessages(t, head) {
		messages = append(messages, string(raw))
	}
	cases := storedTranscriptCases(t)
	for _, c := range cases {
		if !json.Valid([]byte(c.raw)) {
			t.Fatalf("%s is not JSON: %s", c.name, c.raw)
		}
		messages = append(messages, c.raw)
	}
	state := spliceTranscript(t, head, messages)
	for _, compact := range []bool{false, true} {
		input := state
		if compact {
			// The summary covers the stored messages; they still pass
			// through unread.
			compacted, err := a.RunNative(context.Background(), agentcore.NativeRun{State: state, ControlOnly: true, Compact: &agentcore.ManualCompaction{}})
			if err != nil || len(compacted.Compactions) != 1 || compacted.Compactions[0].Failed {
				t.Fatalf("compact: %v %+v", err, compacted.Compactions)
			}
			input = compacted.NativeState
		}
		result := runTranscriptBenchTurn(t, a, input)
		if !json.Valid(result.NativeState) {
			t.Fatalf("compact=%v: the state is not JSON", compact)
		}
		out := storedMessages(t, result.NativeState)
		if len(out) != len(messages)+2 {
			t.Fatalf("compact=%v: %d messages, want %d", compact, len(out), len(messages)+2)
		}
		for i, raw := range messages {
			if !bytes.Equal(out[i], []byte(raw)) {
				t.Errorf("compact=%v: message %d changed\n got %s\nwant %s", compact, i, out[i], raw)
			}
		}
		// The run's own messages are encoded as an ai.Message encodes.
		for _, raw := range out[len(messages):] {
			if again := rewrittenByDecode(t, raw); !bytes.Equal(again, raw) {
				t.Errorf("compact=%v: appended message is not in its own encoding\n got %s\nwant %s", compact, raw, again)
			}
		}
		if len(result.Messages) != 2 || result.Messages[0].Content != "one more question" {
			t.Errorf("compact=%v: RunResult.Messages = %+v, want this run's two messages", compact, result.Messages)
		}
	}
	// What 2ai writes, decoding wrote back unchanged too: for these messages
	// the state is the same bytes as before.
	for _, c := range cases {
		if again := rewrittenByDecode(t, json.RawMessage(c.raw)); c.written && !bytes.Equal(again, []byte(c.raw)) {
			t.Errorf("%s: decoding rewrote a message 2ai wrote\n got %s\nwant %s", c.name, again, c.raw)
		} else if !c.written && !bytes.Equal(again, []byte(c.raw)) {
			t.Logf("%s: decoding used to rewrite it as %s", c.name, again)
		}
	}
}

func TestNativeRunStillRejectsCorruptStoredHistory(t *testing.T) {
	a := transcriptBenchAgent(t)
	head := transcriptBenchState(t, a, 3)
	var messages []string
	for _, raw := range storedMessages(t, head) {
		messages = append(messages, string(raw))
	}
	call := func(id string) string {
		return `{"role":"assistant","content":[{"type":"toolCall","id":"` + id + `","name":"read","arguments":{}}],"stopReason":"toolUse","timestamp":30}`
	}
	result := func(id string) string {
		return `{"role":"toolResult","toolCallId":"` + id + `","toolName":"read","content":[{"type":"text","text":"ok"}],"timestamp":31}`
	}
	user := `{"role":"user","content":"next","timestamp":32}`
	reply := `{"role":"assistant","content":[{"type":"text","text":"fin"}],"stopReason":"stop","timestamp":33}`
	for _, tc := range []struct {
		name     string
		stored   []string
		unsettle bool
	}{
		{name: "open batch mid-history", stored: []string{call("c1"), user, reply}, unsettle: true},
		{name: "half-open batch mid-history", stored: []string{`{"role":"assistant","content":[{"type":"toolCall","id":"c1","name":"read"},{"type":"toolCall","id":"c2","name":"read"}],"stopReason":"toolUse","timestamp":30}`, result("c1"), user, reply}, unsettle: true},
		{name: "open batch at the end", stored: []string{user, call("c1")}, unsettle: true},
		{name: "half-open batch at the end", stored: []string{user, `{"role":"assistant","content":[{"type":"toolCall","id":"c1","name":"read"},{"type":"toolCall","id":"c2","name":"read"}],"stopReason":"toolUse","timestamp":30}`, result("c2")}, unsettle: true},
		{name: "a message that is not JSON", stored: []string{user, `{"role":"assistant",}`, reply}},
		{name: "a message of the wrong type", stored: []string{user, `{"role":"user","content":7,"timestamp":1}`, reply}},
		{name: "a member of the wrong type", stored: []string{user, `{"role":"assistant","content":[{"type":"text","text":5}],"stopReason":"stop"}`, reply}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := spliceTranscript(t, head, append(append([]string(nil), messages...), tc.stored...))
			_, err := a.RunNative(context.Background(), agentcore.NativeRun{State: state, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "one more question"}}})
			if err == nil {
				t.Fatal("a corrupt stored transcript resumed")
			}
			if tc.unsettle && !errors.Is(err, nativehost.ErrUnsettledEffect) {
				t.Fatalf("error %v, want an unsettled tool batch", err)
			}
		})
	}
	t.Run("open batch inside a summarized prefix", func(t *testing.T) {
		compacted := transcriptBenchCompacted(t, a, 40)
		var stored []string
		for _, raw := range storedMessages(t, compacted) {
			stored = append(stored, string(raw))
		}
		stored[10] = call("c1")
		_, err := a.RunNative(context.Background(), agentcore.NativeRun{State: spliceTranscript(t, compacted, stored), Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "one more question"}}})
		if !errors.Is(err, nativehost.ErrUnsettledEffect) {
			t.Fatalf("error %v, want an unsettled tool batch", err)
		}
	})
	for _, tc := range []struct{ name, state string }{
		{name: "a transcript that is not JSON", state: string(head[:len(head)-1]) + `,"messages":[{"role":"user"},]}`},
		{name: "a transcript that is no array", state: string(head[:len(head)-1]) + `,"messages":{"role":"user"}}`},
		{name: "a state cut short", state: string(head[:len(head)/2])},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var checkpoint map[string]json.RawMessage
			if json.Unmarshal(head, &checkpoint) != nil {
				t.Fatal("head is not JSON")
			}
			if _, err := a.RunNative(context.Background(), agentcore.NativeRun{State: json.RawMessage(tc.state), Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "one more question"}}}); err == nil {
				t.Fatal("a corrupt state resumed")
			}
		})
	}
}

// A run on a compacted checkpoint reads the messages it sends and the ones it
// adds; the stored ones only pass through, so its allocations do not grow with
// the stored transcript.
func TestNativeRunAllocationsDoNotGrowWithTheStoredTranscript(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 4,000-message checkpoint")
	}
	a := transcriptBenchAgent(t)
	allocs := func(n int) float64 {
		state := transcriptBenchCompacted(t, a, n)
		return testing.AllocsPerRun(3, func() { runTranscriptBenchTurn(t, a, state) })
	}
	small, large := allocs(50), allocs(4000)
	t.Logf("allocations per run: %.0f at 50 messages, %.0f at 4,000", small, large)
	if large > 1.25*small {
		t.Fatalf("a run on 4,000 stored messages allocates %.0f times, on 50 %.0f", large, small)
	}
}

// A summary written before the raw prefix digest existed still applies, and
// the next state carries the digest.
func TestNativeRunAppliesASummaryWithoutARawDigest(t *testing.T) {
	var views []int
	reply := ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "ok"})}
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{
		Model: json.RawMessage(`{"id":"bench","api":"scripted","provider":"scripted","contextWindow":100000000}`),
		Stream: func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
			if strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
				return ai.ScriptedStream(*summaryReply("<summary>1. Goal and intent: a long benchmark thread.</summary>"))(ctx, model, view, options)
			}
			views = append(views, len(view.Messages()))
			return ai.ScriptedStream(reply)(ctx, model, view, options)
		},
	}}}
	limits := agentcore.Limits{MaxTurns: 4, MaxToolCalls: 4, MaxContextTokens: 50_000_000}
	a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "bench", Limits: &limits})
	if err != nil {
		t.Fatal(err)
	}
	state := transcriptBenchCompacted(t, a, 40)
	var checkpoint map[string]json.RawMessage
	if err := json.Unmarshal(state, &checkpoint); err != nil {
		t.Fatal(err)
	}
	var summary map[string]json.RawMessage
	if err := json.Unmarshal(checkpoint["summary"], &summary); err != nil || summary["prefix_raw_digest"] == nil {
		t.Fatalf("a new summary has no raw digest: %s", checkpoint["summary"])
	}
	delete(summary, "prefix_raw_digest")
	old, _ := json.Marshal(summary)
	checkpoint["summary"] = old
	legacy, _ := json.Marshal(checkpoint)

	views = nil
	result := runTranscriptBenchTurn(t, a, legacy)
	if len(result.Compactions) != 0 || len(views) != 1 || views[0] > 6 {
		t.Fatalf("the old summary did not apply: compactions %+v, request sizes %v", result.Compactions, views)
	}
	var next map[string]json.RawMessage
	if err := json.Unmarshal(result.NativeState, &next); err != nil {
		t.Fatal(err)
	}
	var upgraded map[string]json.RawMessage
	if err := json.Unmarshal(next["summary"], &upgraded); err != nil || upgraded["prefix_raw_digest"] == nil || string(upgraded["prefix_digest"]) != string(summary["prefix_digest"]) {
		t.Fatalf("the next state's summary %s, want the old one with a raw digest", next["summary"])
	}
	// Resuming from the upgraded state takes the raw digest.
	views = nil
	if again := runTranscriptBenchTurn(t, a, result.NativeState); len(again.Compactions) != 0 || len(views) != 1 || views[0] > 8 {
		t.Fatalf("the upgraded summary did not apply: compactions %+v, request sizes %v", again.Compactions, views)
	}
}
