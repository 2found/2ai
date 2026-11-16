package host

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/2found/2ai/core/ai"
)

// scanCorpus is messages in and out of the shape 2ai writes: every one must
// either scan to exactly what decodeFacts reads, or be left to decodeFacts.
var scanCorpus = []string{
	`{"role":"user","content":"hi","timestamp":1}`,
	`{"role":"user","content":[{"type":"text","text":"xin chào 😀 \ud83d\ude00"}],"timestamp":2}`,
	`{"role":"assistant","content":[{"type":"text","text":"a"},{"type":"toolCall","id":"c1","name":"read","arguments":{"path":"x"}},{"type":"toolCall","id":"c2","name":"read","arguments":null},{"type":"toolCall","id":"c3","name":"read"}],"stopReason":"toolUse","usage":{"input":10,"output":2.5,"cacheRead":1e2,"cacheWrite":null,"totalTokens":4},"timestamp":3}`,
	`{"role":"toolResult","toolCallId":"c1","toolName":"read","content":[{"type":"text","text":"ok"}],"isError":false,"timestamp":4}`,
	`{"role":"toolResult","toolCallId":"c2","toolName":"read","content":[],"isError":true,"timestamp":5}`,
	`{"role":"assistant","content":null,"stopReason":"stop"}`,
	`{"role":"assistant","stopReason":"error"}`,
	`{"role":"assistant","content":[null,{"type":null,"id":null}],"stopReason":"aborted"}`,
	`{"role":"user","content":"s","agentrayContextSummary":"older turns","timestamp":9}`,
	`{"role":"user","content":"s","agentrayContextSummary":"","agentrayContextReducedAt":12}`,
	`{"role":"system","sections":[{"section":"agentray","text":"be kind"}],"toolsAdded":[{"name":"read"}]}`,
	`{"timestamp":-0,"role":"user","content":"x"}`,
	`{"content":"key order","timestamp":7,"role":"user","usage":null}`,
	`{"role":"user","content":[{"type":"image","data":"` + strings.Repeat("QUJD", 400) + `","mimeType":"image/png"}]}`,
	// Out of shape: each must fall back.
	`{"Role":"user","content":"case"}`,
	`{"role":"user","ROLE":"assistant"}`,
	`{"role":"user","role":"assistant"}`,
	`{"r\u006fle":"assistant"}`,
	`{"role":"\u0075ser"}`,
	`{"role":7}`,
	`{"role":"user","timestamp":1.5}`,
	`{"role":"user","timestamp":1e3}`,
	`{"role":"user","timestamp":9223372036854775808}`,
	`{"role":"user","timestamp":"1"}`,
	`{"role":"user","isError":"yes"}`,
	`{"role":"user","agentrayContextSummary":3}`,
	`{"role":"assistant","content":"text","stopReason":"stop"}`,
	`{"role":"assistant","stopReason":"stop"}`,
	`{"role":"assistant","content":[1],"stopReason":"stop"}`,
	`{"role":"assistant","content":[{"Type":"toolCall","id":"x"}],"stopReason":"stop"}`,
	`{"role":"assistant","content":[{"type":"toolCall","id":"x","id":"y"}],"stopReason":"stop"}`,
	`{"role":"assistant","content":[{"type":"toolCall","id":"\u0078"}],"stopReason":"stop"}`,
	`{"role":"assistant","content":[{"type":"toolCall","id":7}],"stopReason":"stop"}`,
	`{"role":"assistant","usage":[],"content":[],"stopReason":"stop"}`,
	`{"role":"assistant","usage":{"input":"1"},"content":[],"stopReason":"stop"}`,
	`{"role":"assistant","usage":{"Input":1},"content":[],"stopReason":"stop"}`,
	`{"role":"assistant","usage":{"input":1e999},"content":[],"stopReason":"stop"}`,
	"{\"role\":\"us\xffer\"}",
	"{\"role\":\"user\",\"content\":\"bad \xff utf8\"}",
}

func TestScanFactsReadsWhatDecodeFactsReads(t *testing.T) {
	scanned := 0
	for _, raw := range scanCorpus {
		if !json.Valid([]byte(raw)) && !strings.Contains(raw, "\xff") {
			t.Fatalf("corpus entry is not JSON: %s", raw)
		}
		// A scanned message counts its tokens when first asked.
		settle := func(f messageFacts) messageFacts {
			f.tokenCount()
			f.raw = nil
			return f
		}
		want := settle(decodeFacts(json.RawMessage(raw)))
		if got, ok := scanFacts(json.RawMessage(raw)); ok {
			scanned++
			if got := settle(got); !reflect.DeepEqual(got, want) {
				t.Errorf("scan of %.80s\n got %+v\nwant %+v", raw, got, want)
			}
		}
		if got := settle(factsOf(json.RawMessage(raw))); !reflect.DeepEqual(got, want) {
			t.Errorf("factsOf(%.80s) = %+v, want %+v", raw, got, want)
		}
	}
	if scanned < 14 {
		t.Fatalf("only %d messages took the scan path", scanned)
	}
}

func TestBatchFactsAgreeWithDecodedValidation(t *testing.T) {
	for _, raw := range scanCorpus {
		if _, ok := scanFacts(json.RawMessage(raw)); !ok {
			continue
		}
		var m struct {
			Role, ToolCallID, StopReason string
			Content                      json.RawMessage
		}
		if err := json.Unmarshal([]byte(raw), &m); err != nil {
			t.Errorf("a scanned message must decode: %v: %s", err, raw)
			continue
		}
		if m.Role == "assistant" && m.StopReason != "error" && m.StopReason != "aborted" {
			var blocks []struct{ Type, ID string }
			if err := json.Unmarshal(m.Content, &blocks); err != nil {
				t.Errorf("a scanned assistant's content must decode: %v: %s", err, raw)
			}
		}
	}
}

func TestSplitMessagesAcceptsWhatEncodingJSONAccepts(t *testing.T) {
	deep := strings.Repeat("[", 9998) + strings.Repeat("]", 9998)
	tooDeep := strings.Repeat("[", 9999) + strings.Repeat("]", 9999)
	for _, raw := range []string{
		`[]`, ` [ ] `, `[{}]`, `[{"a":1},{"b":[1,2,{"c":"d"}]}]`, `[null]`, `[1,"x",true,false,null]`,
		`[{"a":"\u00e9\n\t\"\\\/"}]`, "[{\"a\":\"\xff\"}]", "[{\"a\":\"\x7f\"}]", `[-0,0.5,1e10,-1E-2]`,
		`[` + deep + `]`, `[` + tooDeep + `]`,
		``, ` `, `null`, `{}`, `[`, `]`, `[,]`, `[1,]`, `[{]`, `[{"a"}]`, `[{"a":}]`, `[{"a":1,}]`, `[01]`, `[1.]`, `[.5]`, `[1e]`, `[-]`,
		`[tru]`, `[nul]`, `["\x"]`, `["\u12"]`, "[\"\x01\"]", `[] []`, `[]x`, "\xef\xbb\xbf[]", `[{"a":1}`, `["a]`,
	} {
		var want []json.RawMessage
		wantErr := json.Unmarshal([]byte(raw), &want)
		if wantErr == nil && want == nil {
			wantErr = errInvalidJSON // null is no transcript
		}
		got, gotErr := SplitMessages(json.RawMessage(raw))
		if (gotErr == nil) != (wantErr == nil) {
			t.Errorf("SplitMessages(%.40q) error %v, encoding/json %v", raw, gotErr, wantErr)
			continue
		}
		if len(got) != len(want) {
			t.Errorf("SplitMessages(%.40q) = %d elements, want %d", raw, len(got), len(want))
			continue
		}
		for i := range got {
			if !bytes.Equal(got[i], want[i]) {
				t.Errorf("SplitMessages(%.40q)[%d] = %s, want %s", raw, i, got[i], want[i])
			}
		}
	}
}

func TestLocateMemberFindsOnlyAnUnambiguousMember(t *testing.T) {
	for _, tc := range []struct {
		raw, value string
		ok, err    bool
	}{
		{raw: `{"revision":2,"messages":[{"a":1}],"model":"m"}`, value: `[{"a":1}]`, ok: true},
		{raw: ` { "messages" : [ ] } `, value: `[ ]`, ok: true},
		{raw: `{"revision":2}`},
		{raw: `{"messages":[],"Messages":[]}`},
		{raw: `{"messages":[],"messages":[]}`},
		{raw: `{"m\u0065ssages":[]}`},
		{raw: `[]`},
		{raw: `{"messages":[}`, err: true},
		{raw: `{"messages":[]} x`, err: true},
		{raw: ``, err: true},
	} {
		start, end, ok, err := LocateMember(json.RawMessage(tc.raw), "messages")
		if (err != nil) != tc.err || ok != tc.ok {
			t.Errorf("LocateMember(%s) ok=%v err=%v, want ok=%v err=%v", tc.raw, ok, err, tc.ok, tc.err)
			continue
		}
		if ok && tc.raw[start:end] != tc.value {
			t.Errorf("LocateMember(%s) = %s, want %s", tc.raw, tc.raw[start:end], tc.value)
		}
	}
}

func TestEstimateTextTokensBytesMatchesTheStringEstimate(t *testing.T) {
	for _, text := range []string{"", "abc", "Nội dung trao đổi dài.", "😀😀 x", "bad \xff\xfe utf8 \xe2\x82", strings.Repeat("ố", 1000)} {
		if got, want := ai.EstimateTextTokensBytes([]byte(text)), ai.EstimateTextTokens(text); got != want {
			t.Errorf("EstimateTextTokensBytes(%q) = %v, want %v", text, got, want)
		}
	}
}

func TestReadingCanonicalMessagesAllocatesNothingPerMessage(t *testing.T) {
	raw := json.RawMessage(`{"role":"assistant","content":[{"type":"text","text":"Nội dung trao đổi dài."}],"api":"scripted","stopReason":"stop","usage":{"input":10,"output":2,"cacheRead":0,"cacheWrite":0,"totalTokens":12},"timestamp":1700000000000}`)
	if allocs := testing.AllocsPerRun(100, func() {
		if _, ok := scanFacts(raw); !ok {
			t.Fatal("canonical message left the scan path")
		}
	}); allocs != 0 {
		t.Fatalf("scanFacts allocated %.0f times", allocs)
	}
}
