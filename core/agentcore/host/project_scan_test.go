package host

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/2found/2ai/core/ai"
)

// projectionCorpus adds content shapes ai.ProjectContent distinguishes to
// scanCorpus.
var projectionCorpus = []string{
	`{"role":"assistant","content":[{"type":"text","text":"a"},{"type":"text","text":"b"},{"type":"image","data":"QUJD","mimeType":"image/png"},{"type":"toolCall","id":"c","name":"n","arguments":{"x":[1, 2]}},{"type":"toolCall","id":"d","name":"n","arguments":null},{"type":"toolCall","id":"e","name":"n"},{"type":"thinking","thinking":"t","thinkingSignature":"s"}],"usage":{"input":1,"output":2,"cacheRead":3,"cacheWrite":4,"totalTokens":10,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0.5}},"stopReason":"toolUse"}`,
	`{"role":"toolResult","toolCallId":"c","toolName":"n","content":[{"type":"text","text":"ok"}],"isError":true}`,
	`{"role":"user","content":[null,{"type":"text","text":"y"},{"type":null,"text":"z"}]}`,
	`{"role":"user","content":[]}`,
	`{"role":"user","content":[{"type":"text"}]}`,
	`{"role":"assistant","usage":null,"content":[]}`,
	`{"role":"assistant","usage":{"input":1},"content":[]}`,
	`{"role":"assistant","usage":{"input":"1"},"content":[]}`,
	`{"role":"custom","content":7}`,
	`{"role":"user","tool_call_id":"x","content":"u"}`,
	`{"role":"user","TOOLNAME":"x","content":"u"}`,
	`{"role":"user","content":[{"type":"text","text":"x","id":5}]}`,
	`{"role":"user","content":[{"type":"text","Text":"x"}]}`,
	`{"role":"user","content":[{"type":"text","text":"x","text":"y"}]}`,
	`{"role":"user","content":[{"type":"text","text":"esc \n"}]}`,
	`{"role":"user","content":"esc \n"}`,
	`{"role":"user","content":{}}`,
	`{"role":"user","content":[1]}`,
	`{"role":"user","content":7}`,
	`{"role":"system","sections":[{"section":"agentray","text":"be kind"}]}`,
	`{}`,
	`{"content":"no role"}`,
}

func TestScanProjectionReadsWhatProjectMessageReads(t *testing.T) {
	scanned := 0
	for _, raw := range append(append([]string(nil), scanCorpus...), projectionCorpus...) {
		want, wantErr := ai.ProjectMessage(json.RawMessage(raw))
		if got, ok := scanProjection(json.RawMessage(raw)); ok {
			scanned++
			if wantErr != nil {
				t.Errorf("scan of %.80s succeeded, ai.ProjectMessage failed: %v", raw, wantErr)
			} else if !reflect.DeepEqual(got, want) {
				t.Errorf("scan of %.80s\n got %+v\nwant %+v", raw, got, want)
			}
		}
		got, err := ProjectMessage(json.RawMessage(raw))
		if (err == nil) != (wantErr == nil) || (err == nil && !reflect.DeepEqual(got, want)) {
			t.Errorf("ProjectMessage(%.80s) = %+v, %v; want %+v, %v", raw, got, err, want, wantErr)
		}
	}
	if scanned < 20 {
		t.Fatalf("only %d messages took the scan path", scanned)
	}
}

func TestScanUsageReadsWhatProjectUsageReads(t *testing.T) {
	scanned := 0
	for _, raw := range []string{
		`{"input":10,"output":2.5,"cacheRead":1e2,"cacheWrite":0,"totalTokens":12,"cost":{"input":0.1,"output":0.2,"cacheRead":0,"cacheWrite":0,"total":0.3}}`,
		`{"input":10,"output":2}`,
		`{"input":null,"output":-0,"cacheWrite1h":5,"reasoning":null,"cost":{"total":null}}`,
		`{"input":1,"cacheWrite1h":7.9,"reasoning":3,"extra":"x","cost":{"total":1,"currency":"USD"}}`,
		`{}`,
		`{"input":1,"cost":null}`,
		`{"input":1,"cost":5}`,
		`{"input":1,"cost":{"total":"1"}}`,
		`{"input":"1"}`,
		`{"input":1e999}`,
		`{"Input":1}`,
		`{"input":1,"input":2}`,
		`{"input":true}`,
		`[]`,
	} {
		want, wantErr := ai.ProjectUsage(json.RawMessage(raw))
		if got, ok := scanUsage([]byte(raw)); ok {
			scanned++
			if wantErr != nil || !reflect.DeepEqual(got, want) {
				t.Errorf("scanUsage(%s) = %+v, want %+v, %v", raw, got, want, wantErr)
			}
		}
	}
	if scanned < 4 {
		t.Fatalf("only %d usage objects took the scan path", scanned)
	}
}
