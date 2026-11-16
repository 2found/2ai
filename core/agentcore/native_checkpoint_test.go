package agentcore

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
)

// readNativeCheckpoint decodes every state as json.Unmarshal does, and hands
// back the transcript's elements as their stored bytes.
func TestReadNativeCheckpointReadsWhatUnmarshalReads(t *testing.T) {
	for _, state := range []string{
		`{"revision":"r","model":{"id":"m"},"messages":[{"role":"user","content":"\u00e9"}],"summary":{"a":1},"plugins":{"p":[1]}}`,
		` { "plugins" : {} , "messages" : [ {"role":"user"} , null ] , "goal" : "g" , "revision" : "r" } `,
		`{"question":{"callId":"c","args":{"x":1}},"revision":"r","messages":[],"goalRevisions":[{"previous":"a","goal":"b","reason":"c"}]}`,
		`{"revision":"r","messages":null}`,
		`{"revision":"r"}`,
		`{"Messages":[{"role":"user"}],"revision":"r"}`,
		`{"messages":[{"role":"user"}],"MESSAGES":[{"role":"assistant"}],"revision":"r"}`,
		`{"messages":[1],"messages":[2]}`,
		`{"m\u0065ssages":[{"role":"user"}],"revision":"r"}`,
		`{"revision":7,"messages":[]}`,
		`{"revision":"r","messages":{}}`,
		`{"revision":"r","messages":[1,]}`,
		`{"revision":"r","messages":[1]`,
		`{"revision":"r","messages":`,
		`{"revision":"r","messages":[1]} trailing`,
		`[]`,
		``,
	} {
		var want nativeCheckpoint
		wantErr := json.Unmarshal([]byte(state), &want)
		var wantStored []json.RawMessage
		if wantErr == nil {
			wantStored, wantErr = splitStoredMessages(want.Messages)
		}
		var got nativeCheckpoint
		stored, err := readNativeCheckpoint(json.RawMessage(state), &got)
		if (err == nil) != (wantErr == nil) {
			t.Errorf("readNativeCheckpoint(%s) error %v, want %v", state, err, wantErr)
			continue
		}
		if err != nil {
			continue
		}
		if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(stored, wantStored) {
			t.Errorf("readNativeCheckpoint(%s)\n got %+v %q\nwant %+v %q", state, got, stored, want, wantStored)
		}
	}
}

// writeNativeCheckpoint is json.Marshal of the checkpoint for messages in the
// encoding json.Marshal writes.
func TestWriteNativeCheckpointIsTheCheckpointsEncoding(t *testing.T) {
	messages := []json.RawMessage{
		json.RawMessage(`{"content":"\u003cb\u003e \u00e9 \ud83d\ude00","role":"user","timestamp":1}`),
		json.RawMessage(`{"content":[{"text":"ok","type":"text"}],"role":"assistant","stopReason":"stop","timestamp":2}`),
	}
	for _, checkpoint := range []nativeCheckpoint{
		{Revision: nativeCheckpointRevision, Model: json.RawMessage(`{"id":"m"}`)},
		{Question: &nativeQuestion{CallID: "c", Args: json.RawMessage(`{"q":"<?>"}`)}, Revision: "r", Model: json.RawMessage(`null`), Summary: json.RawMessage(`{"prefix_count":1}`), Cleared: json.RawMessage(`[1]`), Goal: "g & h", GoalRevisions: []GoalRevision{{Previous: "a", Goal: "b", Reason: "c"}}, Plugins: map[string]json.RawMessage{"z": json.RawMessage(`1`), "a": json.RawMessage(`{"k":[]}`)}},
	} {
		for _, transcript := range [][]json.RawMessage{nil, messages} {
			got, err := writeNativeCheckpoint(checkpoint, transcript)
			if err != nil {
				t.Fatal(err)
			}
			checkpoint.Messages = joinRawMessages(transcript)
			want, err := json.Marshal(checkpoint)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("writeNativeCheckpoint\n got %s\nwant %s", got, want)
			}
		}
	}
}
