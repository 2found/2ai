package ai

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"strconv"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

type piMessagesProtocolFixture struct {
	UpstreamCommit string
	Model          json.RawMessage
	Cases          []struct {
		Input struct {
			Name, Wire, Action string
			Bytes              []byte
			ChunkSize          int
			ReadError          bool
		}
		RawEvents, Snapshots, Retained, Result json.RawMessage
	}
}

func readPiMessagesProtocolFixture(t *testing.T) piMessagesProtocolFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-messages-protocol.json")
	if err != nil {
		t.Fatal(err)
	}
	var f piMessagesProtocolFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 127 {
		t.Fatal("unexpected pi-messages protocol coverage")
	}
	return f
}

type piMessagesFixtureReader struct {
	data    []byte
	size    int
	failure bool
}

func (r *piMessagesFixtureReader) Read(p []byte) (int, error) {
	if len(r.data) == 0 {
		if r.failure {
			return 0, errors.New("reader failed")
		}
		return 0, io.EOF
	}
	n := len(p)
	if r.size > 0 && n > r.size {
		n = r.size
	}
	if n > len(r.data) {
		n = len(r.data)
	}
	copy(p, r.data[:n])
	r.data = r.data[n:]
	return n, nil
}
func TestPiMessagesProtocol(t *testing.T) {
	f := readPiMessagesProtocolFixture(t)
	for i, tc := range f.Cases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) {
			model := catalogDecode(t, f.Model).(*Object)
			now := float64(1000000)
			converter := newPiMessagesEventConverter(model, func() float64 { return now })
			data := tc.Input.Bytes
			if data == nil {
				data = []byte(string(jsonjs.StringCodePoints(tc.Input.Wire)))
			}
			rawEvents, snapshots, retained := NewArray(), NewArray(), NewArray()
			var result any
			terminal, aborted := false, false
			snapshot := func(value any) any { var clone jsonjs.ValueCloner; return clone.Clone(value) }
			err := readPiMessagesEvents(&piMessagesFixtureReader{data: data, size: tc.Input.ChunkSize, failure: tc.Input.ReadError}, func(event any) (bool, error) {
				if tc.Input.Action == "advance-clock" {
					now += 10
				}
				if tc.Input.Action == "mutate" && catalogProperty(event, "type") == "text_delta" {
					event.(*Object).Set("delta", "changed")
				}
				rawEvents.Append(snapshot(event))
				if tc.Input.Action == "callback-error" || tc.Input.Action == "callback-abort" {
					aborted = tc.Input.Action == "callback-abort"
					return false, errors.New("callback failed")
				}
				converted, err := converter.convert(event)
				if err != nil {
					return false, err
				}
				snapshots.Append(snapshot(converted))
				retained.Append(converted)
				kind := converted.Get("type")
				terminal = kind == "done" || kind == "error"
				if terminal {
					if kind == "done" {
						result = converted.Get("message")
					} else {
						result = converted.Get("error")
					}
				}
				return !terminal, nil
			})
			if err == nil && !terminal {
				err = errors.New("gateway stream ended without a terminal event")
			}
			if err != nil {
				// The transport catch creates a fresh empty message, not the converter's
				// partial output. This fixture records that boundary from public stream().
				failure := newPiMessagesEventConverter(model, func() float64 { return now }).partial
				reason := "error"
				if aborted {
					reason = "aborted"
				}
				failure.Set("stopReason", reason)
				failure.Set("errorMessage", err.Error())
				event := NewObject(Property{Name: "type", Value: "error"}, Property{Name: "reason", Value: reason}, Property{Name: "error", Value: failure})
				snapshots.Append(snapshot(event))
				retained.Append(event)
				result = failure
			}
			catalogCompare(t, rawEvents, tc.RawEvents)
			catalogCompare(t, snapshots, tc.Snapshots)
			catalogCompare(t, retained, tc.Retained)
			catalogCompare(t, result, tc.Result)
		})
	}
}
