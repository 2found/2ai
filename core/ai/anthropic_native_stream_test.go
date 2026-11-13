package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"testing"
)

// Preserve the oracle's fetch read boundaries, including a CR and LF delivered
// in separate reads. EOF is delivered on the next read, as in ReadableStream.
type anthropicFixtureReader struct {
	body      []byte
	sizes     []int
	index     int
	remaining int
}

func (r *anthropicFixtureReader) Read(p []byte) (int, error) {
	if len(r.body) == 0 {
		return 0, io.EOF
	}
	if r.remaining == 0 {
		r.remaining = len(r.body)
		if len(r.sizes) > 0 {
			r.remaining = r.sizes[r.index%len(r.sizes)]
			r.index++
		}
	}
	n := min(len(p), len(r.body), r.remaining)
	copy(p, r.body[:n])
	r.body = r.body[n:]
	r.remaining -= n
	return n, nil
}

func TestPiAnthropicStreamOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-anthropic-stream.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          map[string]json.RawMessage
		Cases          []struct {
			Input struct {
				Name       string
				Model      map[string]json.RawMessage
				Options    map[string]json.RawMessage
				OAuth      bool
				Tools      []Tool
				Chunks     []json.RawMessage
				Body       *string
				PartSizes  []int
				FailBefore *int
				AbortAtEnd bool
				Mutate     string
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 86 {
		t.Fatal("unexpected stream oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			fields := map[string]json.RawMessage{}
			for k, v := range fixture.Model {
				fields[k] = v
			}
			for k, v := range tc.Input.Model {
				fields[k] = v
			}
			encoded, _ := json.Marshal(fields)
			var model completionsModel
			if err := json.Unmarshal(encoded, &model); err != nil {
				t.Fatal(err)
			}
			stream := NewAssistantMessageEventStream()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			acc := newAnthropicAccumulator(model, tc.Input.OAuth, tc.Input.Tools, tc.Input.Options, stream, func() int64 { return 100 })
			events := []json.RawMessage{}
			live := []AssistantMessageEvent{}
			acc.push = func(event AssistantMessageEvent) {
				encoded, err := json.Marshal(event)
				if err != nil {
					t.Fatal(err)
				}
				events = append(events, encoded)
				live = append(live, event)
				if tc.Input.AbortAtEnd && event.Type == "text_end" {
					cancel()
				}
				stream.Push(event)
			}
			var body bytes.Buffer
			if tc.Input.Body != nil {
				body.WriteString(*tc.Input.Body)
			} else {
				for _, chunk := range tc.Input.Chunks {
					fields, _ := samplingObject(chunk)
					fmt.Fprintf(&body, "event: %s\ndata: ", samplingString(fields["type"]))
					if err := json.Compact(&body, chunk); err != nil {
						t.Fatal(err)
					}
					body.WriteString("\n\n")
				}
			}
			acc.start()
			received := 0
			err := readAnthropicSSE(ctx, &anthropicFixtureReader{body: body.Bytes(), sizes: tc.Input.PartSizes}, func(raw json.RawMessage) error {
				i := received
				received++
				if tc.Input.FailBefore != nil && i == *tc.Input.FailBefore {
					return errors.New("fixture failure")
				}
				fields, _ := samplingObject(raw)
				switch {
				case tc.Input.Mutate == "stop" && samplingString(fields["type"]) == "message_delta":
					delta, _ := samplingObject(fields["delta"])
					delta["stop_reason"] = json.RawMessage(`"max_tokens"`)
					fields["delta"], _ = json.Marshal(delta)
					raw, _ = json.Marshal(fields)
				case tc.Input.Mutate == "end" && samplingString(fields["type"]) == "message_stop", tc.Input.Mutate == "start" && samplingString(fields["type"]) == "message_start":
					fields["type"] = json.RawMessage(`"ignored"`)
					raw, _ = json.Marshal(fields)
				}
				return acc.chunk(raw)
			})
			if err != nil {
				acc.fail(err.Error(), ctx.Err() != nil)
			} else {
				acc.finish(ctx)
			}
			result, err := stream.Result(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"events": events, "result": result})
			if err != nil {
				t.Fatal(err)
			}
			decode := func(raw []byte) any {
				var value any
				d := json.NewDecoder(bytes.NewReader(raw))
				d.UseNumber()
				if err := d.Decode(&value); err != nil {
					t.Fatal(err)
				}
				return value
			}
			if !reflect.DeepEqual(decode(actual), decode(tc.Expected)) {
				t.Fatalf("stream mismatch\nGo: %s\nPi: %s", actual, tc.Expected)
			}
			for _, event := range live {
				if event.Partial != nil && event.Partial != result {
					t.Fatal("partial pointer was replaced")
				}
			}
			if err := stream.WaitForEnd(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestAnthropicStreamConcurrentSnapshots(t *testing.T) {
	stream := NewAssistantMessageEventStream()
	acc := newAnthropicAccumulator(completionsModel{ID: "test"}, false, nil, nil, stream, func() int64 { return 100 })
	content := acc.output.Content
	producer := make(chan error, 1)
	go func() {
		acc.start()
		send := func(raw string) bool {
			if err := acc.chunk(json.RawMessage(raw)); err != nil {
				acc.fail(err.Error(), false)
				producer <- err
				return false
			}
			return true
		}
		if !send(`{"type":"content_block_start","index":500,"content_block":{"type":"text","text":""}}`) {
			return
		}
		for i := 0; i < 100; i++ {
			if !send(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"tool_use","id":"call_%d","name":"tool","input":{}}}`, i, i)) {
				return
			}
			if !send(`{"type":"content_block_delta","index":500,"delta":{"type":"text_delta","text":"x"}}`) {
				return
			}
			if !send(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":"{\"value\":1}"}}`, i)) {
				return
			}
			if !send(fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, i)) {
				return
			}
		}
		if !send(`{"type":"content_block_stop","index":500}`) {
			return
		}
		if !send(`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`) {
			return
		}
		acc.finish(context.Background())
		producer <- nil
	}()
	ends := map[int]*ContentBlock{}
	var partial *Message
	for {
		event, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		if event.Partial != nil {
			if partial == nil {
				partial = event.Partial
			}
			if partial != event.Partial {
				t.Fatal("changed partial identity")
			}
		}
		if event.ToolCall != nil {
			ends[event.ContentIndex] = event.ToolCall
			stream.Synchronize(func() {
				if event.Partial.Content.Blocks.Get(event.ContentIndex) != event.ToolCall {
					t.Fatal("toolcall_end and transcript do not share the block")
				}
			})
		}
		snapshot, err := stream.SnapshotEvent(event)
		if err != nil {
			t.Fatal(err)
		}
		if snapshot.ToolCall != nil && snapshot.Partial.Content.Blocks.Get(snapshot.ContentIndex) != snapshot.ToolCall {
			t.Fatal("snapshot separated toolcall_end from its transcript block")
		}
	}
	if err := <-producer; err != nil {
		t.Fatal(err)
	}
	if err := stream.WaitForEnd(context.Background()); err != nil {
		t.Fatal(err)
	}
	stream.Synchronize(func() {
		if content.Blocks != acc.output.Content.Blocks || content.Blocks.Len() != 101 {
			t.Fatal("provider growth did not reach the retained content list")
		}
	})
	result, err := stream.SnapshotResult(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "toolUse" || result.Content.Blocks.Len() != 101 || len(ends) != 100 || len(result.Content.Blocks.Get(0).Text) != 100 {
		t.Fatalf("lost Anthropic blocks: %+v", result)
	}
	for index, block := range ends {
		if block != acc.output.Content.Blocks.Get(index) {
			t.Fatal("toolcall_end lost original block identity")
		}
		if string(block.Arguments) != `{"value":1}` || len(block.Extra) != 0 {
			t.Fatalf("tool finalization: %+v", block)
		}
	}
}

func TestAnthropicSSENilBody(t *testing.T) {
	err := readAnthropicSSE(context.Background(), nil, func(json.RawMessage) error { t.Fatal("unexpected event"); return nil })
	if err == nil || err.Error() != "Attempted to iterate over an Anthropic response with no body" {
		t.Fatalf("nil body: %v", err)
	}
}
