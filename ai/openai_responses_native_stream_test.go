package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
)

func TestPiOpenAIResponsesStreamOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-responses-stream.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          map[string]json.RawMessage
		Tool           Tool
		Cases          []struct {
			Input struct {
				Name       string
				Model      map[string]json.RawMessage
				Compat     json.RawMessage
				Options    map[string]json.RawMessage
				Chunks     []json.RawMessage
				FailBefore *int
				AbortAtEnd bool
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 98 {
		t.Fatal("unexpected stream oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			modelFields := map[string]json.RawMessage{}
			for key, value := range fixture.Model {
				modelFields[key] = value
			}
			for key, value := range tc.Input.Model {
				modelFields[key] = value
			}
			if len(tc.Input.Compat) > 0 {
				modelFields["compat"] = tc.Input.Compat
			}
			encoded, _ := json.Marshal(modelFields)
			var model completionsModel
			if err := json.Unmarshal(encoded, &model); err != nil {
				t.Fatal(err)
			}
			compat, err := ResolveOpenAIResponsesCompat(encoded)
			if err != nil {
				t.Fatal(err)
			}
			grammar, err := CreateGrammarToolInputProperties([]Tool{fixture.Tool}, compat.SupportsOpenAIGrammarTools)
			if err != nil {
				t.Fatal(err)
			}
			stream := NewAssistantMessageEventStream()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			acc := newResponsesAccumulator(model, grammar, stream, 100)
			acc.serviceTier = tc.Input.Options["serviceTier"]
			acc.applyServiceTierPricing = func(usage *Usage, tier json.RawMessage) { responsesServiceTierPricing(model.ID, usage, tier) }
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
			acc.start()
			failed := false
			for i, chunk := range tc.Input.Chunks {
				// The SDK iterator stops admitting SSE events once its request is
				// aborted. The accumulator still performs the shared EOF checks.
				if ctx.Err() != nil {
					break
				}
				if tc.Input.FailBefore != nil && i == *tc.Input.FailBefore {
					acc.fail("fixture failure", false)
					failed = true
					break
				}
				if err := acc.chunk(chunk); err != nil {
					acc.fail(err.Error(), ctx.Err() != nil)
					failed = true
					break
				}
			}
			if !failed {
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

func TestResponsesStreamConcurrentSnapshots(t *testing.T) {
	stream := NewAssistantMessageEventStream()
	acc := newResponsesAccumulator(completionsModel{ID: "test"}, nil, stream, 100)
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
		if !send(`{"type":"response.output_item.added","output_index":500,"item":{"type":"message","id":"msg_1"}}`) {
			return
		}
		for i := 0; i < 100; i++ {
			if !send(fmt.Sprintf(`{"type":"response.output_item.added","output_index":%d,"item":{"type":"function_call","id":"fc_%d","call_id":"call_%d","name":"tool","arguments":""}}`, i, i, i)) {
				return
			}
			if !send(`{"type":"response.output_text.delta","output_index":500,"delta":"x"}`) {
				return
			}
			if !send(fmt.Sprintf(`{"type":"response.output_item.done","output_index":%d,"item":{"type":"function_call","id":"fc_%d","call_id":"call_%d","name":"tool","arguments":"{\"value\":1}"}}`, i, i, i)) {
				return
			}
		}
		if !send(`{"type":"response.output_item.done","output_index":500,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"final"}]}}`) {
			return
		}
		if !send(`{"type":"response.completed","response":{"status":"completed"}}`) {
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
	if result.StopReason != "toolUse" || result.Content.Blocks.Len() != 101 || len(ends) != 100 || result.Content.Blocks.Get(0).Text != "final" {
		t.Fatalf("lost Responses blocks: %+v", result)
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
