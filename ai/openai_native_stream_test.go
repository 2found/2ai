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

func TestPiOpenAICompletionsStreamOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-completions-stream.json")
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
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 55 {
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
			compat, err := ResolveOpenAICompletionsCompat(encoded)
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
			acc := newCompletionsAccumulator(model, compat, grammar, stream, 100)
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
				if tc.Input.FailBefore != nil && i == *tc.Input.FailBefore {
					acc.fail("fixture failure", false)
					failed = true
					break
				}
				if err := acc.chunk(chunk); err != nil {
					acc.fail(err.Error(), false)
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

func TestCompletionsStreamConcurrentSnapshots(t *testing.T) {
	stream := NewAssistantMessageEventStream()
	acc := newCompletionsAccumulator(completionsModel{ID: "test"}, OpenAICompletionsCompat{SupportsFinishReason: true}, nil, stream, 100)
	content := acc.output.Content
	go func() {
		acc.start()
		// Grow the block array repeatedly while updating the first tool through
		// its original index, which must continue to address the same object.
		for i := 0; i < 100; i++ {
			acc.chunk(json.RawMessage(fmt.Sprintf(`{"choices":[{"delta":{"tool_calls":[{"index":%d,"id":"c%d","function":{"name":"tool","arguments":"{"}}]}}]}`, i, i)))
			acc.chunk(json.RawMessage(`{"choices":[{"delta":{"content":"x","tool_calls":[{"index":0,"function":{"arguments":" "}}]}}]}`))
		}
		acc.chunk(json.RawMessage(`{"choices":[{"finish_reason":"tool_calls","delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}`))
		acc.finish(context.Background())
	}()
	var partial *Message
	ends := map[int]*ContentBlock{}
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
				t.Fatal("changed message identity")
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
	stream.Synchronize(func() {
		if content.Blocks != acc.output.Content.Blocks || content.Blocks.Len() != 101 {
			t.Fatal("provider growth did not reach the retained content list")
		}
	})
	result, err := stream.SnapshotResult(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "toolUse" || result.Content.Blocks.Len() != 101 || len(ends) != 100 {
		t.Fatalf("lost blocks: %+v", result)
	}
	if string(result.Content.Blocks.Get(0).Arguments) != "{}" || result.Content.Blocks.Get(1).Text != string(bytes.Repeat([]byte("x"), 100)) {
		t.Fatal("updates lost after growing content array")
	}
	if ends[0] != acc.output.Content.Blocks.Get(0) {
		t.Fatal("toolcall_end did not retain block identity")
	}
	// Producer settlement makes retained pointers safe to inspect directly.
	if err := stream.WaitForEnd(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, block := range acc.output.Content.Blocks.Values() {
		if len(block.Extra) > 0 {
			t.Fatal("scratch data survived finalization")
		}
	}
}
