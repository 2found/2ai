package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestPiCodexResponsesStreamOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-codex-stream.json")
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
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 112 {
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
			acc := newCodexResponsesAccumulator(model, grammar, stream, 100)
			acc.serviceTier = tc.Input.Options["serviceTier"]
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
				// Codex admits all frames already in the current SSE read buffer.
				if tc.Input.FailBefore != nil && i == *tc.Input.FailBefore {
					acc.fail("fixture failure", false)
					failed = true
					break
				}
				terminal, err := acc.chunk(chunk)
				if err != nil {
					acc.fail(err.Error(), ctx.Err() != nil)
					failed = true
					break
				}
				if terminal {
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
