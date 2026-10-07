package ai

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

func diagnosticValue(t *testing.T, raw []byte) any {
	t.Helper()
	value, err := jsonjs.DecodeJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func TestPiJSONRepairDiagnostics(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-json-repair-errors.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct{ Input, Value, Error json.RawMessage }
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 2064 {
		t.Fatal("unexpected repair diagnostic coverage")
	}
	for i, tc := range fixture.Cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			input := diagnosticValue(t, tc.Input).(string)
			value, err := ParseJSONWithRepair(input)
			var actualError any
			if err != nil {
				actualError = err.Error()
			}
			if expected := diagnosticValue(t, tc.Error); actualError != expected {
				t.Fatalf("input %q: Go error %q, Pi %q", input, actualError, expected)
			}
			if err == nil && string(value) != diagnosticValue(t, tc.Value).(string) {
				t.Fatalf("input %q: Go %s, Pi %s", input, value, tc.Value)
			}
		})
	}
}

func TestPiAnthropicJSONDiagnostics(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-anthropic-json-errors.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          completionsModel
		Cases          []struct {
			Name, Body string
			Expected   json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 204 {
		t.Fatal("unexpected Anthropic diagnostic coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			stream := NewAssistantMessageEventStream()
			acc := newAnthropicAccumulator(fixture.Model, false, nil, nil, stream, func() int64 { return 100 })
			events := []json.RawMessage{}
			acc.push = func(event AssistantMessageEvent) {
				raw, err := json.Marshal(event)
				if err != nil {
					t.Fatal(err)
				}
				events = append(events, raw)
				stream.Push(event)
			}
			acc.start()
			ctx := context.Background()
			if err := readAnthropicSSE(ctx, strings.NewReader(tc.Body), acc.chunk); err != nil {
				acc.fail(err.Error(), false)
			} else {
				acc.finish(ctx)
			}
			result, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"events": events, "result": result})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(diagnosticValue(t, actual), diagnosticValue(t, tc.Expected)) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
		})
	}
}
