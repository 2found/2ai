package ai

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestPiResponsesMessagesOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-responses-messages.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          map[string]json.RawMessage
		Allowed        []string
		Cases          []struct {
			Input struct {
				Name    string
				Model   map[string]json.RawMessage
				Context Context
				Tools   []Tool
				Options ResponsesMessagesOptions
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 83 {
		t.Fatal("unexpected Responses oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			before, _ := json.Marshal(tc.Input)
			model := map[string]json.RawMessage{}
			for key, value := range fixture.Model {
				model[key] = value
			}
			for key, value := range tc.Input.Model {
				model[key] = value
			}
			encoded, _ := json.Marshal(model)
			options := tc.Input.Options
			var messages, tools json.RawMessage
			properties, failure := CreateGrammarToolInputProperties(tc.Input.Tools, options.ToolOptions.SupportsOpenAIGrammarTools)
			options.GrammarToolInputProperties = properties
			if failure == nil {
				messages, failure = ConvertResponsesMessages(encoded, NormalizeContext(tc.Input.Context), fixture.Allowed, options)
			}
			if failure == nil {
				tools, failure = ConvertResponsesTools(tc.Input.Tools, options.ToolOptions)
			}
			var message *string
			if failure != nil {
				text := failure.Error()
				message = &text
			}
			actual, err := json.Marshal(map[string]any{"messages": messages, "tools": tools, "error": message})
			if err != nil {
				t.Fatal(err)
			}
			decode := func(raw []byte) any {
				t.Helper()
				var value any
				decoder := json.NewDecoder(bytes.NewReader(raw))
				decoder.UseNumber()
				if err := decoder.Decode(&value); err != nil {
					t.Fatal(err)
				}
				return value
			}
			if !reflect.DeepEqual(decode(actual), decode(tc.Expected)) {
				t.Errorf("Go: %s\nPi: %s", actual, tc.Expected)
			}
			after, _ := json.Marshal(tc.Input)
			if string(before) != string(after) {
				t.Fatal("conversion mutated transcript/tool input")
			}
		})
	}
}
