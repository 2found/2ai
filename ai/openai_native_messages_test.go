package ai

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestPiOpenAICompletionsMessagesOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-completions-messages.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          map[string]json.RawMessage
		Cases          []struct {
			Input struct {
				Name    string
				Model   map[string]json.RawMessage
				Compat  json.RawMessage
				Context Context
				Tools   []Tool
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 73 {
		t.Fatal("unexpected completions message oracle revision/coverage")
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
			if len(tc.Input.Compat) > 0 {
				model["compat"] = tc.Input.Compat
			}
			encoded, _ := json.Marshal(model)
			compat, err := ResolveOpenAICompletionsCompat(encoded)
			if err != nil {
				t.Fatal(err)
			}
			var messages, tools json.RawMessage
			properties, failure := CreateGrammarToolInputProperties(tc.Input.Tools, compat.SupportsOpenAIGrammarTools)
			if failure == nil {
				messages, failure = ConvertOpenAICompletionsMessages(encoded, NormalizeContext(tc.Input.Context), compat, properties)
			}
			if failure == nil {
				tools, failure = ConvertOpenAICompletionsTools(tc.Input.Tools, compat)
			}
			var message *string
			if failure != nil {
				text := failure.Error()
				message = &text
			}
			actual, err := json.Marshal(map[string]any{"compat": compat, "messages": messages, "tools": tools, "error": message})
			if err != nil {
				t.Fatal(err)
			}
			var got, want map[string]any
			actualDecoder := json.NewDecoder(bytes.NewReader(actual))
			actualDecoder.UseNumber()
			if err := actualDecoder.Decode(&got); err != nil {
				t.Fatal(err)
			}
			expectedDecoder := json.NewDecoder(bytes.NewReader(tc.Expected))
			expectedDecoder.UseNumber()
			if err := expectedDecoder.Decode(&want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				for _, key := range []string{"compat", "messages", "tools", "error"} {
					if !reflect.DeepEqual(got[key], want[key]) {
						g, _ := json.Marshal(got[key])
						w, _ := json.Marshal(want[key])
						t.Errorf("%s\nGo: %s\nPi: %s", key, g, w)
					}
				}
			}
			after, _ := json.Marshal(tc.Input)
			if string(before) != string(after) {
				t.Fatal("request conversion mutated transcript/tool input")
			}
		})
	}
}

func TestPiCompletionsArgumentStringification(t *testing.T) {
	raw := json.RawMessage(`{"z":"\u003c\u0026\u003e\u2028","a":1.0,"10":"ten","2":"two","nested":{"last":-0,"first":1e-7},"overflow":1e400}`)
	got, err := stringifyCompletionsJSON(raw)
	want := "{\"2\":\"two\",\"10\":\"ten\",\"z\":\"<&>\u2028\",\"a\":1,\"nested\":{\"last\":0,\"first\":1e-7},\"overflow\":null}"
	if err != nil || string(got) != want {
		t.Fatalf("argument string changed: %q want %q: %v", got, want, err)
	}
}
