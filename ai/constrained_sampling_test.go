package ai

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"
)

func TestPiConstrainedSamplingOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-constrained-sampling.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name, Mode, RejectKey, Property string
				Schema, Args                    json.RawMessage
				Tool                            Tool
				Supported                       bool
				Steps                           []struct {
					Input string
					Close bool
				}
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 66 {
		t.Fatal("unexpected constrained sampling oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			before, _ := json.Marshal(tc.Input)
			var result any
			var failure error
			var check UnsupportedStrictSchemaKeywordCheck
			if tc.Input.RejectKey != "" {
				check = func(key string, _ json.RawMessage) bool { return key == tc.Input.RejectKey }
			}
			switch tc.Input.Mode {
			case "strict":
				var strict *bool
				strict, failure = ResolveJSONSchemaStrictSampling(tc.Input.Tool, tc.Input.Supported, check)
				if failure == nil {
					var parameters json.RawMessage
					parameters, failure = GetJSONSchemaToolParameters(tc.Input.Tool, strict)
					if failure == nil {
						result = map[string]any{"strict": strict, "parameters": parameters}
					}
				}
			case "grammar":
				var grammar *GrammarConstrainedSampling
				grammar, failure = ResolveGrammarConstrainedSampling(tc.Input.Tool, tc.Input.Supported)
				if failure == nil {
					var properties map[string]string
					properties, failure = CreateGrammarToolInputProperties([]Tool{tc.Input.Tool}, tc.Input.Supported)
					if failure == nil {
						result = map[string]any{"grammar": grammar, "properties": properties}
					}
				}
			case "input":
				var input string
				input, failure = GetGrammarToolInput("grammar", tc.Input.Args, tc.Input.Property)
				if failure == nil {
					result = input
				}
			case "delta":
				buffer := GrammarToolInputJSONBuffer{}
				steps := []map[string]any{}
				for _, step := range tc.Input.Steps {
					delta, err := AppendGrammarToolInputJSONDelta(&buffer, tc.Input.Property, step.Input, step.Close)
					var failure *string
					if err != nil {
						message := err.Error()
						failure = &message
					}
					steps = append(steps, map[string]any{"delta": delta, "error": failure, "buffer": buffer})
				}
				result = steps
			default:
				var schema, second json.RawMessage
				schema, failure = MakeStrictJSONSchema(tc.Input.Schema, check)
				if failure == nil {
					second, failure = MakeStrictJSONSchema(schema, check)
					if failure == nil {
						result = map[string]any{"schema": schema, "second": second}
					}
				}
			}
			var message *string
			if failure != nil {
				text := failure.Error()
				message = &text
			}
			actual, err := json.Marshal(map[string]any{"result": result, "error": message})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			_ = json.Unmarshal(actual, &got)
			_ = json.Unmarshal(tc.Expected, &want)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("constrained sampling differs\nGo: %s\nPi: %s", actual, tc.Expected)
			}
			after, _ := json.Marshal(tc.Input)
			if string(before) != string(after) {
				t.Fatal("conversion mutated caller input")
			}
		})
	}
}

func TestPiStrictSamplingPreservesErrorKindAndDoesNotMaskCallbackPanics(t *testing.T) {
	_, err := MakeStrictJSONSchema(json.RawMessage(`{"type":"object","oneOf":[]}`), nil)
	var unsupported *UnsupportedStrictJSONSchemaError
	if !errors.As(err, &unsupported) {
		t.Fatalf("lost strict-schema error type: %v", err)
	}
	failure := errors.New("provider check failed")
	defer func() {
		if value := recover(); value != failure {
			t.Fatalf("provider check failure masked: %v", value)
		}
	}()
	_, _ = ResolveJSONSchemaStrictSampling(Tool{Name: "tool", Parameters: json.RawMessage(`{"type":"object"}`), ConstrainedSampling: json.RawMessage(`{"type":"json_schema","strict":"prefer"}`)}, true, func(string, json.RawMessage) bool { panic(failure) })
}

func TestPiGrammarDeltasReconstructArguments(t *testing.T) {
	buffer := GrammarToolInputJSONBuffer{}
	assembled := ""
	for _, step := range []struct {
		input string
		close bool
	}{{"", false}, {"a", false}, {"a\n\"<>&\u2028\u2029😀", false}, {"a\n\"<>&\u2028\u2029😀", true}} {
		delta, err := AppendGrammarToolInputJSONDelta(&buffer, "code", step.input, step.close)
		if err != nil {
			t.Fatal(err)
		}
		if delta != nil {
			assembled += *delta
		}
	}
	value, err := GetGrammarToolInput("grammar", json.RawMessage(assembled), "code")
	if err != nil || value != buffer.Input {
		t.Fatalf("invalid reconstructed arguments: %q %v", assembled, err)
	}
}
