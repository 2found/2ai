package ai

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestPiAnthropicParamsOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-anthropic-params.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          map[string]json.RawMessage
		Context        Context
		Cases          []struct {
			Input struct {
				Name, ProcessCache string
				OAuth              bool
				Model              map[string]json.RawMessage
				Compat, Options    json.RawMessage
				Context            *Context
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 156 {
		t.Fatal("unexpected request-body oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			t.Setenv("PI_CACHE_RETENTION", tc.Input.ProcessCache)
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
			current := fixture.Context
			if tc.Input.Context != nil {
				current = *tc.Input.Context
			}
			before, _ := json.Marshal(current)
			compat, err := ResolveAnthropicCompat(encoded)
			if err != nil {
				t.Fatal(err)
			}
			params, err := BuildAnthropicParams(encoded, ResolveTranscript(NormalizeContext(current), compat.SupportsMidConvoSystemMessages), tc.Input.OAuth, tc.Input.Options)
			var failure *string
			if err != nil {
				message := err.Error()
				failure = &message
			}
			actual, err := json.Marshal(map[string]any{"compat": compat, "params": params, "error": failure})
			if err != nil {
				t.Fatal(err)
			}
			var got, want map[string]any
			gd := json.NewDecoder(bytes.NewReader(actual))
			gd.UseNumber()
			if err := gd.Decode(&got); err != nil {
				t.Fatal(err)
			}
			wd := json.NewDecoder(bytes.NewReader(tc.Expected))
			wd.UseNumber()
			if err := wd.Decode(&want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("request body mismatch\nGo: %s\nPi: %s", actual, tc.Expected)
			}
			after, _ := json.Marshal(current)
			if string(before) != string(after) {
				t.Fatal("request builder mutated transcript")
			}
		})
	}
}
