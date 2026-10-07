package ai

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

func TestPiAzureResponsesParams(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-azure-responses-params.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, Runtime string
		Model                   map[string]json.RawMessage
		Context                 Context
		Cases                   []struct {
			Input struct {
				Name            string
				Model           map[string]json.RawMessage
				Options, Compat json.RawMessage
				Context         *Context
				ProcessEnv      map[string]string
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || fixture.Runtime != "1.3.14" || len(fixture.Cases) != 289 {
		t.Fatalf("unexpected Azure oracle: %d cases", len(fixture.Cases))
	}
	decode := func(raw []byte) any {
		value, err := jsonjs.DecodeJSON(raw)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			for _, key := range []string{"AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_RESOURCE_NAME", "AZURE_OPENAI_DEPLOYMENT_NAME_MAP"} {
				t.Setenv(key, tc.Input.ProcessEnv[key])
			}
			fields := map[string]json.RawMessage{}
			for key, value := range fixture.Model {
				fields[key] = value
			}
			for key, value := range tc.Input.Model {
				fields[key] = value
			}
			if len(tc.Input.Compat) > 0 {
				fields["compat"] = tc.Input.Compat
			}
			model, err := json.Marshal(fields)
			if err != nil {
				t.Fatal(err)
			}
			current := fixture.Context
			if tc.Input.Context != nil {
				current = *tc.Input.Context
			}
			transcript := NormalizeContext(current)
			before, err := json.Marshal(transcript)
			if err != nil {
				t.Fatal(err)
			}
			config, configErr := ResolveAzureResponsesConfig(model, tc.Input.Options)
			params, paramsErr := BuildAzureResponsesParams(model, transcript, tc.Input.Options)
			actual := map[string]any{"config": config, "configError": nil, "params": params, "paramsError": nil}
			if configErr != nil {
				actual["config"] = nil
				actual["configError"] = configErr.Error()
			}
			if paramsErr != nil {
				actual["paramsError"] = paramsErr.Error()
			}
			encoded, err := json.Marshal(actual)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(decode(encoded), decode(tc.Expected)) {
				t.Fatalf("Go: %s\nPi: %s", encoded, tc.Expected)
			}
			after, err := json.Marshal(transcript)
			if err != nil || string(before) != string(after) {
				t.Fatal("builder mutated transcript", err)
			}
		})
	}
}

func TestPiOpenAIPromptCache(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-azure-responses-params.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		CacheCases []struct {
			Input, Value json.RawMessage
			Error        *string
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.CacheCases) != 34 {
		t.Fatal("unexpected cache coverage")
	}
	for i, tc := range fixture.CacheCases {
		actual, err := ClampOpenAIPromptCacheKey(tc.Input)
		if tc.Error != nil {
			if err == nil || err.Error() != *tc.Error {
				t.Fatalf("case %d: Go %v; Pi %s", i, err, *tc.Error)
			}
		} else {
			if err != nil {
				t.Fatalf("case %d: %v", i, err)
			}
			if len(actual) == 0 && len(tc.Value) == 0 {
				continue
			}
			got, err := jsonjs.DecodeJSON(actual)
			if err != nil {
				t.Fatal(err)
			}
			want, err := jsonjs.DecodeJSON(tc.Value)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("case %d: Go %s; Pi %s", i, actual, tc.Value)
			}
		}
	}
}
