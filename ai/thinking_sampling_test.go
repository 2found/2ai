package ai

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestThinkingSamplingUpstreamOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-thinking-sampling.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Name, Level              string
			Model, Request, Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "200387122ca450d6387f033949423114a270b96c" || len(fixture.Cases) != 14 {
		t.Fatal("unexpected thinking-level sampling oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			model, request := bytes.Clone(tc.Model), bytes.Clone(tc.Request)
			got, err := ResolveSamplingParams(tc.Model, tc.Level, tc.Request)
			if err != nil {
				t.Fatal(err)
			}
			if (len(got) == 0) != (len(tc.Expected) == 0) {
				t.Fatalf("sampling presence differs: got %s, want %s", got, tc.Expected)
			}
			var actual, expected any
			if len(got) > 0 {
				if err := json.Unmarshal(got, &actual); err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(tc.Expected, &expected); err != nil {
					t.Fatal(err)
				}
			}
			if !reflect.DeepEqual(actual, expected) {
				t.Fatalf("Go: %s\nPi: %s", got, tc.Expected)
			}
			if !bytes.Equal(model, tc.Model) || !bytes.Equal(request, tc.Request) {
				t.Fatal("sampling resolution mutated inputs")
			}
		})
	}
}

func TestThinkingSamplingReachesCompatibleProviders(t *testing.T) {
	for _, provider := range []struct {
		api     string
		build   func(json.RawMessage, TranscriptContext, json.RawMessage) (json.RawMessage, error)
		prepare func(json.RawMessage, TranscriptContext, json.RawMessage) (json.RawMessage, error)
	}{
		{"openai-completions", BuildOpenAICompletionsParams, BuildOpenAICompletionsSimpleOptions},
		{"openai-responses", BuildOpenAIResponsesParams, BuildOpenAIResponsesSimpleOptions},
		{"azure-openai-responses", BuildAzureResponsesParams, BuildAzureResponsesSimpleOptions},
	} {
		t.Run(provider.api, func(t *testing.T) {
			model := json.RawMessage(`{"id":"custom","api":"` + provider.api + `","provider":"custom","baseUrl":"http://localhost/v1","reasoning":true,"contextWindow":128000,"maxTokens":16384,"thinkingLevelMap":{"low":null,"medium":null},"samplingParams":{"temperature":1,"top_p":0.95},"samplingParamsByThinkingLevel":{"off":{"temperature":0.7},"high":{"temperature":0.8,"top_k":64}}}`)
			transcript := NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent("hello")}}})
			for _, simple := range []bool{false, true} {
				options := json.RawMessage(`{"reasoningEffort":"low","reasoning":"low","temperature":0,"samplingParams":{"top_p":0.5}}`)
				if simple {
					var err error
					options, err = provider.prepare(model, transcript, options)
					if err != nil {
						t.Fatal(err)
					}
				}
				payload, err := provider.build(model, transcript, options)
				if err != nil {
					t.Fatal(err)
				}
				var params map[string]any
				if err := json.Unmarshal(payload, &params); err != nil {
					t.Fatal(err)
				}
				if params["temperature"] != 0.8 || params["top_p"] != 0.5 || params["top_k"] != float64(64) {
					t.Fatalf("simple=%v: sampling precedence lost: %s", simple, payload)
				}
			}
			if provider.api == "openai-completions" {
				return
			}
			model = json.RawMessage(`{"id":"custom","api":"` + provider.api + `","provider":"custom","reasoning":true,"samplingParamsByThinkingLevel":{"off":{"temperature":0.7},"medium":{"temperature":0.8}}}`)
			for _, tc := range []struct {
				options     string
				temperature float64
			}{
				{`{"reasoningSummary":"auto"}`, 0.8},
				{`{"reasoningEffort":null,"reasoningSummary":"auto"}`, 0.8},
				{`{"reasoningEffort":"","reasoningSummary":"auto"}`, 0.7},
			} {
				payload, err := provider.build(model, transcript, json.RawMessage(tc.options))
				if err != nil {
					t.Fatal(err)
				}
				var params map[string]any
				if err := json.Unmarshal(payload, &params); err != nil {
					t.Fatal(err)
				}
				if params["temperature"] != tc.temperature {
					t.Fatalf("summary-only sampling: %s", payload)
				}
			}
		})
	}
}

func TestThinkingSamplingStaysOutOfOtherProviderBodies(t *testing.T) {
	transcript := NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent("hello")}}})
	for _, provider := range []struct {
		name    string
		prepare func(json.RawMessage, TranscriptContext, json.RawMessage) (json.RawMessage, error)
		build   func(json.RawMessage, TranscriptContext, json.RawMessage) (json.RawMessage, error)
	}{
		{"anthropic-messages", BuildAnthropicSimpleOptions, func(m json.RawMessage, c TranscriptContext, o json.RawMessage) (json.RawMessage, error) {
			return BuildAnthropicParams(m, c, false, o)
		}},
		{"openai-codex-responses", BuildCodexResponsesSimpleOptions, func(m json.RawMessage, c TranscriptContext, o json.RawMessage) (json.RawMessage, error) {
			return BuildCodexResponsesParams(m, c, o, nil)
		}},
	} {
		t.Run(provider.name, func(t *testing.T) {
			model := json.RawMessage(`{"id":"test","api":"` + provider.name + `","provider":"test","maxTokens":4096,"contextWindow":128000,"samplingParams":{"top_p":0.9},"samplingParamsByThinkingLevel":{"off":{"top_k":64}}}`)
			options, err := provider.prepare(model, transcript, json.RawMessage(`{}`))
			if err != nil {
				t.Fatal(err)
			}
			payload, err := provider.build(model, transcript, options)
			if err != nil {
				t.Fatal(err)
			}
			var params map[string]json.RawMessage
			if err := json.Unmarshal(payload, &params); err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"top_p", "top_k", "samplingParams"} {
				if _, exists := params[key]; exists {
					t.Fatalf("unsupported %s leaked into %s", key, payload)
				}
			}
		})
	}
}
