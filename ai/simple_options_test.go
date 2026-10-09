package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestPiSimpleOptionsOracle(t *testing.T) {
	t.Setenv("PI_CACHE_RETENTION", "")
	updates := readPiReconUpdates(t)
	raw, err := os.ReadFile("testdata/pi-simple-options.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          map[string]json.RawMessage
		Cases          []struct {
			Input struct {
				Name           string
				Model, Options map[string]json.RawMessage
				Context        Context
			}
			Expected json.RawMessage
		}
		Budgets []struct {
			Input struct {
				Level   string
				Base    *float64
				Ceiling float64
				Custom  map[string]json.RawMessage
			}
			Expected ThinkingTokenBudget
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 92 || len(fixture.Budgets) != 72 {
		t.Fatalf("unexpected oracle coverage: %d/%d", len(fixture.Cases), len(fixture.Budgets))
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
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			expected := updates.simpleExpectation(t, "pi-simple-options", tc.Input.Name, tc.Expected)
			model := map[string]json.RawMessage{}
			for key, value := range fixture.Model {
				model[key] = value
			}
			for key, value := range tc.Input.Model {
				model[key] = value
			}
			rawModel, _ := json.Marshal(model)
			options := map[string]json.RawMessage{"apiKey": json.RawMessage(`"fixture-key"`)}
			for key, value := range tc.Input.Options {
				options[key] = value
			}
			rawOptions, _ := json.Marshal(options)
			transcript := NormalizeContext(tc.Input.Context)
			before, _ := json.Marshal(transcript)
			prepared, err := BuildOpenAICompletionsSimpleOptions(rawModel, transcript, rawOptions)
			if err != nil {
				t.Fatal(err)
			}
			compat, err := ResolveOpenAICompletionsCompat(rawModel)
			if err != nil {
				t.Fatal(err)
			}
			params, err := BuildOpenAICompletionsParams(rawModel, ResolveTranscript(transcript, compat.SupportsMidConvoSystemMessages), prepared)
			if err != nil {
				t.Fatal(err)
			}
			levels, err := GetSupportedThinkingLevels(rawModel)
			if err != nil {
				t.Fatal(err)
			}
			messageTokens := []float64{}
			for _, message := range transcript.Messages() {
				messageTokens = append(messageTokens, EstimateMessageTokens(message))
			}
			actual, err := json.Marshal(map[string]any{"options": prepared, "params": params, "estimate": EstimateContextTokens(transcript.Messages()), "messageTokens": messageTokens, "levels": levels})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(decode(actual), decode(expected)) {
				t.Fatalf("simple options mismatch\nGo: %s\nPi: %s", actual, expected)
			}
			after, _ := json.Marshal(transcript)
			if !bytes.Equal(before, after) {
				t.Fatal("simple options mutated transcript")
			}
		})
	}
	for _, tc := range fixture.Budgets {
		if got := AdjustMaxTokensForThinking(tc.Input.Base, tc.Input.Ceiling, tc.Input.Level, tc.Input.Custom); got != tc.Expected {
			t.Fatalf("budget %+v: %+v want %+v", tc.Input, got, tc.Expected)
		}
	}
}

func TestCompletionsSimpleChecksCredentialsBeforeAdmission(t *testing.T) {
	called := false
	stream, err := StreamOpenAICompletionsSimple(context.Background(), json.RawMessage(`{"provider":"openai","id":"test","api":"openai-completions"}`), NormalizeContext(Context{}), OpenAICompletionsStreamOptions{OnPayload: func(context.Context, json.RawMessage, json.RawMessage) (json.RawMessage, error) {
		called = true
		return nil, nil
	}})
	if stream != nil || err == nil || err.Error() != "No API key for provider: openai" || called {
		t.Fatalf("credential admission: %v %v %v", stream, err, called)
	}
}
