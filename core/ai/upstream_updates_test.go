package ai

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"
)

type piReconUpdates struct {
	UpstreamCommit string
	SourceHashes   map[string]string
	Estimates      []struct {
		Name          string
		Estimate      ContextUsageEstimate
		MessageTokens []float64
		MaxTokens     float64
	}
	Headers []struct {
		Name      string
		Websocket bool
		Expected  map[string]string
	}
	Texts []struct {
		Text   string
		Tokens float64
	}
	Caps []struct {
		Fixture         string
		Name            string
		MaxTokens       float64
		MaxOutputTokens *float64
	}
}

func readPiReconUpdates(t *testing.T) piReconUpdates {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-recon-updates.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture piReconUpdates
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "6fb2e7815167e6b19006fc526d1a5d0f5f998787" || len(fixture.SourceHashes) != 5 || len(fixture.Estimates) != 15 || len(fixture.Headers) != 18 || len(fixture.Texts) != 4 || len(fixture.Caps) != 62 {
		t.Fatal("unexpected recon-update provenance/coverage")
	}
	return fixture
}

// Update only estimator-owned fields. The rest of each historical transport
// oracle stays unchanged, including opaque numbers and request/error shapes.
func (updates piReconUpdates) simpleExpectation(t *testing.T, fixture, name string, original json.RawMessage) json.RawMessage {
	t.Helper()
	var value map[string]json.RawMessage
	if err := json.Unmarshal(original, &value); err != nil {
		t.Fatal(err)
	}
	for _, cap := range updates.Caps {
		if cap.Fixture != fixture || cap.Name != name {
			continue
		}
		for _, container := range []string{"options", "params"} {
			raw := value[container]
			if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
				continue
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &fields); err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"maxTokens", "max_tokens", "max_completion_tokens"} {
				if _, exists := fields[key]; exists {
					fields[key], _ = json.Marshal(cap.MaxTokens)
				}
			}
			if _, exists := fields["max_output_tokens"]; exists && cap.MaxOutputTokens != nil {
				fields["max_output_tokens"], _ = json.Marshal(*cap.MaxOutputTokens)
			}
			value[container], _ = json.Marshal(fields)
		}
	}
	if fixture == "pi-simple-options" {
		for _, estimate := range updates.Estimates {
			if estimate.Name == name {
				value["estimate"], _ = json.Marshal(estimate.Estimate)
				value["messageTokens"], _ = json.Marshal(estimate.MessageTokens)
			}
		}
	}
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestPiConservativeTextEstimates(t *testing.T) {
	for _, tc := range readPiReconUpdates(t).Texts {
		if got := EstimateTextTokens(tc.Text); got != tc.Tokens {
			t.Fatalf("estimate %q = %v, want %v", tc.Text, got, tc.Tokens)
		}
	}
}
