package ai

import (
	"encoding/json"
	"math"
	"testing"
)

// A native candidate on a first-party API carries its list price, so a run's
// usage is priced (PricingSource "model") instead of reported as an unknown $0.
// A gateway serving the same model id stays unpriced.
func TestNativeCandidateCarriesTheListPrice(t *testing.T) {
	cost := func(t *testing.T, spec ClientSpec, model string) completionsCost {
		t.Helper()
		client, err := NewNativeClient(spec)
		if err != nil {
			t.Fatal(err)
		}
		candidate, err := client.Candidate(model, 128000)
		if err != nil {
			t.Fatal(err)
		}
		var metadata struct{ Cost completionsCost }
		if err := json.Unmarshal(candidate.Model, &metadata); err != nil {
			t.Fatal(err)
		}
		return metadata.Cost
	}
	deepseek := cost(t, ClientSpec{Name: "deepseek", APIKey: "k"}, "deepseek-flash")
	if !deepseek.observed || deepseek.Input != 0.30 || deepseek.Output != 1.20 || deepseek.CacheRead != 0.006 || deepseek.CacheWrite != 0.30 {
		t.Fatalf("deepseek-flash cost metadata = %+v, want observed list rates", deepseek)
	}
	usage := &Usage{Input: 1_000_000, CacheRead: 1_000_000, Output: 1_000_000}
	calculateNativeUsageCost(deepseek, usage)
	if want := 0.30 + 0.006 + 1.20; math.Abs(usage.Cost.Total-want) > 1e-9 || !usage.Observation.PricingObserved || usage.Observation.PricingSource != "model" {
		t.Fatalf("deepseek-flash usage cost = %v (%+v), want %v priced by model", usage.Cost.Total, usage.Observation, want)
	}
	if c := cost(t, ClientSpec{Name: "anthropic", APIKey: "k"}, "claude-sonnet-4-5"); !c.observed || c.Input != 3 || c.Output != 15 {
		t.Fatalf("claude-sonnet-4-5 cost metadata = %+v, want observed list rates", c)
	}
	if c := cost(t, ClientSpec{Name: "deepseek", APIKey: "k"}, "mystery-model"); c.observed {
		t.Fatalf("a model with no list price must stay unpriced: %+v", c)
	}
	if c := cost(t, ClientSpec{Name: "opencode-go", APIKey: "k"}, "deepseek-flash"); c.observed {
		t.Fatalf("a gateway serving a priced model id must stay unpriced: %+v", c)
	}
}
