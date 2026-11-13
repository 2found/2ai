package telemetry

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

func TestSafeFrozenSchemaFixtureAndLiveRecords(t *testing.T) {
	raw, err := os.ReadFile("testdata/safe-c4-v1.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	compiler.UseLoader(nil)
	const resource = "https://2ai.invalid/safe-c4-v1.schema.json"
	if err := compiler.AddResource(resource, doc); err != nil {
		t.Fatal(err)
	}
	schema, err := compiler.Compile(resource)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = os.ReadFile("testdata/safe-c4-v1.example.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Records []SafeRecord
	}
	// Explicit wire field names keep the fixture independent of Go naming.
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(object["records"], &fixture.Records); err != nil {
		t.Fatal(err)
	}
	var expected struct {
		Attempts      int        `json:"attempts"`
		UsageObserved int        `json:"usage_observed"`
		Priced        int        `json:"priced"`
		Unpriced      int        `json:"unpriced"`
		MissingUsage  int        `json:"missing_usage"`
		Tokens        SafeTokens `json:"tokens"`
		Cost          float64    `json:"known_cost_usd"`
	}
	if err := json.Unmarshal(object["expected"], &expected); err != nil {
		t.Fatal(err)
	}
	attempts, observed, priced, unpriced, missing := 0, 0, 0, 0, 0
	totals := SafeTokens{}
	cost := 0.0
	validate := func(r SafeRecord) {
		t.Helper()
		data := mustJSON(t, r)
		if len(data) > SafeRecordLimit {
			t.Fatal("fixture exceeds record cap")
		}
		value, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		if err := schema.Validate(value); err != nil {
			t.Fatal(err)
		}
	}
	for _, r := range fixture.Records {
		validate(r)
		if r.Kind != "attempt_settled" {
			continue
		}
		attempts++
		a := r.Accounting
		if !a.UsageObserved {
			missing++
			continue
		}
		observed++
		totals.Input += a.Tokens.Input
		totals.Output += a.Tokens.Output
		totals.CacheRead += a.Tokens.CacheRead
		totals.CacheWrite += a.Tokens.CacheWrite
		if a.PriceState == "priced" {
			priced++
			cost += *a.KnownCostUSD
		} else {
			unpriced++
		}
	}
	if attempts != expected.Attempts || observed != expected.UsageObserved || priced != expected.Priced || unpriced != expected.Unpriced || missing != expected.MissingUsage || totals != expected.Tokens || math.Abs(cost-expected.Cost) > 1e-12 {
		t.Fatal("frozen accounting oracle changed")
	}
	o, ctx := safeFixture(t, SafeObserverOptions{})
	_ = SafeContext(Context{}, ctx).StartSpan(SpanOptions{Name: "agentray.agent.run"}, func(span *Span) error {
		NewSafeCall(WithContext(ctx, span.Context())).StartAttempt(1, SafeLabels{}).Finish("completed", "none", SafeAccounting{})
		ReconcileSafe(ctx, SafeAccounting{})
		return nil
	})
	for _, r := range o.Drain(256) {
		validate(r)
	}
	// The recursive allowlist rejects a private field even under accounting.
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(mustJSON(t, fixture.Records[0])))
	if err != nil {
		t.Fatal(err)
	}
	m := value.(map[string]any)
	m["raw_error"] = "private-canary"
	if schema.Validate(value) == nil {
		t.Fatal("schema allowed private top-level field")
	}
	delete(m, "raw_error")
	m["accounting"].(map[string]any)["prompt"] = "private-canary"
	if schema.Validate(value) == nil {
		t.Fatal("schema allowed nested private content")
	}
}
