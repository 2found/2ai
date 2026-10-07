package ai

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestPiUsageJSONNumbers(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-usage-json.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Raw, Mutation string }
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 48 {
		t.Fatal("unexpected usage JSON coverage")
	}
	for i, tc := range fixture.Cases {
		t.Run(strconv.Itoa(i)+"/"+tc.Input.Mutation, func(t *testing.T) {
			var usage Usage
			if err := json.Unmarshal([]byte(tc.Input.Raw), &usage); err != nil {
				t.Fatal(err)
			}
			var original map[string]json.RawMessage
			if err := json.Unmarshal([]byte(tc.Input.Raw), &original); err != nil {
				t.Fatal(err)
			}
			paths := map[string]bool{}
			var cost map[string]json.RawMessage
			if raw := original["cost"]; len(raw) > 0 {
				if err := json.Unmarshal(raw, &cost); err != nil {
					t.Fatal(err)
				}
			}
			for _, key := range []string{"input", "output", "cacheRead", "cacheWrite", "cacheWrite1h", "reasoning", "totalTokens"} {
				if raw := original[key]; len(raw) > 0 && string(raw) != "null" {
					paths[key] = true
				}
			}
			for _, key := range []string{"input", "output", "cacheRead", "cacheWrite", "total"} {
				if raw := cost[key]; len(raw) > 0 && string(raw) != "null" {
					paths["cost."+key] = true
				}
			}
			classify := func(n float64) string {
				if math.IsNaN(n) {
					return "NaN"
				}
				if math.IsInf(n, 1) {
					return "Infinity"
				}
				if math.IsInf(n, -1) {
					return "-Infinity"
				}
				if n == 0 && math.Signbit(n) {
					return "-0"
				}
				return "finite"
			}
			numbers := func() map[string]string {
				result := map[string]string{}
				values := map[string]float64{"input": usage.Input, "output": usage.Output, "cacheRead": usage.CacheRead, "cacheWrite": usage.CacheWrite, "totalTokens": usage.TotalTokens, "cost.input": usage.Cost.Input, "cost.output": usage.Cost.Output, "cost.cacheRead": usage.Cost.CacheRead, "cost.cacheWrite": usage.Cost.CacheWrite, "cost.total": usage.Cost.Total}
				if usage.CacheWrite1h != nil {
					values["cacheWrite1h"] = *usage.CacheWrite1h
				}
				if usage.Reasoning != nil {
					values["reasoning"] = *usage.Reasoning
				}
				for key := range paths {
					result[key] = classify(values[key])
				}
				return result
			}
			beforeNumbers := numbers()
			before, err := json.Marshal(usage)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(beforeNumbers, numbers()) {
				t.Fatal("export changed live numbers")
			}
			if tc.Input.Mutation != "none" {
				n, err := strconv.ParseFloat(tc.Input.Mutation, 64)
				if err != nil {
					t.Fatal(err)
				}
				usage.Input = n
				usage.Reasoning = &n
				usage.CacheWrite1h = &n
				if cost == nil {
					if err := json.Unmarshal([]byte(`{}`), &usage.Cost); err != nil {
						t.Fatal(err)
					}
				}
				usage.Cost.Total = n
				for _, key := range []string{"input", "reasoning", "cacheWrite1h", "cost.total"} {
					paths[key] = true
				}
			}
			afterNumbers := numbers()
			after, err := json.Marshal(usage)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(afterNumbers, numbers()) {
				t.Fatal("export changed mutated live numbers")
			}
			stream := NewAssistantMessageEventStream()
			snapshot, err := stream.SnapshotEvent(AssistantMessageEvent{Type: "start", Partial: &Message{Role: "assistant", Usage: &usage}})
			if err != nil {
				t.Fatal(err)
			}
			snapshotJSON, err := json.Marshal(snapshot.Partial.Usage)
			if err != nil || !bytes.Equal(snapshotJSON, after) {
				t.Fatal("snapshot changed usage export", err)
			}
			actual, err := json.Marshal(map[string]any{"before": json.RawMessage(before), "beforeNumbers": beforeNumbers, "after": json.RawMessage(after), "afterNumbers": afterNumbers})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			decoder := json.NewDecoder(bytes.NewReader(actual))
			decoder.UseNumber()
			if err = decoder.Decode(&got); err != nil {
				t.Fatal(err)
			}
			decoder = json.NewDecoder(bytes.NewReader(tc.Expected))
			decoder.UseNumber()
			if err = decoder.Decode(&want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}

func TestUsageRejectsNonNumberJSON(t *testing.T) {
	for _, value := range []string{`"1"`, `true`, `[]`, `{}`, `"NaN"`} {
		for _, path := range []string{"input", "reasoning", "cost"} {
			payload := `{"` + path + `":` + value + `}`
			if path == "cost" {
				payload = `{"cost":{"total":` + value + `}}`
			}
			var usage Usage
			if err := json.Unmarshal([]byte(payload), &usage); err == nil {
				t.Fatalf("typed usage accepted %s", payload)
			}
		}
	}
	// Public standalone cost must use the same normalization as nested cost.
	var cost UsageCost
	if err := json.Unmarshal([]byte(`{"total":1e400,"vendor":9007199254740993}`), &cost); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(cost)
	if err != nil || !strings.Contains(string(raw), `"total":null`) || !strings.Contains(string(raw), `"vendor":9007199254740992`) || !math.IsInf(cost.Total, 1) {
		t.Fatalf("cost export %s: %v", raw, err)
	}
}
