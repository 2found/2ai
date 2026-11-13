package jscompat_test

import (
	"encoding/json"
	"os"
	"strconv"
	"testing"

	"github.com/2found/2ai/core/ai/jscompat"
)

func TestPiJSONParsingOracle(t *testing.T) {
	data, err := os.ReadFile("../testdata/pi-streaming.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct {
		JSON []struct {
			Input, Repaired     string
			Complete, Streaming json.RawMessage
			CompleteError       bool
		}
	}
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	for i, fixture := range fixtures.JSON {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			if actual := jscompat.RepairJSON(fixture.Input); actual != fixture.Repaired {
				t.Fatalf("repair %q: want %q, got %q", fixture.Input, fixture.Repaired, actual)
			}
			complete, err := jscompat.ParseJSONWithRepair(fixture.Input)
			if (err != nil) != fixture.CompleteError {
				t.Fatalf("complete %q: want error %v, got %v", fixture.Input, fixture.CompleteError, err)
			}
			if err == nil {
				assertPiJSON(t, fixture.Complete, complete)
			}
			assertPiJSON(t, fixture.Streaming, jscompat.ParseStreamingJSON(fixture.Input))
		})
	}
}

func TestPiJSONPrefixOracle(t *testing.T) {
	data, err := os.ReadFile("../testdata/pi-json-prefixes.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct {
		UpstreamCommit string
		Cases          []struct {
			Name, Input, Repaired, Streaming string
			Complete                         *string
		}
	}
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	if fixtures.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixtures.Cases) != 848 {
		t.Fatal("unexpected JSON prefix oracle coverage")
	}
	for _, fixture := range fixtures.Cases {
		t.Run(fixture.Name, func(t *testing.T) {
			if actual := jscompat.RepairJSON(fixture.Input); actual != fixture.Repaired {
				t.Errorf("repair: want %q, got %q", fixture.Repaired, actual)
			}
			complete, err := jscompat.ParseJSONWithRepair(fixture.Input)
			if (err == nil) != (fixture.Complete != nil) {
				t.Errorf("complete: want success %v, got %v", fixture.Complete != nil, err)
			} else if err == nil && string(complete) != *fixture.Complete {
				t.Errorf("complete: want %q, got %q", *fixture.Complete, complete)
			}
			if actual := jscompat.ParseStreamingJSON(fixture.Input); string(actual) != fixture.Streaming {
				t.Errorf("streaming %q: want %q, got %q", fixture.Input, fixture.Streaming, actual)
			}
		})
	}
}
