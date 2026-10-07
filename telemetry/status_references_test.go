package telemetry_test

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/2found/2ai/telemetry"
)

func TestPiStatusReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-status-references.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Kind, Mode, Phase string }
			Raw      string
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 48 {
		t.Fatal("unexpected status-reference coverage")
	}
	for i, tc := range fixture.Cases {
		t.Run(strconv.Itoa(i)+"/"+tc.Input.Kind+"/"+tc.Input.Mode+"/"+tc.Input.Phase, func(t *testing.T) {
			details := new(telemetry.ErrorDetails)
			if err := json.Unmarshal([]byte(tc.Raw), details); err != nil {
				t.Fatal(err)
			}
			if strings.HasPrefix(tc.Input.Kind, "shared") {
				details.SetMessageValue(details.NameValue())
			}
			recorder := telemetry.NewInMemory()
			var retained telemetry.SpanStatus
			var before, active map[string]any
			encode := func(value any) string {
				raw, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				return string(raw)
			}
			inspect := func(status telemetry.SpanStatus) map[string]any {
				return map[string]any{"wire": encode(status), "nameInput": status.Error.NameValue() == details.NameValue(), "messageInput": status.Error.MessageValue() == details.MessageValue(), "same": status.Error.NameValue() == status.Error.MessageValue()}
			}
			mutate := func(value any) {
				switch value := value.(type) {
				case *telemetry.Array:
					value.Append(float64(3))
				case *telemetry.Object:
					value.Set("late", float64(3))
					value.Delete("z")
					value.Set("z", float64(4))
				default:
					t.Fatalf("unexpected field: %T", value)
				}
			}
			action := func() {
				switch tc.Input.Phase {
				case "input-nested":
					mutate(details.NameValue())
				case "snapshot-nested":
					mutate(retained.Error.NameValue())
				case "input-replace":
					details.SetNameValue("")
					details.SetMessageValue(nil)
				case "snapshot-replace":
					retained.Error.SetNameValue("")
					retained.Error.SetMessageValue(telemetry.Undefined)
				}
			}
			failure := recorder.StartSpan(telemetry.SpanOptions{Name: "reference"}, func(span *telemetry.Span) error {
				if tc.Input.Mode == "automatic" {
					return details
				}
				span.SetStatus(telemetry.SpanStatus{Status: "error", Error: details})
				retained = recorder.GetSpans()[0].Status
				before = inspect(retained)
				action()
				active = inspect(recorder.GetSpans()[0].Status)
				return nil
			})
			if tc.Input.Mode == "automatic" {
				retained = recorder.GetSpans()[0].Status
				before = inspect(retained)
				action()
				active = inspect(recorder.GetSpans()[0].Status)
			}
			if tc.Input.Phase == "settled-nested" {
				mutate(details.NameValue())
			}
			failurePreserved := failure == nil
			if tc.Input.Mode == "automatic" {
				failurePreserved = failure == details
			}
			actual, err := json.Marshal(map[string]any{"before": before, "active": active, "retained": inspect(retained), "settled": inspect(recorder.GetSpans()[0].Status), "input": encode(details), "failurePreserved": failurePreserved})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err = json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}

func TestErrorFieldPresenceNumbersAndCycles(t *testing.T) {
	var details telemetry.ErrorDetails
	if err := json.Unmarshal([]byte(`{"name":1e400,"message":-0}`), &details); err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(details.NameValue().(float64), 1) || !math.Signbit(details.MessageValue().(float64)) {
		t.Fatal("decoded error fields lost live numeric values")
	}
	copy := details
	details.SetNameValue("")
	details.SetMessageValue(telemetry.Undefined)
	raw, err := json.Marshal(details)
	if err != nil || string(raw) != `{"name":""}` {
		t.Fatalf("explicit replacement: %s %v", raw, err)
	}
	raw, err = json.Marshal(copy)
	if err != nil || string(raw) != `{"name":null,"message":0}` {
		t.Fatalf("replacement reached copied fields: %s %v", raw, err)
	}
	if err := json.Unmarshal([]byte(`{}`), &details); err != nil {
		t.Fatal(err)
	}
	if details.NameValue() != telemetry.Undefined || details.MessageValue() != telemetry.Undefined {
		t.Fatal("absent fields were invented")
	}
	details.SetNameValue(nil)
	details.SetMessageValue("")
	raw, err = json.Marshal(details)
	if err != nil || string(raw) != `{"name":null,"message":""}` {
		t.Fatalf("null/empty replacement: %s %v", raw, err)
	}
	object := telemetry.NewObject()
	object.Set("self", object)
	details.SetNameValue(object)
	details.SetMessageValue(object)
	recorder := telemetry.NewInMemory()
	if err := recorder.StartSpan(telemetry.SpanOptions{}, func(span *telemetry.Span) error {
		span.SetStatus(telemetry.SpanStatus{Status: "error", Error: &details})
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := json.Marshal(recorder.GetSpans()); err == nil {
		t.Fatal("cyclic error field exported")
	}
	object.Delete("self")
	object.Set("value", 1)
	raw, err = json.Marshal(recorder.GetSpans()[0].Status)
	if err != nil || string(raw) != `{"status":"error","error":{"name":{"value":1},"message":{"value":1}}}` {
		t.Fatalf("cycle edit lost shared identity: %s %v", raw, err)
	}
	calls := 0
	details.SetNameValue(func() { calls++ })
	details.SetMessageValue(attributeString("passive"))
	raw, err = json.Marshal(details)
	if err != nil || string(raw) != `{"message":"passive"}` || calls != 0 {
		t.Fatalf("non-passive error export: %s %v", raw, err)
	}
	details.SetNameValue(make(chan int))
	if _, err := json.Marshal(details); err == nil {
		t.Fatal("unsupported Go error field exported")
	}
}
