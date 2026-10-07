package telemetry_test

import (
	"encoding/json"
	"os"
	"reflect"
	"strconv"
	"testing"

	"github.com/2found/2ai/telemetry"
)

func TestPiAttributeOrder(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-attribute-order.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Raw, Phase string
			Expected   map[string]string
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 64 {
		t.Fatal("unexpected order coverage")
	}
	for i, tc := range fixture.Cases {
		for _, native := range []bool{false, true} {
			t.Run(strconv.Itoa(i)+"/"+tc.Phase+"/native="+strconv.FormatBool(native), func(t *testing.T) {
				var attrs telemetry.Attributes
				if err := json.Unmarshal([]byte(tc.Raw), &attrs); err != nil {
					t.Fatal(err)
				}
				if native {
					attrs = telemetry.NewAttributes(attrs.Entries()...)
				}
				recorder := telemetry.NewInMemory()
				encode := func(value any) string {
					raw, err := json.Marshal(value)
					if err != nil {
						t.Fatal(err)
					}
					return string(raw)
				}
				actual := map[string]string{}
				var retained []telemetry.RecordedSpan
				if err := recorder.StartSpan(telemetry.SpanOptions{Name: "order", Attributes: attrs}, func(span *telemetry.Span) error {
					span.AddEvent("initial", attrs)
					retained = recorder.GetSpans()
					actual["before"] = encode(retained)
					patch := telemetry.NewAttributes(telemetry.Property{Name: "y", Value: 1}, telemetry.Property{Name: "b", Value: 2}, telemetry.Property{Name: "a", Value: 3})
					switch tc.Phase {
					case "merge":
						span.SetAttributes(patch)
					case "undefined":
						span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "y", Value: telemetry.Undefined}, telemetry.Property{Name: "b", Value: 2}, telemetry.Property{Name: "z", Value: telemetry.Undefined}, telemetry.Property{Name: "a", Value: 3}))
					case "input", "reinsert":
						if entries := attrs.Entries(); len(entries) > 0 {
							p := entries[0]
							attrs.Delete(p.Name)
							attrs.Set(p.Name, p.Value)
						}
						attrs.Set("late", 9)
						if tc.Phase == "reinsert" {
							span.SetAttributes(attrs)
						}
					case "snapshot":
						values := &retained[0].Attributes
						if entries := values.Entries(); len(entries) > 0 {
							p := entries[0]
							values.Delete(p.Name)
							values.Set(p.Name, p.Value)
						}
						values.Set("late", 9)
					case "nested":
						value := attrs.Get("value")
						if array, ok := value.(*telemetry.Array); ok {
							value = array.Get(0)
						}
						if object, ok := value.(*telemetry.Object); ok {
							p := object.Entries()[0]
							object.Delete(p.Name)
							object.Set(p.Name, p.Value)
							object.Set("late", 9)
						}
					case "reader":
						span.SetAttributesFrom(func() telemetry.Attributes {
							span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "inner", Value: 1}))
							return telemetry.NewAttributes(telemetry.Property{Name: "outer", Value: patch})
						})
					}
					actual["after"] = encode(recorder.GetSpans())
					return nil
				}); err != nil {
					t.Fatal(err)
				}
				actual["retained"], actual["settled"] = encode(retained), encode(recorder.GetSpans())
				if !reflect.DeepEqual(actual, tc.Expected) {
					for phase, want := range tc.Expected {
						if actual[phase] != want {
							t.Errorf("%s\nGo: %s\nPi: %s", phase, actual[phase], want)
						}
					}
				}
			})
		}
	}
}

func TestOrderedObjectAliasesCyclesAndPassiveExport(t *testing.T) {
	object := telemetry.NewObject(telemetry.Property{Name: "z", Value: 1}, telemetry.Property{Name: "a", Value: 2})
	attrs := telemetry.NewAttributes(telemetry.Property{Name: "object", Value: object})
	alias := attrs
	attrs.Set("late", 3)
	if alias.Get("late") != 3 {
		t.Fatal("attribute copies lost object identity")
	}
	object.Set("self", object)
	recorder := telemetry.NewInMemory()
	if err := recorder.StartSpan(telemetry.SpanOptions{Attributes: attrs}, func(*telemetry.Span) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(recorder.GetSpans()) != 1 {
		t.Fatal("cycle rejected during recording")
	}
	if _, err := json.Marshal(recorder.GetSpans()); err == nil {
		t.Fatal("cycle exported")
	}
	object.Delete("self")
	object.Set("callable", func() { panic("must not invoke") })
	object.Set("omitted", telemetry.Undefined)
	raw, err := json.Marshal(recorder.GetSpans()[0].Attributes)
	if err != nil || string(raw) != `{"object":{"z":1,"a":2},"late":3}` {
		t.Fatalf("%s: %v", raw, err)
	}
	entries := object.Entries()
	entries[0].Value = 99
	if object.Get("z") != 1 {
		t.Fatal("entries exposed object storage")
	}
	for _, invalid := range []any{make(chan int), map[int]string{1: "unsupported"}} {
		bad := telemetry.NewObject(telemetry.Property{Name: "invalid", Value: invalid})
		if _, err := json.Marshal(bad); err == nil {
			t.Fatal("standalone object accepted unsupported Go payload")
		}
	}
}
