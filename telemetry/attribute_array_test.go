package telemetry_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"strconv"
	"testing"

	"github.com/2found/2ai/telemetry"
)

func describeAttributeArray(array *telemetry.Array) map[string]any {
	keys := []string{}
	for _, key := range array.Keys() {
		keys = append(keys, strconv.Itoa(key))
	}
	values := []string{}
	for _, value := range array.Values() {
		switch value {
		case telemetry.Undefined:
			values = append(values, "undefined")
		case nil:
			values = append(values, "null")
		default:
			values = append(values, fmt.Sprint(value))
		}
	}
	return map[string]any{"length": array.Len(), "keys": keys, "values": values}
}

func TestPiAttributeArrayReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-attribute-arrays.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Placement, Target, Operation string
				Sparse                       bool
			}
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 112 {
		t.Fatal("unexpected array coverage")
	}
	for i, tc := range fixture.Cases {
		for _, decoded := range []bool{false, true} {
			t.Run(strconv.Itoa(i)+"/"+tc.Input.Placement+"/"+tc.Input.Operation+"/decoded="+strconv.FormatBool(decoded), func(t *testing.T) {
				original := telemetry.NewArray(float64(1), float64(2))
				if decoded {
					var attrs telemetry.Attributes
					if err := json.Unmarshal([]byte(`{"array":[1,2]}`), &attrs); err != nil {
						t.Fatal(err)
					}
					original = attrs.Get("array").(*telemetry.Array)
				}
				if tc.Input.Sparse {
					original.Delete(0)
					original.SetLength(4)
				}
				var value any = original
				switch tc.Input.Placement {
				case "object":
					value = telemetry.NewObject(telemetry.Property{Name: "inner", Value: original})
				case "array":
					value = telemetry.NewArray(original)
				case "shared":
					value = telemetry.NewObject(telemetry.Property{Name: "left", Value: original}, telemetry.Property{Name: "right", Value: original})
				}
				selectArray := func(value any) *telemetry.Array {
					switch tc.Input.Placement {
					case "object":
						return value.(*telemetry.Object).Get("inner").(*telemetry.Array)
					case "array":
						return value.(*telemetry.Array).Get(0).(*telemetry.Array)
					case "shared":
						return value.(*telemetry.Object).Get("left").(*telemetry.Array)
					default:
						return value.(*telemetry.Array)
					}
				}
				project := func(spans []telemetry.RecordedSpan) map[string]any {
					wire, err := json.Marshal(spans)
					if err != nil {
						t.Fatal(err)
					}
					arrays := []any{}
					for _, span := range spans {
						attribute := selectArray(span.Attributes.Get("value"))
						event := selectArray(span.Events[0].Attributes.Get("value"))
						description := map[string]any{"attribute": describeAttributeArray(attribute), "event": describeAttributeArray(event), "attributeInput": attribute == original, "eventInput": event == original, "same": attribute == event}
						if tc.Input.Placement == "shared" {
							object := span.Attributes.Get("value").(*telemetry.Object)
							description["shared"] = object.Get("left") == object.Get("right")
						}
						arrays = append(arrays, description)
					}
					return map[string]any{"wire": string(wire), "arrays": arrays}
				}
				recorder := telemetry.NewInMemory()
				var before, after map[string]any
				var retained []telemetry.RecordedSpan
				var result any
				attrs := telemetry.NewAttributes(telemetry.Property{Name: "value", Value: value})
				if err := recorder.StartSpan(telemetry.SpanOptions{Name: "array", Attributes: attrs}, func(span *telemetry.Span) error {
					span.AddEvent("copy", attrs)
					retained = recorder.GetSpans()
					before = project(retained)
					array := original
					if tc.Input.Target == "snapshot" {
						array = selectArray(retained[0].Attributes.Get("value"))
					}
					switch tc.Input.Operation {
					case "append":
						result = array.Append(float64(3), float64(4))
					case "set":
						array.Set(6, float64(7))
					case "delete":
						array.Delete(1)
					case "shrink":
						array.SetLength(1)
					case "grow":
						array.SetLength(6)
					case "pop":
						result = array.Pop()
						if result == telemetry.Undefined {
							result = "undefined"
						}
					case "undefined":
						array.Set(0, telemetry.Undefined)
						array.Set(1, nil)
					}
					after = project(recorder.GetSpans())
					return nil
				}); err != nil {
					t.Fatal(err)
				}
				actual, err := json.Marshal(map[string]any{"before": before, "after": after, "retained": project(retained), "settled": project(recorder.GetSpans()), "original": describeAttributeArray(original), "result": result})
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
}

func TestAttributeArrayCyclesAndPassiveExport(t *testing.T) {
	array := telemetry.NewArray("first")
	array.Append(array)
	recorder := telemetry.NewInMemory()
	if err := recorder.StartSpan(telemetry.SpanOptions{Attributes: telemetry.NewAttributes(telemetry.Property{Name: "value", Value: array})}, func(*telemetry.Span) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(recorder.GetSpans()) != 1 {
		t.Fatal("cyclic array rejected during recording")
	}
	if _, err := json.Marshal(recorder.GetSpans()); err == nil {
		t.Fatal("cycle exported")
	}
	array.Set(1, "cleared")
	snapshot := recorder.GetSpans()[0].Attributes.Get("value").(*telemetry.Array)
	if snapshot == array || snapshot.Get(1) != array {
		t.Fatal("shallow spread lost nested array identity")
	}
	raw, err := json.Marshal(snapshot)
	if err != nil || string(raw) != `["first",["first","cleared"]]` {
		t.Fatalf("%s: %v", raw, err)
	}
	calls := 0
	passive := telemetry.NewArray(attributeString("text"), func() { calls++ }, telemetry.Undefined, nil, math.Inf(1), math.Copysign(0, -1))
	passive.Delete(0)
	raw, err = json.Marshal(passive)
	if err != nil || string(raw) != `[null,null,null,null,null,0]` || calls != 0 {
		t.Fatalf("passive export: %s %v (%d calls)", raw, err, calls)
	}
	values := passive.Values()
	values[1] = "changed"
	if _, ok := passive.Get(1).(func()); !ok {
		t.Fatal("Values leaked outer storage")
	}
	for _, invalid := range []any{make(chan int), map[int]string{1: "key"}} {
		bad := telemetry.NewArray(invalid)
		if _, err := json.Marshal(bad); err == nil {
			t.Fatal("unsupported array payload exported")
		}
	}
}

func TestAttributeArraySparseLimits(t *testing.T) {
	var array telemetry.Array
	array.Set(4294967294, "last")
	if array.Len() != 4294967295 || !reflect.DeepEqual(array.Keys(), []int{4294967294}) || array.Get(0) != telemetry.Undefined || array.Has(0) {
		t.Fatal("sparse array expanded holes or lost length")
	}
	array.SetLength(0)
	if array.Len() != 0 || len(array.Keys()) != 0 || array.Pop() != telemetry.Undefined {
		t.Fatal("truncation retained elements")
	}
	for _, length := range []int{-1, 4294967296} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("invalid length accepted")
				}
			}()
			array.SetLength(length)
		}()
		if array.Len() != 0 {
			t.Fatal("invalid length changed array")
		}
	}
}
