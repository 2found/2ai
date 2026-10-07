package telemetry_test

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/jsonjs"
	"github.com/2found/2ai/telemetry"
)

func describeNamedArray(t *testing.T, array *telemetry.Array, meta *telemetry.Object) map[string]any {
	t.Helper()
	wire, err := json.Marshal(array)
	if err != nil {
		t.Fatal(err)
	}
	entries := []any{}
	for _, name := range array.PropertyKeys() {
		value, present := array.GetProperty(name)
		if !present {
			t.Fatalf("missing enumerated property %q", name)
		}
		var description map[string]any
		switch value := value.(type) {
		case *telemetry.Object:
			description = map[string]any{"kind": "object", "n": value.Get("n"), "input": value == meta}
		case *telemetry.Array:
			description = map[string]any{"kind": "array", "self": value == array}
		case string:
			description = map[string]any{"kind": "string", "value": value}
		case nil:
			description = map[string]any{"kind": "null"}
		default:
			if value == telemetry.Undefined {
				description = map[string]any{"kind": "undefined"}
			} else if reflect.TypeOf(value).Kind() == reflect.Func {
				description = map[string]any{"kind": "function"}
			} else {
				description = map[string]any{"kind": "number", "value": value}
			}
		}
		entries = append(entries, []any{name, description})
	}
	return map[string]any{"length": array.Len(), "keys": array.PropertyKeys(), "wire": string(wire), "entries": entries}
}

func TestPiTelemetryArrayProperties(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-array-properties.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Placement, Target, Phase, Operation string }
			Expected json.RawMessage
		}
		Clones []struct {
			Operation string
			Expected  json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 240 || len(fixture.Clones) != 7 {
		t.Fatal("unexpected named-array oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Placement+"/"+tc.Input.Target+"/"+tc.Input.Phase+"/"+tc.Input.Operation, func(t *testing.T) {
			input := tc.Input
			original := telemetry.NewArray(1, 2)
			meta := telemetry.NewObject(telemetry.Property{Name: "n", Value: 1})
			for _, p := range []telemetry.Property{{Name: "meta", Value: meta}, {Name: "-1", Value: 3}, {Name: "01", Value: 4}, {Name: "constructor", Value: "tag"}, {Name: "4294967295", Value: 5}, {Name: "__proto__", Value: meta}} {
				original.SetProperty(p.Name, p.Value)
			}
			var value any = original
			switch input.Placement {
			case "object", "status":
				value = telemetry.NewObject(telemetry.Property{Name: "inner", Value: original})
			case "array":
				value = telemetry.NewArray(original)
			case "shared":
				value = telemetry.NewObject(telemetry.Property{Name: "left", Value: original}, telemetry.Property{Name: "right", Value: original})
			}
			selectArray := func(value any) *telemetry.Array {
				switch input.Placement {
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
			arrays := func(span telemetry.RecordedSpan) (*telemetry.Array, *telemetry.Array) {
				if input.Placement == "status" {
					return span.Status.Error.NameValue().(*telemetry.Array), span.Status.Error.MessageValue().(*telemetry.Array)
				}
				return selectArray(span.Attributes.Get("value")), selectArray(span.Events[0].Attributes.Get("value"))
			}
			project := func(spans []telemetry.RecordedSpan) map[string]any {
				wire, err := json.Marshal(spans)
				if err != nil {
					t.Fatal(err)
				}
				descriptions := []any{}
				for _, span := range spans {
					first, second := arrays(span)
					descriptions = append(descriptions, map[string]any{"first": describeNamedArray(t, first, meta), "second": describeNamedArray(t, second, meta), "same": first == second, "firstInput": first == original, "secondInput": second == original})
				}
				return map[string]any{"wire": string(wire), "arrays": descriptions}
			}
			recorder := telemetry.NewInMemory()
			var before, after map[string]any
			var retained []telemetry.RecordedSpan
			calls, applied := 0, true
			mutate := func() {
				array := original
				if input.Target == "snapshot" {
					array, _ = arrays(retained[0])
				}
				switch input.Operation {
				case "late":
					array.SetProperty("late", 7)
				case "overwrite":
					array.SetProperty("meta", 7)
				case "reinsert":
					array.DeleteProperty("01")
					array.SetProperty("01", 8)
				case "shrink":
					array.SetLength(0)
				case "numeric":
					array.SetProperty("3", 4)
				case "nested":
					if meta, found := array.GetProperty("meta"); found {
						meta.(*telemetry.Object).Set("n", 8)
					} else {
						applied = false
					}
				case "cycle":
					array.SetProperty("self", array)
				case "undefined":
					array.SetProperty("note", telemetry.Undefined)
				case "function":
					array.SetProperty("fn", func() { calls++ })
				case "constructor":
					array.SetProperty("constructor", nil)
				case "proto":
					array.SetProperty("__proto__", telemetry.NewObject(telemetry.Property{Name: "n", Value: 9}))
				case "max":
					array.SetProperty("4294967295", 9)
				}
			}
			attributes := telemetry.NewAttributes()
			if input.Placement != "status" {
				attributes.Set("value", value)
			}
			if err := recorder.StartSpan(telemetry.SpanOptions{Name: "properties", Attributes: attributes}, func(span *telemetry.Span) error {
				if input.Placement == "status" {
					details := &telemetry.ErrorDetails{}
					details.SetNameValue(original)
					details.SetMessageValue(original)
					span.SetStatus(telemetry.SpanStatus{Status: "error", Error: details})
				} else {
					span.AddEvent("copy", attributes)
				}
				retained = recorder.GetSpans()
				before = project(retained)
				if input.Phase == "active" {
					mutate()
					after = project(recorder.GetSpans())
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if input.Phase == "settled" {
				mutate()
				after = project(recorder.GetSpans())
			}
			assertSchemaJSON(t, tc.Expected, map[string]any{"before": before, "after": after, "retained": project(retained), "settled": project(recorder.GetSpans()), "original": describeNamedArray(t, original, meta), "calls": calls, "applied": applied})
		})
	}
	for _, tc := range fixture.Clones {
		t.Run("clone/"+tc.Operation, func(t *testing.T) {
			meta := telemetry.NewObject(telemetry.Property{Name: "n", Value: 1})
			array := telemetry.NewArray(meta)
			array.SetProperty("meta", meta)
			array.SetProperty("self", array)
			array.SetProperty("__proto__", meta)
			array.SetProperty("constructor", nil)
			if tc.Operation == "sparse" {
				array.SetLength(3)
				array.Delete(0)
			}
			root := telemetry.NewObject(telemetry.Property{Name: "array", Value: array}, telemetry.Property{Name: "meta", Value: meta})
			var cloner jsonjs.ValueCloner
			clone := cloner.Clone(root).(*telemetry.Object)
			clonedArray := clone.Get("array").(*telemetry.Array)
			switch tc.Operation {
			case "clone_meta":
				value, _ := clonedArray.GetProperty("meta")
				value.(*telemetry.Object).Set("n", 2)
			case "input_meta":
				meta.Set("n", 2)
			case "replace_meta":
				clonedArray.SetProperty("meta", telemetry.NewObject(telemetry.Property{Name: "n", Value: 3}))
			case "delete_meta":
				clonedArray.DeleteProperty("meta")
			case "truncate":
				clonedArray.SetLength(0)
			}
			inspect := func(root *telemetry.Object) map[string]any {
				array := root.Get("array").(*telemetry.Array)
				named, _ := array.GetProperty("meta")
				proto, _ := array.GetProperty("__proto__")
				self, _ := array.GetProperty("self")
				return map[string]any{"array": describeNamedArray(t, array, meta), "meta": root.Get("meta").(*telemetry.Object).Get("n"), "same": named == root.Get("meta"), "proto": proto == root.Get("meta"), "self": self == array}
			}
			assertSchemaJSON(t, tc.Expected, map[string]any{"original": inspect(root), "clone": inspect(clone), "detached": clonedArray != array && clone.Get("meta") != meta})
		})
	}
}

func TestNamedArrayIndexBoundaryAndIgnoredPayloads(t *testing.T) {
	array := telemetry.NewArray()
	array.SetProperty("4294967295", "named")
	array.SetProperty("4294967294", "indexed")
	if array.Len() != 4294967295 || !reflect.DeepEqual(array.PropertyKeys(), []string{"4294967294", "4294967295"}) {
		t.Fatal("array-index boundary changed length or property order")
	}
	array.SetLength(0)
	if value, found := array.GetProperty("4294967295"); !found || value != "named" {
		t.Fatal("truncation removed named property")
	}
	if value, found := array.GetProperty("4294967294"); found || value != telemetry.Undefined {
		t.Fatal("truncation retained array index")
	}
	array.SetProperty("opaque", make(chan int))
	array.SetProperty("self", array)
	if raw, err := json.Marshal(array); err != nil || string(raw) != `[]` {
		t.Fatalf("ignored named fields affected JSON: %s, %v", raw, err)
	}
	recorder := telemetry.NewInMemory()
	if err := recorder.StartSpan(telemetry.SpanOptions{Attributes: telemetry.NewAttributes(telemetry.Property{Name: "nested", Value: telemetry.NewObject(telemetry.Property{Name: "array", Value: array})})}, func(*telemetry.Span) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(recorder.GetSpans()) != 1 {
		t.Fatal("ignored named payload rejected recording")
	}
}
