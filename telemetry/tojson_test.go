package telemetry_test

import (
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/2found/2ai/jsonjs"
	"github.com/2found/2ai/telemetry"
)

// Expected values were captured from the unchanged pinned Pi recorder and Bun's
// JSON.stringify. The oracle provenance is stored with the data; tests need Go only.
func TestPiTelemetryToJSON(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-tojson.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Placement, Phase, Outcome string }
			Expected json.RawMessage
		}
		Mutations []struct {
			Operation string
			Expected  json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 128 || len(fixture.Mutations) != 13 {
		t.Fatal("unexpected toJSON oracle coverage")
	}
	failure := errors.New("hook failure")
	encode := func(value any, span bool) map[string]any {
		var raw []byte
		var err error
		if span {
			raw, err = json.Marshal(value)
		} else {
			raw, err = telemetry.StringifyValue(value)
		}
		var wire, problem any
		if err != nil {
			if errors.Is(err, failure) {
				problem = failure.Error()
			} else {
				var cycle *json.UnsupportedValueError
				if !errors.As(err, &cycle) {
					t.Fatalf("unexpected export error: %v", err)
				}
				problem = "cycle"
			}
		} else if raw != nil {
			wire = string(raw)
		}
		return map[string]any{"wire": wire, "error": problem}
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Placement+"/"+tc.Input.Phase+"/"+tc.Input.Outcome, func(t *testing.T) {
			input := tc.Input
			calls := []any{}
			var original any = telemetry.NewObject(telemetry.Property{Name: "n", Value: 1})
			if input.Placement == "outer_array" || input.Placement == "nested_array" {
				original = telemetry.NewArray(1, 2)
			}
			method := telemetry.JSONMethod(func(receiver any, key string) (any, error) {
				_, array := receiver.(*telemetry.Array)
				calls = append(calls, map[string]any{"key": key, "same": receiver == original, "array": array})
				switch input.Outcome {
				case "scalar":
					return "value:" + key, nil
				case "null":
					return nil, nil
				case "undefined":
					return telemetry.Undefined, nil
				case "function":
					return func() { t.Fatal("called returned function") }, nil
				case "self":
					return receiver, nil
				case "replacement":
					return telemetry.NewObject(telemetry.Property{Name: "ok", Value: true}, telemetry.Property{Name: "toJSON", Value: telemetry.JSONMethod(func(any, string) (any, error) { t.Fatal("called replacement hook"); return nil, nil })}), nil
				case "throw":
					return nil, failure
				case "cycle":
					value := telemetry.NewObject()
					value.Set("self", value)
					return value, nil
				default:
					t.Fatal("unknown outcome")
					return nil, nil
				}
			})
			if array, ok := original.(*telemetry.Array); ok {
				array.SetProperty("toJSON", method)
			} else {
				original.(*telemetry.Object).Set("toJSON", method)
			}
			recorder := telemetry.NewInMemory()
			var value any
			before, spanValue := 0, false
			switch input.Placement {
			case "root":
				value = original
			case "property":
				value = telemetry.NewObject(telemetry.Property{Name: "value", Value: original})
			case "array":
				value = telemetry.NewArray(original)
			default:
				spanValue = true
				attributes := telemetry.NewAttributes()
				switch input.Placement {
				case "attributes":
					attributes = telemetry.NewAttributes(original.(*telemetry.Object).Entries()...)
				case "outer_array":
					attributes.Set("value", original)
				case "nested_array":
					attributes.Set("value", telemetry.NewArray(original))
				}
				if err := recorder.StartSpan(telemetry.SpanOptions{Name: "json", Attributes: attributes}, func(span *telemetry.Span) error {
					if input.Placement == "event" {
						span.AddEvent("event", telemetry.NewAttributes(original.(*telemetry.Object).Entries()...))
					}
					if input.Placement == "status" {
						details := &telemetry.ErrorDetails{}
						details.SetNameValue(original)
						details.SetMessageValue(original)
						span.SetStatus(telemetry.SpanStatus{Status: "error", Error: details})
					}
					span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "other", Value: 2}))
					value = recorder.GetSpans()
					recorder.GetSpans()
					before = len(calls)
					return nil
				}); err != nil {
					t.Fatal(err)
				}
				if input.Phase == "settled" {
					value = recorder.GetSpans()
				}
			}
			result := encode(value, spanValue)
			result["before"], result["calls"] = before, calls
			assertSchemaJSON(t, tc.Expected, result)
		})
	}
	for _, tc := range fixture.Mutations {
		t.Run("mutation/"+tc.Operation, func(t *testing.T) {
			operation, calls := tc.Operation, []string{}
			first := telemetry.NewObject()
			object := telemetry.NewObject(telemetry.Property{Name: "first", Value: first}, telemetry.Property{Name: "later", Value: 2})
			array := telemetry.NewArray(first, 2)
			var root any = object
			if len(operation) >= 6 && operation[:6] == "array_" {
				root = array
			}
			first.Set("toJSON", telemetry.JSONMethod(func(_ any, key string) (any, error) {
				calls = append(calls, key)
				switch operation {
				case "replace":
					object.Set("later", 3)
				case "delete":
					object.Delete("later")
				case "append":
					object.Set("added", 4)
				case "reinsert":
					object.Delete("later")
					object.Set("later", 5)
				case "change_hook":
					object.Get("later").(*telemetry.Object).Set("toJSON", telemetry.JSONMethod(func(_ any, key string) (any, error) { calls = append(calls, key); return 7, nil }))
				case "array_replace":
					array.Set(1, 3)
				case "array_delete":
					array.Delete(1)
				case "array_append":
					array.Append(4)
				case "array_shrink":
					array.SetLength(0)
				}
				return 1, nil
			}))
			switch operation {
			case "change_hook":
				object.Set("later", telemetry.NewObject(telemetry.Property{Name: "toJSON", Value: telemetry.JSONMethod(func(any, string) (any, error) { t.Fatal("called old hook"); return nil, nil })}))
			case "numeric":
				object = telemetry.NewObject()
				for _, number := range []int{10, 2} {
					name := "10"
					if number == 2 {
						name = "2"
					}
					object.Set(name, telemetry.NewObject(telemetry.Property{Name: "toJSON", Value: telemetry.JSONMethod(func(_ any, key string) (any, error) { calls = append(calls, key); return number, nil })}))
				}
				object.Set("x", first)
				root = object
			case "break_cycle", "keep_cycle":
				object = telemetry.NewObject()
				object.Set("self", object)
				object.Set("toJSON", telemetry.JSONMethod(func(receiver any, key string) (any, error) {
					calls = append(calls, key)
					if operation == "break_cycle" {
						return 1, nil
					}
					return receiver, nil
				}))
				root = object
			case "escape":
				root = telemetry.NewObject(telemetry.Property{Name: "toJSON", Value: telemetry.JSONMethod(func(any, string) (any, error) { calls = append(calls, ""); return "<>&\u2028\u2029\xed\xa0\x80", nil })})
			}
			result := encode(root, false)
			result["calls"] = calls
			assertSchemaJSON(t, tc.Expected, result)
		})
	}
}

func TestToJSONNativeBoundaries(t *testing.T) {
	t.Run("map cycle identity survives hook mutation", func(t *testing.T) {
		calls := 0
		value := map[string]any{}
		value["self"] = value
		value["toJSON"] = telemetry.JSONMethod(func(receiver any, key string) (any, error) {
			calls++
			if calls > 2 {
				t.Fatal("map mutation bypassed cycle detection")
			}
			name := "first"
			if calls == 2 {
				name = "second"
			}
			value[name] = true
			return receiver, nil
		})
		_, err := telemetry.StringifyValue(value)
		var cycle *json.UnsupportedValueError
		if !errors.As(err, &cycle) || calls != 2 {
			t.Fatalf("cycle: %v calls=%d", err, calls)
		}
	})
	t.Run("clone and recording are passive", func(t *testing.T) {
		calls := 0
		original := telemetry.NewArray(1)
		original.SetProperty("toJSON", telemetry.JSONMethod(func(receiver any, key string) (any, error) { calls++; return receiver.(*telemetry.Array).Get(0), nil }))
		var cloner jsonjs.ValueCloner
		clone := cloner.Clone(original).(*telemetry.Array)
		if calls != 0 {
			t.Fatal("clone invoked hook")
		}
		clone.Set(0, 2)
		raw, err := telemetry.StringifyValue(clone)
		if err != nil || string(raw) != "2" || calls != 1 || original.Get(0) != 1 {
			t.Fatalf("clone export: %s %v calls=%d", raw, err, calls)
		}
	})
	t.Run("native map visits keys in numeric order and reads live values", func(t *testing.T) {
		calls := []string{}
		value := map[string]any{"10": 10, "2": nil, "later": 1}
		value["2"] = map[string]any{"toJSON": telemetry.JSONMethod(func(receiver any, key string) (any, error) {
			calls = append(calls, key)
			value["10"] = 11
			delete(value, "later")
			value["new"] = 3
			return 2, nil
		})}
		raw, err := telemetry.StringifyValue(value)
		if err != nil || string(raw) != `{"2":2,"10":11}` || len(calls) != 1 || calls[0] != "2" {
			t.Fatalf("map export: %s %v %v", raw, err, calls)
		}
	})
	t.Run("hook replacement precedes unsupported values", func(t *testing.T) {
		value := telemetry.NewObject(telemetry.Property{Name: "opaque", Value: make(chan int)}, telemetry.Property{Name: "toJSON", Value: telemetry.JSONMethod(func(any, string) (any, error) { return 1, nil })})
		raw, err := telemetry.StringifyValue(value)
		if err != nil || string(raw) != "1" {
			t.Fatalf("export: %s %v", raw, err)
		}
	})
	t.Run("error and panic identity", func(t *testing.T) {
		failure := errors.New("identity")
		value := telemetry.NewObject(telemetry.Property{Name: "toJSON", Value: telemetry.JSONMethod(func(any, string) (any, error) { return nil, failure })})
		if _, err := telemetry.StringifyValue(value); err != failure {
			t.Fatalf("error identity: %v", err)
		}
		value.Set("toJSON", telemetry.JSONMethod(func(any, string) (any, error) { panic(failure) }))
		defer func() {
			if got := recover(); got != failure {
				t.Fatalf("panic identity: %v", got)
			}
		}()
		telemetry.StringifyValue(value)
	})
	t.Run("Go marshaler adapts omitted root", func(t *testing.T) {
		value := telemetry.NewObject(telemetry.Property{Name: "toJSON", Value: telemetry.JSONMethod(func(any, string) (any, error) { return telemetry.Undefined, nil })})
		raw, err := telemetry.StringifyValue(value)
		if err != nil || raw != nil {
			t.Fatalf("stringify: %s %v", raw, err)
		}
		raw, err = json.Marshal(value)
		if err != nil || string(raw) != "null" {
			t.Fatalf("Go adapter: %s %v", raw, err)
		}
	})
}
