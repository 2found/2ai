package telemetry_test

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"sort"
	"strconv"
	"testing"

	"github.com/2found/2ai/telemetry"
)

type attributeValueSpec struct {
	Entries map[string]attributeValueSpec `json:"entries"`
	Kind    string                        `json:"kind"`
	Value   string                        `json:"value"`
	Values  []attributeValueSpec          `json:"values"`
}

func attributeInput(t *testing.T, spec attributeValueSpec) any {
	t.Helper()
	var err error
	var result any
	switch spec.Kind {
	case "object":
		object := map[string]any{}
		for key, child := range spec.Entries {
			object[key] = attributeInput(t, child)
		}
		return object
	case "json":
		var value any
		if err := json.Unmarshal([]byte(spec.Value), &value); err != nil {
			t.Fatal(err)
		}
		return value
	case "function":
		return func() { panic("attribute function invoked") }
	case "omit":
		return telemetry.Undefined
	case "string":
		return spec.Value
	case "bool":
		return spec.Value == "true"
	case "any[]":
		values := make([]any, len(spec.Values))
		for i, value := range spec.Values {
			values[i] = attributeInput(t, value)
		}
		return values
	case "uint8[]":
		var values []uint8
		for _, value := range spec.Values {
			n, err := strconv.ParseUint(value.Value, 10, 8)
			if err != nil {
				t.Fatal(err)
			}
			values = append(values, uint8(n))
		}
		return values
	case "float64[]":
		var values []float64
		for _, value := range spec.Values {
			values = append(values, attributeInput(t, value).(float64))
		}
		return values
	case "float64", "float32":
		var n float64
		n, err = strconv.ParseFloat(spec.Value, 64)
		result = n
		if spec.Kind == "float32" {
			result = float32(n)
		}
	case "int", "int8", "int16", "int64":
		var n int64
		n, err = strconv.ParseInt(spec.Value, 10, 64)
		switch spec.Kind {
		case "int":
			result = int(n)
		case "int8":
			result = int8(n)
		case "int16":
			result = int16(n)
		default:
			result = n
		}
	case "uint16", "uint64":
		var n uint64
		n, err = strconv.ParseUint(spec.Value, 10, 64)
		result = n
		if spec.Kind == "uint16" {
			result = uint16(n)
		}
	default:
		t.Fatalf("unknown attribute kind %q", spec.Kind)
	}
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestPiAttributeJSONOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/pi-attributes.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string `json:"upstreamCommit"`
		Cases          []struct {
			InspectFunctions bool                `json:"inspectFunctions"`
			Name             string              `json:"name"`
			Key              string              `json:"key"`
			Start            attributeValueSpec  `json:"start"`
			Update           *attributeValueSpec `json:"update"`
			Expected         json.RawMessage     `json:"expected"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 69 {
		t.Fatal("unexpected attribute oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			recorder := telemetry.NewInMemory()
			key := tc.Key
			if key == "" {
				key = "value"
			}
			attrs := telemetry.NewAttributes(telemetry.Property{Name: key, Value: attributeInput(t, tc.Start)})
			var active []telemetry.RecordedSpan
			if err := recorder.StartSpan(telemetry.SpanOptions{Name: tc.Name, Attributes: attrs}, func(span *telemetry.Span) error {
				span.AddEvent("copy", attrs)
				if tc.Update != nil {
					span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: key, Value: attributeInput(t, *tc.Update)}))
				}
				active = recorder.GetSpans()
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			settled := recorder.GetSpans()
			payload := map[string]any{"active": active, "settled": settled}
			if tc.InspectFunctions {
				payload["functions"] = map[string]any{"active": attributeFunctionPaths(active), "settled": attributeFunctionPaths(settled)}
			}
			actual, err := json.Marshal(payload)
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			decoder := json.NewDecoder(bytes.NewReader(actual))
			decoder.UseNumber()
			if err := decoder.Decode(&got); err != nil {
				t.Fatal(err)
			}
			decoder = json.NewDecoder(bytes.NewReader(tc.Expected))
			decoder.UseNumber()
			if err := decoder.Decode(&want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}

// Inspect memory separately: JSON alone would conceal discarded functions.
func attributeFunctionPaths(spans []telemetry.RecordedSpan) []string {
	paths := []string{}
	var visit func(reflect.Value, string)
	visit = func(value reflect.Value, path string) {
		for value.IsValid() && value.Kind() == reflect.Interface {
			value = value.Elem()
		}
		if !value.IsValid() {
			return
		}
		if value.Type() == reflect.TypeFor[*telemetry.Array]() {
			visit(reflect.ValueOf(value.Interface().(*telemetry.Array).Values()), path)
			return
		}
		if properties, ok := attributeTestProperties(value); ok {
			for _, property := range properties {
				visit(reflect.ValueOf(property.Value), path+"."+property.Name)
			}
			return
		}
		switch value.Kind() {
		case reflect.Func:
			paths = append(paths, path)
		case reflect.Map:
			entries := value.MapRange()
			for entries.Next() {
				visit(entries.Value(), path+"."+entries.Key().String())
			}
		case reflect.Slice, reflect.Array:
			for i := 0; i < value.Len(); i++ {
				visit(value.Index(i), path+"."+strconv.Itoa(i))
			}
		}
	}
	for i, span := range spans {
		prefix := strconv.Itoa(i)
		visit(reflect.ValueOf(span.Attributes), prefix+".attributes")
		for j, event := range span.Events {
			visit(reflect.ValueOf(event.Attributes), prefix+".events."+strconv.Itoa(j)+".attributes")
		}
	}
	sort.Strings(paths)
	return paths
}

type attributeString string
type attributeNumber int64
type attributeBool bool
type attributeBytes []byte

func (attributeString) MarshalJSON() ([]byte, error) { panic("must not serialize user string") }
func (attributeNumber) MarshalJSON() ([]byte, error) { panic("must not serialize user number") }
func (attributeBool) MarshalJSON() ([]byte, error)   { panic("must not serialize user bool") }
func (attributeBytes) MarshalJSON() ([]byte, error)  { panic("must not serialize user slice") }

func TestAttributeExportIsPassiveAndLeavesSnapshotsUnchanged(t *testing.T) {
	recorder := telemetry.NewInMemory()
	bytes := attributeBytes{1, 2}
	numbers := []any{attributeNumber(3), int8(4), float32(0.5)}
	attrs := telemetry.NewAttributes(
		telemetry.Property{Name: "bytes", Value: bytes}, telemetry.Property{Name: "numbers", Value: numbers},
		telemetry.Property{Name: "text", Value: attributeString("plain")}, telemetry.Property{Name: "flag", Value: attributeBool(true)},
		telemetry.Property{Name: "nan", Value: math.NaN()}, telemetry.Property{Name: "negativeZero", Value: math.Copysign(0, -1)},
	)
	if err := recorder.StartSpan(telemetry.SpanOptions{Name: "passive-export", Attributes: attrs}, func(span *telemetry.Span) error {
		span.AddEvent("values", attrs)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	bytes[0], numbers[0] = 255, "changed input"
	snapshot := recorder.GetSpans()[0]
	for _, got := range []telemetry.Attributes{snapshot.Attributes, snapshot.Events[0].Attributes} {
		raw, err := json.Marshal(got)
		if err != nil {
			t.Fatal(err)
		}
		if string(raw) != `{"bytes":[1,2],"numbers":[3,4,0.5],"text":"plain","flag":true,"nan":null,"negativeZero":0}` {
			t.Fatalf("export called a user serializer or changed numeric values: %s", raw)
		}
		if got.Get("bytes").(attributeBytes)[0] != 1 || got.Get("numbers").([]any)[0] != attributeNumber(3) || !math.IsNaN(got.Get("nan").(float64)) || !math.Signbit(got.Get("negativeZero").(float64)) {
			t.Fatal("export changed the detached in-memory snapshot")
		}
		got.Get("bytes").(attributeBytes)[0] = 200
		got.Get("numbers").([]any)[0] = "changed snapshot"
	}
	if latest := recorder.GetSpans()[0]; latest.Attributes.Get("bytes").(attributeBytes)[0] != 1 || latest.Events[0].Attributes.Get("numbers").([]any)[0] != attributeNumber(3) {
		t.Fatal("snapshot alias reached the recorder")
	}
}

func TestUnsupportedGoAttributesRejectAtomically(t *testing.T) {
	for _, invalid := range []any{make(chan int), struct{ Value int }{1}, map[int]string{1: "key"}, []any{1, make(chan int)}, map[string]any{"nested": make(chan int)}} {
		recorder := telemetry.NewInMemory()
		called := false
		_ = recorder.StartSpan(telemetry.SpanOptions{Name: "invalid-admission", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "invalid", Value: invalid})}, func(*telemetry.Span) error { called = true; return nil })
		if !called || len(recorder.GetSpans()) != 0 {
			t.Fatalf("invalid input affected callback/recorded partial span: %T", invalid)
		}
		_ = recorder.StartSpan(telemetry.SpanOptions{Name: "atomic", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "kept", Value: true})}, func(span *telemetry.Span) error {
			span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "partial", Value: true}, telemetry.Property{Name: "invalid", Value: invalid}))
			span.AddEvent("invalid", telemetry.NewAttributes(telemetry.Property{Name: "partial", Value: true}, telemetry.Property{Name: "invalid", Value: invalid}))
			return nil
		})
		span := recorder.GetSpans()[0]
		if !reflect.DeepEqual(span.Attributes, telemetry.NewAttributes(telemetry.Property{Name: "kept", Value: true})) || len(span.Events) != 0 {
			t.Fatalf("invalid mutation was partially recorded: %+v", span)
		}
		if _, err := json.Marshal(telemetry.NewAttributes(telemetry.Property{Name: "invalid", Value: invalid})); err == nil {
			t.Fatalf("invalid direct export accepted: %T", invalid)
		}
	}
}

func TestPiAttributeGraphOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/pi-attributes.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Graphs []struct {
			Kind     string          `json:"kind"`
			Expected json.RawMessage `json:"expected"`
		} `json:"graphs"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Graphs) != 4 {
		t.Fatal("unexpected graph oracle coverage")
	}
	for _, tc := range fixture.Graphs {
		t.Run(tc.Kind, func(t *testing.T) {
			recorder := telemetry.NewInMemory()
			var original any
			switch tc.Kind {
			case "object":
				original = map[string]any{"value": 1, "nested": map[string]any{"value": 2}}
			case "array":
				original = []any{1, map[string]any{"value": 2}, []any{3}}
			case "cyclic-object":
				value := map[string]any{}
				value["self"] = value
				original = value
			case "cyclic-array":
				value := []any{"original", nil}
				value[1] = value
				original = value
			default:
				t.Fatalf("unknown graph kind %s", tc.Kind)
			}
			exportFailed := false
			err := recorder.StartSpan(telemetry.SpanOptions{Name: tc.Kind, Attributes: telemetry.NewAttributes(telemetry.Property{Name: "value", Value: original})}, func(span *telemetry.Span) error {
				span.AddEvent("copy", telemetry.NewAttributes(telemetry.Property{Name: "value", Value: original}))
				spans := recorder.GetSpans()
				if len(spans) != 1 {
					t.Fatal("readable graph prevented span admission")
				}
				snapshot := spans[0].Attributes.Get("value")
				switch tc.Kind {
				case "object":
					original.(map[string]any)["value"] = 3
					snapshot.(map[string]any)["nested"].(map[string]any)["value"] = 4
				case "array":
					original.([]any)[0] = 9
					array := snapshot.([]any)
					array[0] = 8
					array[1].(map[string]any)["value"] = 4
					array[2].([]any)[0] = 5
				default:
					_, err := json.Marshal(recorder.GetSpans())
					exportFailed = err != nil
					if tc.Kind == "cyclic-object" {
						original.(map[string]any)["self"] = "cleared"
					} else {
						original.([]any)[1] = "cleared"
					}
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"exportFailed": exportFailed, "spans": recorder.GetSpans()})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}

type attributeObject map[string]any

func (attributeObject) MarshalJSON() ([]byte, error) { panic("must not serialize user object") }

func TestNestedAttributeExportIsPassive(t *testing.T) {
	object := attributeObject{
		"bytes":  attributeBytes{1, 2},
		"number": attributeNumber(3),
		"array":  [2]attributeString{"first", "second"},
		"nested": []any{attributeBool(true), math.Inf(1)},
	}
	raw, err := json.Marshal(telemetry.NewAttributes(telemetry.Property{Name: "object", Value: object}))
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `{"object":{"array":["first","second"],"bytes":[1,2],"nested":[true,null],"number":3}}` {
		t.Fatalf("nested export changed values: %s", raw)
	}
}

func TestAttributeNullRoundTripAndUndefinedPresence(t *testing.T) {
	var options telemetry.SpanOptions
	if err := json.Unmarshal([]byte(`{"name":"decoded","attributes":{"kept":null,"replaced":1,"nested":{"kept":null},"array":[null]}}`), &options); err != nil {
		t.Fatal(err)
	}
	nested := map[string]any{"kept": nil, "omitted": telemetry.Undefined}
	recorder := telemetry.NewInMemory()
	if err := recorder.StartSpan(options, func(span *telemetry.Span) error {
		span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "kept", Value: telemetry.Undefined}, telemetry.Property{Name: "replaced", Value: nil}, telemetry.Property{Name: "omitted", Value: telemetry.Undefined}, telemetry.Property{Name: "zero", Value: uint8(0)}))
		span.AddEvent("presence", telemetry.NewAttributes(telemetry.Property{Name: "kept", Value: nil}, telemetry.Property{Name: "omitted", Value: telemetry.Undefined}, telemetry.Property{Name: "nested", Value: nested}, telemetry.Property{Name: "array", Value: []any{telemetry.Undefined, nil}}))
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	spans := recorder.GetSpans()
	for _, attrs := range []telemetry.Attributes{spans[0].Attributes, spans[0].Events[0].Attributes} {
		if value, present := attrs.Lookup("kept"); !present || value != nil {
			t.Fatalf("null became absent or undefined: %#v", attrs)
		}
		if _, present := attrs.Lookup("omitted"); present {
			t.Fatalf("undefined became an own attribute: %#v", attrs)
		}
	}
	if value, present := spans[0].Attributes.Lookup("replaced"); !present || value != nil {
		t.Fatal("null failed to overwrite the previous value")
	}
	event := spans[0].Events[0].Attributes
	if event.Get("nested").(map[string]any)["omitted"] != telemetry.Undefined || event.Get("array").([]any)[0] != telemetry.Undefined {
		t.Fatal("snapshot erased nested undefined values")
	}
	raw, err := json.Marshal(spans)
	if err != nil {
		t.Fatal(err)
	}
	var decoded []telemetry.RecordedSpan
	if err = json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	again, err := json.Marshal(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != string(again) {
		t.Fatalf("JSON round trip erased null: %s -> %s", raw, again)
	}
	if _, present := decoded[0].Events[0].Attributes.Get("nested").(*telemetry.Object).Lookup("omitted"); present {
		t.Fatal("nested undefined was exported as a value")
	}
	if decoded[0].Attributes.Get("zero") != float64(0) {
		t.Fatal("ordinary uint8 zero was confused with Undefined")
	}
	nested["omitted"] = nil
	latest := recorder.GetSpans()[0].Events[0].Attributes.Get("nested").(map[string]any)
	if value, present := latest["omitted"]; !present || value != nil {
		t.Fatal("nested null edit lost shared identity")
	}
}

type attributeFunction func() int

func (attributeFunction) MarshalJSON() ([]byte, error) {
	panic("must not serialize user function")
}

func TestFunctionAttributesRetainClosuresWithoutInvokingThem(t *testing.T) {
	calls, captured := 0, 1
	callable := attributeFunction(func() int { calls++; return captured })
	nested := map[string]any{"callback": callable, "value": 1}
	array := []any{callable, nested}
	var nilFunction attributeFunction
	attrs := telemetry.NewAttributes(telemetry.Property{Name: "callback", Value: callable}, telemetry.Property{Name: "nilFunction", Value: nilFunction}, telemetry.Property{Name: "array", Value: array})
	recorder := telemetry.NewInMemory()
	if err := recorder.StartSpan(telemetry.SpanOptions{Name: "closures", Attributes: attrs}, func(span *telemetry.Span) error {
		span.AddEvent("copy", attrs)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	array[0] = "replaced input"
	nested["value"] = 2
	captured = 3
	snapshot := recorder.GetSpans()[0]
	for _, values := range []telemetry.Attributes{snapshot.Attributes, snapshot.Events[0].Attributes} {
		raw, err := json.Marshal(values)
		if err != nil || string(raw) != `{"array":[null,{"value":2}]}` {
			t.Fatalf("function export: %s, %v", raw, err)
		}
		if calls != 0 {
			t.Fatal("recording or export invoked a function")
		}
		if value, present := values.Lookup("nilFunction"); !present || value.(attributeFunction) != nil {
			t.Fatal("typed nil function lost in snapshot")
		}
		if _, ok := values.Get("array").([]any)[0].(attributeFunction); !ok {
			t.Fatal("input array mutation replaced recorded function")
		}
	}
	// Only the caller invokes the retained closure, after recording/export.
	for _, values := range []telemetry.Attributes{snapshot.Attributes, snapshot.Events[0].Attributes} {
		if values.Get("callback").(attributeFunction)() != 3 || values.Get("array").([]any)[0].(attributeFunction)() != 3 {
			t.Fatal("recording replaced the original closure")
		}
	}
	if calls != 4 {
		t.Fatalf("unexpected invocation count: %d", calls)
	}
}
