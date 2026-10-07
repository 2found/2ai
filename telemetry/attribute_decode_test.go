package telemetry_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"strconv"
	"testing"

	"github.com/2found/2ai/telemetry"
)

func attributeTestProperties(value reflect.Value) ([]telemetry.Property, bool) {
	if !value.IsValid() {
		return nil, false
	}
	if value.Type() == reflect.TypeFor[telemetry.Attributes]() {
		return value.Interface().(telemetry.Attributes).Entries(), true
	}
	if value.Type() == reflect.TypeFor[*telemetry.Object]() {
		return value.Interface().(*telemetry.Object).Entries(), true
	}
	return nil, false
}

func attributeNumberPaths(value any) map[string]string {
	paths := map[string]string{}
	var visit func(reflect.Value, string)
	visit = func(v reflect.Value, path string) {
		for v.IsValid() && v.Kind() == reflect.Interface {
			v = v.Elem()
		}
		if !v.IsValid() {
			return
		}
		if v.Type() == reflect.TypeFor[*telemetry.Array]() {
			for i, value := range v.Interface().(*telemetry.Array).Values() {
				visit(reflect.ValueOf(value), path+"/"+strconv.Itoa(i))
			}
			return
		}
		if properties, ok := attributeTestProperties(v); ok {
			for _, property := range properties {
				visit(reflect.ValueOf(property.Value), path+"/"+property.Name)
			}
			return
		}
		switch v.Kind() {
		case reflect.Float64:
			paths[path] = fmt.Sprintf("%016x", math.Float64bits(v.Float()))
		case reflect.Int:
			paths[path] = fmt.Sprintf("%016x", math.Float64bits(float64(v.Int())))
		case reflect.Map:
			entries := v.MapRange()
			for entries.Next() {
				visit(entries.Value(), path+"/"+entries.Key().String())
			}
		case reflect.Slice:
			for i := 0; i < v.Len(); i++ {
				visit(v.Index(i), path+"/"+strconv.Itoa(i))
			}
		}
	}
	visit(reflect.ValueOf(value), "")
	return paths
}

func decodedAttributeSnapshot(spans []telemetry.RecordedSpan) map[string]any {
	values := make([]any, len(spans))
	for i, span := range spans {
		events := make([]any, len(span.Events))
		for j, event := range span.Events {
			events[j] = event.Attributes
		}
		values[i] = map[string]any{"attributes": span.Attributes, "events": events}
	}
	return map[string]any{"spans": spans, "numbers": attributeNumberPaths(values)}
}

func TestPiAttributeDecodeNumbers(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-attribute-decode.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Raw, Phase string
			Expected   json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 60 {
		t.Fatal("unexpected attribute decode coverage")
	}
	for i, tc := range fixture.Cases {
		t.Run(strconv.Itoa(i)+"/"+tc.Phase, func(t *testing.T) {
			var decoded telemetry.Attributes
			if err := json.Unmarshal([]byte(tc.Raw), &decoded); err != nil {
				t.Fatal(err)
			}
			recorder := telemetry.NewInMemory()
			attributes := telemetry.NewAttributes(telemetry.Property{Name: "kept", Value: true}, telemetry.Property{Name: "value", Value: float64(7)})
			if tc.Phase == "start" {
				attributes = decoded
			}
			var active map[string]any
			if err := recorder.StartSpan(telemetry.SpanOptions{Name: "decoded", Attributes: attributes}, func(span *telemetry.Span) error {
				if tc.Phase == "merge" {
					span.SetAttributes(decoded)
				}
				if tc.Phase == "event" {
					span.AddEvent("decoded", decoded)
				}
				active = decodedAttributeSnapshot(recorder.GetSpans())
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"decodedNumbers": attributeNumberPaths(decoded), "active": active, "settled": decodedAttributeSnapshot(recorder.GetSpans())})
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

func TestAttributeDecodeReplacementAndFailure(t *testing.T) {
	attributes := telemetry.NewAttributes(telemetry.Property{Name: "old", Value: "retained by alias"})
	alias := attributes
	if err := json.Unmarshal([]byte(`{"next":1e400,"nested":{"zero":-1e-400}}`), &attributes); err != nil {
		t.Fatal(err)
	}
	if _, exists := attributes.Lookup("old"); exists || alias.Len() != 1 || alias.Get("old") != "retained by alias" {
		t.Fatal("decode merged into or modified the previous object")
	}
	if !math.IsInf(attributes.Get("next").(float64), 1) || !math.Signbit(attributes.Get("nested").(*telemetry.Object).Get("zero").(float64)) {
		t.Fatal("decode lost live numeric values")
	}
	before := attributes
	beforeJSON, err := json.Marshal(attributes)
	if err != nil {
		t.Fatal(err)
	}
	beforeNumbers := attributeNumberPaths(attributes)
	for _, raw := range []string{`{"value":1e}`, `{"value":NaN}`, `{"value":Infinity}`, `{"value":+1}`, `{"value":01}`, `{"value":1} {}`, `[]`, `true`, `42`, `"text"`} {
		t.Run(raw, func(t *testing.T) {
			// Exercise the method directly as well: callers need not use json.Unmarshal.
			if err := attributes.UnmarshalJSON([]byte(raw)); err == nil {
				t.Fatal("accepted malformed JSON or a non-object attribute payload")
			}
			afterJSON, err := json.Marshal(attributes)
			if err != nil || !bytes.Equal(beforeJSON, afterJSON) || !reflect.DeepEqual(beforeNumbers, attributeNumberPaths(attributes)) || attributes != before {
				t.Fatal("rejected decode changed the receiver")
			}
		})
	}
	if err := json.Unmarshal([]byte(`{}`), &attributes); err != nil || attributes == (telemetry.Attributes{}) || attributes.Len() != 0 {
		t.Fatalf("empty object: %#v, %v", attributes, err)
	}
	if err := json.Unmarshal([]byte(`null`), &attributes); err != nil || attributes != (telemetry.Attributes{}) {
		t.Fatalf("null object: %#v, %v", attributes, err)
	}
}

func TestAttributeDecodeInRecordedTypes(t *testing.T) {
	var options telemetry.SpanOptions
	if err := json.Unmarshal([]byte(`{"name":"input","attributes":{"infinite":-1e400,"array":[1e400,-0]}}`), &options); err != nil {
		t.Fatal(err)
	}
	recorder := telemetry.NewInMemory()
	if err := recorder.StartSpan(options, func(span *telemetry.Span) error {
		var event telemetry.RecordedEvent
		if err := json.Unmarshal([]byte(`{"name":"event","attributes":{"infinite":1e400}}`), &event); err != nil {
			return err
		}
		span.AddEvent(event.Name, event.Attributes)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	spans := recorder.GetSpans()
	if !math.IsInf(spans[0].Attributes.Get("infinite").(float64), -1) || !math.IsInf(spans[0].Events[0].Attributes.Get("infinite").(float64), 1) {
		t.Fatal("options or event decoding lost numeric values")
	}
	// Exported nonfinite numbers are null, and importing that JSON must keep
	// null rather than reconstructing the original live infinity.
	raw, err := json.Marshal(spans)
	if err != nil {
		t.Fatal(err)
	}
	var decoded []telemetry.RecordedSpan
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	value, exists := decoded[0].Attributes.Lookup("infinite")
	if !exists || value != nil || decoded[0].Events[0].Attributes.Get("infinite") != nil {
		t.Fatal("exported null was lost or reinterpreted")
	}
}
