package telemetry_test

import (
	"encoding/json"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/2found/2ai/jsonjs"
	"github.com/2found/2ai/telemetry"
)

// JSON string text is ordinary ASCII/UTF-8 here; wrapping it one level deeper
// prevents the test fixture decoder from collapsing distinct surrogate keys.
func inspectAttributeUnicode(v reflect.Value) any {
	for v.IsValid() && v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	if !v.IsValid() {
		return nil
	}
	if v.Type() == reflect.TypeFor[*telemetry.Array]() {
		return inspectAttributeUnicode(reflect.ValueOf(v.Interface().(*telemetry.Array).Values()))
	}
	if properties, ok := attributeTestProperties(v); ok {
		object := map[string]any{}
		for _, property := range properties {
			object[string(jsonjs.QuoteString(property.Name))] = inspectAttributeUnicode(reflect.ValueOf(property.Value))
		}
		return map[string]any{"object": object}
	}
	if v.Kind() == reflect.Pointer {
		return inspectAttributeUnicode(v.Elem())
	}
	switch v.Kind() {
	case reflect.String:
		return map[string]any{"string": string(jsonjs.QuoteString(v.String()))}
	case reflect.Map:
		object := map[string]any{}
		iter := v.MapRange()
		for iter.Next() {
			object[string(jsonjs.QuoteString(iter.Key().String()))] = inspectAttributeUnicode(iter.Value())
		}
		return map[string]any{"object": object}
	case reflect.Struct:
		object := map[string]any{}
		for i := 0; i < v.NumField(); i++ {
			field := v.Type().Field(i)
			if !field.IsExported() {
				continue
			}
			tag := strings.Split(field.Tag.Get("json"), ",")
			if tag[0] == "-" {
				continue
			}
			name := tag[0]
			if name == "" {
				name = field.Name
			}
			if len(tag) > 1 && tag[1] == "omitempty" && v.Field(i).IsZero() {
				continue
			}
			object[string(jsonjs.QuoteString(name))] = inspectAttributeUnicode(v.Field(i))
		}
		return map[string]any{"object": object}
	case reflect.Slice:
		items := make([]any, v.Len())
		for i := range items {
			items[i] = inspectAttributeUnicode(v.Index(i))
		}
		return items
	default:
		return v.Interface()
	}
}

func unicodeAttributeSnapshot(t *testing.T, spans []telemetry.RecordedSpan) map[string]any {
	t.Helper()
	raw, err := json.Marshal(spans)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := jsonjs.DecodeJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	return map[string]any{"memory": inspectAttributeUnicode(reflect.ValueOf(spans)), "wire": inspectAttributeUnicode(reflect.ValueOf(decoded))}
}

func TestPiAttributeUnicode(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-attribute-unicode.json")
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
		t.Fatal("unexpected Unicode coverage")
	}
	for i, tc := range fixture.Cases {
		t.Run(strconv.Itoa(i)+"/"+tc.Phase, func(t *testing.T) {
			var attributes telemetry.Attributes
			if err := json.Unmarshal([]byte(tc.Raw), &attributes); err != nil {
				t.Fatal(err)
			}
			recorder := telemetry.NewInMemory()
			start := telemetry.NewAttributes(telemetry.Property{Name: "kept", Value: true})
			if tc.Phase == "start" {
				start = attributes
			}
			var active map[string]any
			if err := recorder.StartSpan(telemetry.SpanOptions{Name: "unicode", Attributes: start}, func(span *telemetry.Span) error {
				if tc.Phase == "merge" {
					span.SetAttributes(attributes)
				}
				if tc.Phase == "event" {
					span.AddEvent("unicode", attributes)
				}
				active = unicodeAttributeSnapshot(t, recorder.GetSpans())
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			result := map[string]any{"input": inspectAttributeUnicode(reflect.ValueOf(attributes)), "active": active, "settled": unicodeAttributeSnapshot(t, recorder.GetSpans())}
			actual, err := json.Marshal(result)
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
