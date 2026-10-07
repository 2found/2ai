package engine

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"github.com/2found/2ai/jsonjs"
)

// Pi catch sites use Error.message, otherwise String(value). Keep Go error
// identity while formatting JSON-shaped panic values with the source's rules.
func failureError(value any) error {
	if _, ok := value.(*runtime.PanicNilError); ok {
		return errors.New("null")
	}
	if failure, ok := value.(error); ok {
		return failure
	}
	return errors.New(jsValueString(value))
}

func jsValueString(value any) string {
	return jsReflectString(reflect.ValueOf(value), make(map[jsArrayIdentity]bool))
}

// Array#join elides a recursive array only while that array is already being
// joined. A shared child in separate branches must still appear in each one.
type jsArrayIdentity struct {
	pointer uintptr
	length  int
}

func jsReflectString(value reflect.Value, active map[jsArrayIdentity]bool) string {
	for value.IsValid() && value.Kind() == reflect.Interface {
		value = value.Elem()
	}
	if !value.IsValid() {
		return "null"
	}
	switch value.Kind() {
	case reflect.String:
		return value.String()
	case reflect.Bool:
		return strconv.FormatBool(value.Bool())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return jsNumberString(float64(value.Int()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return jsNumberString(float64(value.Uint()))
	case reflect.Float32, reflect.Float64:
		return jsNumberString(value.Float())
	case reflect.Map:
		return "[object Object]"
	case reflect.Slice, reflect.Array:
		if value.Len() == 0 {
			return ""
		}
		if value.Kind() == reflect.Slice {
			identity := jsArrayIdentity{pointer: value.Pointer(), length: value.Len()}
			if active[identity] {
				return ""
			}
			active[identity] = true
			defer delete(active, identity)
		}
		items := make([]string, value.Len())
		for i := range items {
			item := value.Index(i)
			for item.IsValid() && item.Kind() == reflect.Interface {
				item = item.Elem()
			}
			// Null/undefined elements join as empty strings, unlike top-level null.
			if item.IsValid() {
				items[i] = jsReflectString(item, active)
			}
		}
		return strings.Join(items, ",")
	default:
		return fmt.Sprint(value.Interface())
	}
}

func jsNumberString(value float64) string {
	switch {
	case math.IsNaN(value):
		return "NaN"
	case math.IsInf(value, 1):
		return "Infinity"
	case math.IsInf(value, -1):
		return "-Infinity"
	}
	raw, _ := marshalJSScalar(value)
	return string(raw)
}

func jsArrayIndex(key string) (uint64, bool) {
	n, err := strconv.ParseUint(key, 10, 32)
	return n, err == nil && n < 4294967295 && strconv.FormatUint(n, 10) == key
}

// Rebuild container identity in source property order without a JSON round trip.
// Scalar values keep their binary64 bits and WTF-8 strings throughout validation.
func orderedArgumentValue(value, template any) any {
	switch value := value.(type) {
	case map[string]any:
		object := jsonjs.NewObject()
		original, _ := template.(*jsonjs.Object)
		for _, property := range original.Entries() {
			if child, exists := value[property.Name]; exists {
				object.Set(property.Name, orderedArgumentValue(child, property.Value))
			}
		}
		extra := []string{}
		for key := range value {
			if _, exists := original.Lookup(key); !exists {
				extra = append(extra, key)
			}
		}
		slices.Sort(extra)
		for _, key := range extra {
			object.Set(key, orderedArgumentValue(value[key], nil))
		}
		return object
	case []any:
		array := jsonjs.NewArray()
		original, _ := template.(*jsonjs.Array)
		for i, child := range value {
			array.Append(orderedArgumentValue(child, original.Get(i)))
		}
		return array
	default:
		return value
	}
}

// Union candidates use structured copies: JSON serialization would turn
// infinity into null, erase negative zero and alter lone UTF-16 surrogates.
func cloneArgument(value any) any {
	switch value := value.(type) {
	case map[string]any:
		clone := make(map[string]any, len(value))
		for key, child := range value {
			clone[key] = cloneArgument(child)
		}
		return clone
	case []any:
		clone := make([]any, len(value))
		for i, child := range value {
			clone[i] = cloneArgument(child)
		}
		return clone
	default:
		return value
	}
}

func marshalArguments(value any, source json.RawMessage) ([]byte, error) {
	var template any
	if len(source) != 0 {
		var err error
		template, err = jsonjs.DecodeValue(source)
		if err != nil {
			return nil, err
		}
	}
	raw, err := jsonjs.MarshalValue(orderedArgumentValue(value, template))
	if err != nil {
		return nil, err
	}
	return jsonjs.StringifyJSON(raw)
}

func marshalJSScalar(value any) ([]byte, error) {
	if text, ok := value.(string); ok {
		return jsonjs.QuoteString(text), nil
	}
	if number, ok := value.(float64); ok && (math.IsInf(number, 0) || math.IsNaN(number)) {
		return []byte("null"), nil
	}
	if number, ok := value.(float64); ok && number == 0 {
		return []byte("0"), nil
	}
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	raw := bytes.TrimSuffix(out.Bytes(), []byte("\n"))
	// Go escapes line/paragraph separators even with HTML escaping disabled;
	// JSON.stringify preserves them. Skip escaped backslashes so literal
	// strings such as "\\u2028" retain their spelling.
	result := make([]byte, 0, len(raw))
	for i := 0; i < len(raw); i++ {
		if raw[i] == '\\' && i+1 < len(raw) {
			if i+6 <= len(raw) && (string(raw[i:i+6]) == `\u2028` || string(raw[i:i+6]) == `\u2029`) {
				if raw[i+5] == '8' {
					result = append(result, []byte("\u2028")...)
				} else {
					result = append(result, []byte("\u2029")...)
				}
				i += 5
				continue
			}
			result = append(result, raw[i], raw[i+1])
			i++
			continue
		}
		result = append(result, raw[i])
	}
	return result, nil
}
