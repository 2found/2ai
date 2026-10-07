package jsonjs

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

// PropertyReadError retains the pinned runtime's diagnostic for reading a
// property on null (present) or undefined (absent).
func PropertyReadError(present bool, expression string) error {
	value := "undefined"
	if present {
		value = "null"
	}
	return fmt.Errorf("%s is not an object (evaluating '%s')", value, expression)
}

// JSONMethod is an explicit own toJSON hook. Export passes the current object
// or array as receiver and its containing property key (empty at the root).
// Recording/cloning never invoke it. Returned errors and panics propagate.
// Other Go function types and arbitrary serialization methods remain passive.
type JSONMethod func(receiver any, key string) (any, error)

// StringifyValue follows JSON.stringify for the supported value graph. An
// undefined/function root, including a hook result, returns no bytes and no error.
func StringifyValue(value any) ([]byte, error) { return StringifyProperty(value, "") }

// StringifyProperty exports a value at a known property boundary. It lets typed
// Go host serializers retain the toJSON key and omit undefined hook results.
func StringifyProperty(value any, key string) ([]byte, error) {
	converted, err := jsonValue(reflect.ValueOf(value), key, make(map[valueIdentity]bool))
	if err != nil {
		return nil, err
	}
	if IsUndefined(converted) {
		return nil, nil
	}
	raw, err := json.Marshal(converted)
	if err != nil {
		return nil, err
	}
	return StringifyJSON(raw)
}

// MarshalValue adapts export to Go's json.Marshaler, which requires a JSON value:
// an undefined root becomes null. StringifyValue preserves root omission instead.
func MarshalValue(value any) ([]byte, error) {
	raw, err := StringifyValue(value)
	if err == nil && raw == nil {
		raw = []byte("null")
	}
	return raw, err
}

func SupportedValue(value any) bool {
	return supportedValue(reflect.ValueOf(value), make(map[valueIdentity]bool))
}

type undefinedValue uint8

const Undefined undefinedValue = 0

func IsUndefined(value any) bool { _, ok := value.(undefinedValue); return ok }

func (o *Object) MarshalJSON() ([]byte, error) { return MarshalValue(o) }

func objectValue(value reflect.Value) (*Object, bool) {
	if !value.IsValid() {
		return nil, false
	}
	if value.Type() == reflect.TypeFor[ObjectValue]() {
		return value.Interface().(ObjectValue).object, true
	}
	if value.Type() == reflect.TypeFor[*Object]() {
		return value.Interface().(*Object), true
	}
	return nil, false
}

func objectIdentity(object *Object) valueIdentity {
	return valueIdentity{kind: reflect.Pointer, pointer: reflect.ValueOf(object).Pointer(), valueType: reflect.TypeFor[*Object]()}
}

// Track active paths during export, so cycles fail without recursing
// forever while shared non-cyclic children can appear more than once.
type valueIdentity struct {
	kind      reflect.Kind
	pointer   uintptr
	length    int
	valueType reflect.Type
}

func valueReference(value reflect.Value) valueIdentity {
	identity := valueIdentity{kind: value.Kind(), pointer: value.Pointer(), valueType: value.Type()}
	// Slice views may share a backing pointer with different lengths. Map
	// identity must stay stable when a hook adds or deletes properties.
	if value.Kind() == reflect.Slice {
		identity.length = value.Len()
	}
	return identity
}

func jsonValue(value reflect.Value, key string, active map[valueIdentity]bool) (any, error) {
	for value.IsValid() && value.Kind() == reflect.Interface {
		value = value.Elem()
	}
	var receiver, candidate any
	if object, ok := objectValue(value); ok && object != nil {
		receiver, candidate = object, object.Get("toJSON")
	} else if array, ok := referenceArray(value); ok && array != nil {
		receiver = array
		candidate, _ = array.GetProperty("toJSON")
	} else if value.IsValid() && value.Kind() == reflect.Map && value.Type().Key().Kind() == reflect.String {
		if method := value.MapIndex(reflect.ValueOf("toJSON").Convert(value.Type().Key())); method.IsValid() {
			receiver, candidate = value.Interface(), method.Interface()
		}
	}
	if method, ok := candidate.(JSONMethod); ok && method != nil {
		replacement, err := method(receiver, key)
		if err != nil {
			return nil, err
		}
		value = reflect.ValueOf(replacement)
		for value.IsValid() && value.Kind() == reflect.Interface {
			value = value.Elem()
		}
	}
	if omittedJSONValue(value) {
		return Undefined, nil
	}
	if !value.IsValid() || (value.CanInterface() && IsNull(value.Interface())) {
		return nil, nil
	}
	if array, ok := referenceArray(value); ok {
		if array != nil {
			identity := arrayIdentity(array)
			if active[identity] {
				return nil, &json.UnsupportedValueError{Value: value, Str: "encountered a cycle via jsonjs.Array"}
			}
			active[identity] = true
			defer delete(active, identity)
		}
		result := make([]any, array.Len())
		for i := range result {
			child, err := jsonValue(reflect.ValueOf(array.Get(i)), strconv.Itoa(i), active)
			if err != nil {
				return nil, err
			}
			if !IsUndefined(child) {
				result[i] = child
			}
		}
		return result, nil
	}
	if object, ok := objectValue(value); ok {
		if object != nil {
			identity := objectIdentity(object)
			if active[identity] {
				return nil, &json.UnsupportedValueError{Value: value, Str: "encountered a cycle via jsonjs.Object"}
			}
			active[identity] = true
			defer delete(active, identity)
		}
		keys := []string{}
		for _, property := range object.Entries() {
			keys = append(keys, property.Name)
		}
		return jsonProperties(keys, func(key string) any {
			if value, found := object.Lookup(key); found {
				return value
			}
			return Undefined
		}, active)
	}
	if value.Kind() == reflect.Map || value.Kind() == reflect.Slice {
		identity := valueReference(value)
		if active[identity] {
			return nil, &json.UnsupportedValueError{Value: value, Str: "encountered a cycle via " + value.Type().String()}
		}
		active[identity] = true
		defer delete(active, identity)
	}
	switch value.Kind() {
	case reflect.String:
		return json.RawMessage(QuoteString(value.String())), nil
	case reflect.Bool:
		return value.Bool(), nil
	case reflect.Slice, reflect.Array:
		array := make([]any, value.Len())
		for i := range array {
			child, err := jsonValue(value.Index(i), strconv.Itoa(i), active)
			if err != nil {
				return nil, err
			}
			if !IsUndefined(child) {
				array[i] = child
			}
		}
		return array, nil
	case reflect.Map:
		if value.Type().Key().Kind() != reflect.String {
			return nil, &json.UnsupportedTypeError{Type: value.Type()}
		}
		// Build escaped keys directly: encoding/json would replace WTF-8
		// surrogate bytes and could collapse distinct JavaScript properties.
		// Native maps have no insertion order; keep deterministic key order,
		// with JavaScript's numeric index keys first.
		keys := value.MapKeys()
		slices.SortFunc(keys, func(a, b reflect.Value) int { return strings.Compare(a.String(), b.String()) })
		names := make([]string, 0, len(keys))
		for _, key := range keys {
			names = append(names, key.String())
		}
		// Sort numeric keys before visiting values, not just before writing bytes.
		slices.SortStableFunc(names, func(a, b string) int {
			ai, ax := ArrayIndex(a)
			bi, bx := ArrayIndex(b)
			if ax && bx {
				if ai < bi {
					return -1
				}
				if ai > bi {
					return 1
				}
				return 0
			}
			if ax {
				return -1
			}
			if bx {
				return 1
			}
			return 0
		})
		return jsonProperties(names, func(key string) any {
			child := value.MapIndex(reflect.ValueOf(key).Convert(value.Type().Key()))
			if child.IsValid() {
				return child.Interface()
			}
			return Undefined
		}, active)
	}
	var number float64
	switch value.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		number = float64(value.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		number = float64(value.Uint())
	case reflect.Float32, reflect.Float64:
		number = value.Float()
	default:
		return nil, &json.UnsupportedTypeError{Type: value.Type()}
	}
	if math.IsNaN(number) || math.IsInf(number, 0) {
		return nil, nil
	}
	if number == 0 {
		return float64(0), nil
	}
	return number, nil
}

func jsonProperties(keys []string, read func(string) any, active map[valueIdentity]bool) (any, error) {
	var object ObjectFields
	for _, key := range keys {
		child, err := jsonValue(reflect.ValueOf(read(key)), key, active)
		if err != nil {
			return nil, err
		}
		if IsUndefined(child) {
			continue
		}
		raw, err := json.Marshal(child)
		if err != nil {
			return nil, err
		}
		object.Set(QuoteString(key), raw)
	}
	return json.RawMessage(object.Marshal()), nil
}

func omittedJSONValue(value reflect.Value) bool {
	for value.IsValid() && value.Kind() == reflect.Interface {
		value = value.Elem()
	}
	return value.IsValid() && (value.Type() == reflect.TypeFor[undefinedValue]() || value.Kind() == reflect.Func)
}

// Arbitrary Go channels, pointers and structs have no JSON-shaped counterpart.
// Mixed arrays, nested objects, functions and cycles remain admissible values;
// export detects cycles separately.
func supportedValue(value reflect.Value, seen map[valueIdentity]bool) bool {
	for value.IsValid() && value.Kind() == reflect.Interface {
		value = value.Elem()
	}
	if array, ok := referenceArray(value); ok {
		if array == nil {
			return true
		}
		identity := arrayIdentity(array)
		if seen[identity] {
			return true
		}
		seen[identity] = true
		// Array export/spread visits indexed entries only. Named properties can
		// retain any opaque value without affecting JSON payload admission.
		for _, child := range array.values {
			if !supportedValue(reflect.ValueOf(child), seen) {
				return false
			}
		}
		return true
	}
	if object, ok := objectValue(value); ok {
		if object == nil {
			return true
		}
		identity := objectIdentity(object)
		if seen[identity] {
			return true
		}
		seen[identity] = true
		for _, property := range object.Entries() {
			if !supportedValue(reflect.ValueOf(property.Value), seen) {
				return false
			}
		}
		return true
	}
	if !value.IsValid() || value.Kind() == reflect.Func || scalarKind(value.Kind()) != reflect.Invalid {
		return true
	}
	switch value.Kind() {
	case reflect.Map:
		if value.Type().Key().Kind() != reflect.String {
			return false
		}
	case reflect.Slice, reflect.Array:
	default:
		return false
	}
	if value.Kind() != reflect.Array {
		identity := valueReference(value)
		if seen[identity] {
			return true
		}
		seen[identity] = true
	}
	if value.Kind() == reflect.Map {
		entries := value.MapRange()
		for entries.Next() {
			if !supportedValue(entries.Value(), seen) {
				return false
			}
		}
	} else {
		for i := 0; i < value.Len(); i++ {
			if !supportedValue(value.Index(i), seen) {
				return false
			}
		}
	}
	return true
}

func scalarKind(kind reflect.Kind) reflect.Kind {
	switch kind {
	case reflect.String, reflect.Bool:
		return kind
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64:
		return reflect.Float64
	default:
		return reflect.Invalid
	}
}
