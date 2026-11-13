package engine

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"strconv"
)

// The protocol builds this object in a fixed order. Preserve that order for
// both wire JSON and user MarshalJSON callbacks, as with JSON.stringify.
type proxyRequestOption struct {
	name  string
	value any
}
type proxyRequestOptions []proxyRequestOption

func (options proxyRequestOptions) MarshalJSON() ([]byte, error) {
	var out bytes.Buffer
	out.WriteByte('{')
	active := make(map[proxyJSONIdentity]bool)
	for i, option := range options {
		if i > 0 {
			out.WriteByte(',')
		}
		name, _ := marshalJSScalar(option.name)
		value, err := marshalProxyOption(reflect.ValueOf(option.value), active)
		if err != nil {
			return nil, err
		}
		out.Write(name)
		out.WriteByte(':')
		out.Write(value)
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

type proxyJSONIdentity struct {
	typ     reflect.Type
	pointer uintptr
	length  int
}

// Visit each value at serialization time so callbacks can change later values.
// Custom Go marshalers supply their JSON before final normalization. JSON-shaped
// values use JS's binary64 number domain and null for non-finite numbers.
func marshalProxyOption(value reflect.Value, active map[proxyJSONIdentity]bool) ([]byte, error) {
	for value.IsValid() && value.Kind() == reflect.Interface {
		value = value.Elem()
	}
	if !value.IsValid() {
		return []byte("null"), nil
	}
	if _, ok := value.Interface().(json.Marshaler); ok {
		return marshalJSScalar(value.Interface())
	}
	if number, ok := value.Interface().(json.Number); ok {
		n, err := strconv.ParseFloat(string(number), 64)
		if err != nil && !math.IsInf(n, 0) {
			return nil, err
		}
		return proxyNumberJSON(n)
	}
	switch value.Kind() {
	case reflect.Float32, reflect.Float64:
		return proxyNumberJSON(value.Float())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return proxyNumberJSON(float64(value.Int()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return proxyNumberJSON(float64(value.Uint()))
	case reflect.Map:
		if value.Type().Key().Kind() != reflect.String {
			return marshalJSScalar(value.Interface())
		}
	case reflect.Slice:
		// Preserve Go's explicitly binary byte-slice representation. RawMessage
		// and custom slice marshalers were handled above.
		if value.Type().Elem().Kind() == reflect.Uint8 {
			return marshalJSScalar(value.Interface())
		}
	case reflect.Array:
	default:
		return marshalJSScalar(value.Interface())
	}
	if value.Kind() != reflect.Array {
		if value.IsNil() {
			return []byte("null"), nil
		}
		identity := proxyJSONIdentity{value.Type(), value.Pointer(), value.Len()}
		if value.Kind() == reflect.Map {
			identity.length = 0 // Deleting fields does not change object identity.
		}
		if active[identity] {
			return nil, errors.New("JSON.stringify cannot serialize cyclic structures.")
		}
		active[identity] = true
		defer delete(active, identity)
	}
	var out bytes.Buffer
	if value.Kind() == reflect.Map {
		keys := value.MapKeys()
		// Go maps do not carry insertion order. Keep deterministic lexical
		// order for ordinary keys and JS's numeric order for array-index keys.
		slices.SortFunc(keys, func(a, b reflect.Value) int {
			left, right := a.String(), b.String()
			x, ix := jsArrayIndex(left)
			y, iy := jsArrayIndex(right)
			if ix && iy {
				return cmp.Compare(x, y)
			}
			if ix {
				return -1
			}
			if iy {
				return 1
			}
			return cmp.Compare(left, right)
		})
		out.WriteByte('{')
		written := false
		for _, key := range keys {
			value := value.MapIndex(key)
			if !value.IsValid() {
				continue // An earlier serializer deleted this captured key.
			}
			if written {
				out.WriteByte(',')
			}
			written = true
			name, _ := marshalJSScalar(key.String())
			child, err := marshalProxyOption(value, active)
			if err != nil {
				return nil, err
			}
			out.Write(name)
			out.WriteByte(':')
			out.Write(child)
		}
		out.WriteByte('}')
	} else {
		out.WriteByte('[')
		for i := 0; i < value.Len(); i++ {
			if i > 0 {
				out.WriteByte(',')
			}
			child, err := marshalProxyOption(value.Index(i), active)
			if err != nil {
				return nil, err
			}
			out.Write(child)
		}
		out.WriteByte(']')
	}
	return out.Bytes(), nil
}

func proxyNumberJSON(number float64) ([]byte, error) {
	if math.IsNaN(number) || math.IsInf(number, 0) {
		return []byte("null"), nil
	}
	return marshalJSScalar(number)
}
