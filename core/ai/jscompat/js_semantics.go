package jscompat

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"

	"github.com/2found/2ai/core/jsonjs"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

func Values(value any) ([]any, error) {
	if jsonjs.IsNullish(value) {
		return nil, errors.New("Object.values requires that input parameter not be null or undefined")
	}
	values := []any{}
	switch value := value.(type) {
	case *jsonjs.Object:
		for _, property := range value.Entries() {
			values = append(values, property.Value)
		}
	case *jsonjs.Array:
		for _, key := range value.PropertyKeys() {
			entry, _ := value.GetProperty(key)
			values = append(values, entry)
		}
	case string:
		for _, point := range jsonjs.StringCodePoints(value) {
			if point > 0xffff {
				point -= 0x10000
				values = append(values, CodeUnit(0xd800+point>>10), CodeUnit(0xdc00+(point&1023)))
			} else {
				values = append(values, CodeUnit(point))
			}
		}
	}
	return values, nil
}

func CodeUnit(point rune) string {
	if point >= 0xd800 && point <= 0xdfff {
		return string([]byte{byte(0xe0 | point>>12), byte(0x80 | point>>6&63), byte(0x80 | point&63)})
	}
	return string(point)
}

func ReadProperty(value any, name string) any {
	switch value := value.(type) {
	case *jsonjs.Object:
		if property, exists := value.Lookup(name); exists {
			return property
		}
	case *jsonjs.Array:
		if property, exists := value.GetProperty(name); exists {
			return property
		}
	}
	return jsonjs.Undefined
}

// Catalog IDs normally are strings. These conversions retain the observable
// behavior of JSON-shaped malformed catalogs without invoking serializers.
func ToPropertyKey(value any, visiting map[*jsonjs.Array]bool) (string, error) {
	if jsonjs.IsUndefined(value) {
		return "undefined", nil
	}
	if value == nil || jsonjs.IsNull(value) {
		return "null", nil
	}
	switch value := value.(type) {
	case string:
		return value, nil
	case *jsonjs.Object:
		if _, exists := value.Lookup("toString"); exists {
			return "", errors.New("No default value")
		}
		return "[object Object]", nil
	case *jsonjs.Array:
		if _, exists := value.GetProperty("toString"); exists {
			return "", errors.New("No default value")
		}
		if _, exists := value.GetProperty("join"); exists {
			return "[object Array]", nil
		}
		if visiting[value] {
			return "", nil
		}
		visiting[value] = true
		defer delete(visiting, value)
		parts := make([]string, value.Len())
		for i := range parts {
			item := value.Get(i)
			if jsonjs.IsNullish(item) {
				continue
			}
			var err error
			parts[i], err = ToPropertyKey(item, visiting)
			if err != nil {
				return "", err
			}
		}
		return strings.Join(parts, ","), nil
	case float64:
		if math.IsNaN(value) {
			return "NaN", nil
		}
		if math.IsInf(value, 1) {
			return "Infinity", nil
		}
		if math.IsInf(value, -1) {
			return "-Infinity", nil
		}
	}
	raw, err := jsonjs.StringifyValue(value)
	return string(raw), err
}

func Truthy(value any) bool {
	if jsonjs.IsNullish(value) {
		return false
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Bool:
		return v.Bool()
	case reflect.String:
		return v.Len() != 0
	case reflect.Float32, reflect.Float64:
		return v.Float() != 0 && !math.IsNaN(v.Float())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() != 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return v.Uint() != 0
	case reflect.Pointer:
		return !v.IsNil()
	}
	return true
}

func StrictEqual(a, b any) bool {
	if jsonjs.IsUndefined(a) || jsonjs.IsUndefined(b) {
		return jsonjs.IsUndefined(a) && jsonjs.IsUndefined(b)
	}
	if a == nil || b == nil || jsonjs.IsNull(a) || jsonjs.IsNull(b) {
		return (a == nil || jsonjs.IsNull(a)) && (b == nil || jsonjs.IsNull(b))
	}
	number := func(value any) (float64, bool) {
		v := reflect.ValueOf(value)
		switch v.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return float64(v.Int()), true
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return float64(v.Uint()), true
		case reflect.Float32, reflect.Float64:
			return v.Float(), true
		}
		return 0, false
	}
	if an, ok := number(a); ok {
		bn, ok := number(b)
		return ok && an == bn
	}
	return reflect.TypeOf(a) == reflect.TypeOf(b) && reflect.TypeOf(a).Comparable() && a == b
}

func SameValueZero(a, b any) bool {
	if StrictEqual(a, b) {
		return true
	}
	isNaN := func(value any) bool {
		if value == nil {
			return false
		}
		v := reflect.ValueOf(value)
		return (v.Kind() == reflect.Float64 || v.Kind() == reflect.Float32) && math.IsNaN(v.Float())
	}
	return isNaN(a) && isNaN(b)
}

func ToNumber(value any) (float64, error) {
	if jsonjs.IsUndefined(value) {
		return math.NaN(), nil
	}
	if value == nil || jsonjs.IsNull(value) {
		return 0, nil
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Bool:
		if v.Bool() {
			return 1, nil
		}
		return 0, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(v.Int()), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return float64(v.Uint()), nil
	case reflect.Float32, reflect.Float64:
		return v.Float(), nil
	}
	text, err := ToPropertyKey(value, make(map[*jsonjs.Array]bool))
	return ParseJSNumber(text), err
}

func Spread(value any) *jsonjs.Object {
	object := jsonjs.NewObject()
	SpreadInto(object, value)
	return object
}

func SpreadInto(target *jsonjs.Object, value any) {
	switch value := value.(type) {
	case *jsonjs.Object:
		for _, property := range value.Entries() {
			target.Set(property.Name, property.Value)
		}
	case *jsonjs.Array:
		for _, key := range value.PropertyKeys() {
			item, _ := value.GetProperty(key)
			target.Set(key, item)
		}
	case string:
		values, _ := Values(value)
		for index, item := range values {
			target.Set(fmt.Sprint(index), item)
		}
	}
}

func Invoke(callback func() (any, error)) (value any, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			var ok bool
			err, ok = recovered.(error)
			if !ok {
				err = fmt.Errorf("%v", recovered)
			}
		}
	}()
	return callback()
}

func LowerHeaderName(name string) string {
	var result, segment strings.Builder
	flush := func() {
		result.WriteString(cases.Lower(language.Und).String(segment.String()))
		segment.Reset()
	}
	for _, point := range jsonjs.StringCodePoints(name) {
		if point >= 0xd800 && point <= 0xdfff {
			flush()
			result.WriteString(CodeUnit(point))
		} else {
			segment.WriteRune(point)
		}
	}
	flush()
	return result.String()
}

func ReadModelProperty(model any, key string) (any, error) {
	if jsonjs.IsNullish(model) {
		return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(model), "model."+key)
	}
	return ReadProperty(model, key), nil
}
