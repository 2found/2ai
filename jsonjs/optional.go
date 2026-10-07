package jsonjs

import "reflect"

type nullValue uint8

// Null represents explicit null at optional Go fields whose nil zero value
// means absent. Nested JSON null values continue to use ordinary nil.
const Null nullValue = 0

func (nullValue) MarshalJSON() ([]byte, error) { return []byte("null"), nil }

func IsNullish(value any) bool {
	return value == nil || IsUndefined(value) || IsNull(value)
}

func IsNull(value any) bool { _, ok := value.(nullValue); return ok }

// MarshalOptional exports an optional object property. nil/Undefined/functions
// are omitted; Null exports an explicit JSON null.
func MarshalOptional(value any, key ...string) ([]byte, error) {
	if value == nil || IsUndefined(value) || reflect.TypeOf(value).Kind() == reflect.Func {
		return nil, nil
	}
	property := ""
	if len(key) > 0 {
		property = key[0]
	}
	return StringifyProperty(value, property)
}

// DecodeOptional retains absence versus null at optional JSON-value fields.
func DecodeOptional(raw []byte) (any, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	value, err := DecodeValue(raw)
	if err != nil {
		return nil, err
	}
	if value == nil {
		return Null, nil
	}
	return value, nil
}
