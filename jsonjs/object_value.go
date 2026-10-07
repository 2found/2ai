package jsonjs

import "encoding/json"

// ObjectValue is a rebindable reference to an ordered object. Value copies
// share the object; decoding replaces this reference without changing aliases.
type ObjectValue struct{ object *Object }

// NewObjectValue preserves the order of the supplied properties. Its copies
// share the input object.
func NewObjectValue(properties ...Property) ObjectValue {
	return ObjectValue{object: NewObject(properties...)}
}
func (a ObjectValue) Get(name string) any            { return a.object.Get(name) }
func (a ObjectValue) Lookup(name string) (any, bool) { return a.object.Lookup(name) }
func (a ObjectValue) Len() int                       { return a.object.Len() }
func (a ObjectValue) IsZero() bool                   { return a.Len() == 0 }
func (a ObjectValue) Entries() []Property            { return a.object.Entries() }
func (a *ObjectValue) Set(name string, value any) {
	if a.object == nil {
		a.object = NewObject()
	}
	a.object.Set(name, value)
}
func (a ObjectValue) Delete(name string) { a.object.Delete(name) }

func (o ObjectValue) MarshalJSON() ([]byte, error) { return MarshalValue(o) }

// DecodeValue constructs live ordered objects and reference arrays from JSON.
func DecodeValue(raw []byte) (any, error) {
	return DecodeJSONContainers(raw, func(properties []Property) any { return NewObject(properties...) }, func(values []any) any { return NewArray(values...) })
}

// UnmarshalJSON reads an object using JSON.parse's number domain.
// Valid numeric overflow becomes infinity, underflow keeps its sign, and all
// numbers round to binary64. Export remains the responsibility of MarshalJSON;
// converting overflow to null here would lose the recorded in-memory value.
// Lone UTF-16 surrogates use WTF-8 within Go strings, including map keys;
// ObjectValue.MarshalJSON restores their original JSON escapes.
// Objects and arrays use *Object and *Array, retaining key order and shared
// nested identity, including array length edits through retained references.
// Each successful decode replaces the object, including on a reused receiver.
// A rejected payload leaves the receiver and any aliases untouched.
func (object *ObjectValue) UnmarshalJSON(raw []byte) error {
	value, err := DecodeValue(raw)
	if err != nil {
		return err
	}
	switch value := value.(type) {
	case nil:
		*object = ObjectValue{}
	case *Object:
		*object = ObjectValue{object: value}
	default:
		// Retain the typed object boundary and Go's diagnostics.
		var rejected map[string]any
		return json.Unmarshal(raw, &rejected)
	}
	return nil
}
