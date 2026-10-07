package jsonjs

import (
	"bytes"
	"encoding/json"
	"slices"
)

// RawObject is a JSON object whose keys retain UTF-16 identity and whose values
// stay serialized. It is for typed envelopes that must preserve opaque fields,
// without decoding their numbers or interpreting nested values. Like Go maps,
// it has no insertion order; export sorts keys for deterministic output.
type RawObject map[string]json.RawMessage

func (o RawObject) MarshalJSON() ([]byte, error) {
	if o == nil {
		return []byte("null"), nil
	}
	keys := make([]string, 0, len(o))
	for key := range o {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	var out bytes.Buffer
	out.WriteByte('{')
	for i, key := range keys {
		value, err := json.Marshal(o[key])
		if err != nil {
			return nil, err
		}
		if i > 0 {
			out.WriteByte(',')
		}
		out.Write(QuoteString(key))
		out.WriteByte(':')
		out.Write(value)
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

// Successful decoding replaces the object and owns its raw value bytes. Invalid
// input leaves the receiver untouched. Repeated keys keep their final value;
// distinct lone surrogate keys do not collapse into the replacement character.
func (o *RawObject) UnmarshalJSON(raw []byte) error {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '{' {
		var object map[string]json.RawMessage
		if err := json.Unmarshal(raw, &object); err != nil {
			return err
		}
		*o = RawObject(object)
		return nil
	}
	properties, err := DecodeObjectProperties(raw)
	if err != nil {
		return err
	}
	object := make(RawObject, len(properties))
	for _, property := range properties {
		object[property.Name] = bytes.Clone(property.Value)
	}
	*o = object
	return nil
}
