package telemetry

import (
	"bytes"
	"encoding/json"
	"strconv"

	"github.com/2found/2ai/jsonjs"
)

// The field descriptors are immutable after publication, so detached error
// copies may share them. Nested object/array values remain live shared references.
// Native string edits override a decoded value when its projection changes.
type errorDetailsEncoding struct {
	name, message         any
	nameText, messageText string
}

// NameValue and MessageValue expose the actual JS-shaped fields, including
// Undefined for absent fields and shared *Object/*Array values. Name/Message
// remain convenient string projections for ordinary Go errors.
func (e ErrorDetails) NameValue() any {
	if e.encoding != nil && e.Name == e.encoding.nameText {
		return e.encoding.name
	}
	return e.Name
}
func (e ErrorDetails) MessageValue() any {
	if e.encoding != nil && e.Message == e.encoding.messageText {
		return e.encoding.message
	}
	return e.Message
}

// SetNameValue and SetMessageValue replace only this error's field. They also
// distinguish explicit empty strings, null and Undefined from an unchanged Go
// zero value. Nested edits through the returned object/array stay shared.
func (e *ErrorDetails) SetNameValue(value any) {
	copy := &errorDetailsEncoding{name: value, message: e.MessageValue(), messageText: e.Message}
	e.Name, _ = value.(string)
	copy.nameText = e.Name
	e.encoding = copy
}
func (e *ErrorDetails) SetMessageValue(value any) {
	copy := &errorDetailsEncoding{name: e.NameValue(), message: value, nameText: e.Name}
	e.Message, _ = value.(string)
	copy.messageText = e.Message
	e.encoding = copy
}

func (e *ErrorDetails) UnmarshalJSON(data []byte) error {
	fields, _, err := statusObject(data)
	if err != nil {
		return err
	}
	decoded := ErrorDetails{}
	for _, field := range []struct {
		name string
		set  func(any)
	}{{"name", decoded.SetNameValue}, {"message", decoded.SetMessageValue}} {
		var value any = Undefined
		if raw := fields[field.name]; raw != nil {
			value, err = jsonjs.DecodeValue(raw)
			if err != nil {
				return err
			}
		}
		field.set(value)
	}
	*e = decoded
	return nil
}

func (e ErrorDetails) MarshalJSON() ([]byte, error) {
	return e.jsonValue().MarshalJSON()
}

func (e ErrorDetails) jsonValue() *Object {
	return NewObject(Property{Name: "name", Value: e.NameValue()}, Property{Name: "message", Value: e.MessageValue()})
}

// UnmarshalJSON projects serialized status inputs using Pi's property reads.
// Non-object primitives have no status/error properties; null is unreadable and
// SetStatus ignores it. Truthy errors copy only their name/message properties.
func (s *SpanStatus) UnmarshalJSON(data []byte) error {
	fields, null, err := statusObject(data)
	if err != nil {
		return err
	}
	decoded := SpanStatus{nullInput: null}
	_ = json.Unmarshal(fields["status"], &decoded.Status)
	if decoded.Status != "ok" && statusTruthy(fields["error"]) {
		decoded.Error = new(ErrorDetails)
		if err := json.Unmarshal(fields["error"], decoded.Error); err != nil {
			return err
		}
	}
	*s = decoded
	return nil
}

func (s SpanStatus) MarshalJSON() ([]byte, error) {
	return jsonjs.MarshalValue(s.jsonValue())
}

func (s SpanStatus) jsonValue() any {
	if s.nullInput && s.Status == "" && s.Error == nil {
		return nil
	}
	value := NewObject(Property{Name: "status", Value: s.Status})
	if s.Error != nil {
		value.Set("error", s.Error.jsonValue())
	}
	return value
}

func statusObject(data []byte) (map[string]json.RawMessage, bool, error) {
	var raw json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, false, err
	}
	raw = bytes.TrimSpace(raw)
	var fields map[string]json.RawMessage
	if len(raw) > 0 && raw[0] == '{' {
		if err := json.Unmarshal(raw, &fields); err != nil {
			return nil, false, err
		}
	}
	return fields, bytes.Equal(raw, []byte("null")), nil
}

func statusTruthy(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return false
	}
	switch raw[0] {
	case 'n', 'f':
		return false
	case '"':
		var value string
		_ = json.Unmarshal(raw, &value)
		return value != ""
	case '{', '[', 't':
		return true
	default:
		value, _ := strconv.ParseFloat(string(raw), 64)
		return value != 0
	}
}
