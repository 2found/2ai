package telemetry

import (
	"encoding/json"

	"github.com/2found/2ai/jsonjs"
)

func (o SpanOptions) MarshalJSON() ([]byte, error) {
	value := NewObject(Property{Name: "name", Value: o.Name})
	if !o.Attributes.IsZero() {
		value.Set("attributes", o.Attributes)
	}
	return jsonjs.MarshalValue(value)
}

func (o *SpanOptions) UnmarshalJSON(raw []byte) error {
	type plain SpanOptions
	var decoded plain
	if err := decodeRecordName(raw, &decoded, &decoded.Name); err != nil {
		return err
	}
	*o = SpanOptions(decoded)
	return nil
}

func (e *RecordedEvent) UnmarshalJSON(raw []byte) error {
	type plain RecordedEvent
	var decoded plain
	if err := decodeRecordName(raw, &decoded, &decoded.Name); err != nil {
		return err
	}
	*e = RecordedEvent(decoded)
	return nil
}

func (s *RecordedSpan) UnmarshalJSON(raw []byte) error {
	type plain RecordedSpan
	var decoded plain
	if err := decodeRecordName(raw, &decoded, &decoded.Name); err != nil {
		return err
	}
	*s = RecordedSpan(decoded)
	return nil
}

// The typed envelope still uses Go's field validation. Read the name with the
// shared UTF-16 codec so lone surrogate units survive admission and snapshot
// imports just as they do for attributes and error details.
func decodeRecordName(raw []byte, target any, name *string) error {
	if err := json.Unmarshal(raw, target); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	if encoded, ok := fields["name"]; ok {
		value, err := jsonjs.DecodeJSON(encoded)
		if err != nil {
			return err
		}
		if text, ok := value.(string); ok {
			*name = text
		}
	}
	return nil
}

// Build the snapshot's typed envelope as a shared JSON graph so metadata hooks
// see their real containing key (attributes/name/message), not a false root key.
func (e RecordedEvent) jsonValue() *Object {
	return NewObject(Property{Name: "name", Value: e.Name}, Property{Name: "attributes", Value: e.Attributes})
}

func (e RecordedEvent) MarshalJSON() ([]byte, error) { return jsonjs.MarshalValue(e.jsonValue()) }

func (s RecordedSpan) MarshalJSON() ([]byte, error) {
	var parent, events any
	if s.ParentID != nil {
		parent = *s.ParentID
	}
	if s.Events != nil {
		array := NewArray()
		for _, event := range s.Events {
			array.Append(event.jsonValue())
		}
		events = array
	}
	value := NewObject(Property{Name: "id", Value: s.ID}, Property{Name: "parentId", Value: parent},
		Property{Name: "name", Value: s.Name}, Property{Name: "attributes", Value: s.Attributes},
		Property{Name: "events", Value: events}, Property{Name: "status", Value: s.Status.jsonValue()},
		Property{Name: "settled", Value: s.Settled})
	if s.EndSequence != nil {
		value.Set("endSequence", *s.EndSequence)
	}
	return jsonjs.MarshalValue(value)
}
