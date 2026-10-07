package telemetry

import "encoding/json"

// AttributeDefinition is the serializable schema metadata from Pi. Values,
// ElementValues and Examples retain the JSON shape of their selected Type:
// scalar examples are arrays, array examples are arrays of arrays. Metadata
// describes instrumentation; it never causes runtime validation or redaction.
type AttributeDefinition struct {
	Type          string          `json:"type"`
	Description   string          `json:"description"`
	Sensitive     *bool           `json:"sensitive,omitempty"`
	Cardinality   string          `json:"cardinality,omitempty"`
	Values        json.RawMessage `json:"values,omitempty"`
	ElementValues json.RawMessage `json:"elementValues,omitempty"`
	Examples      json.RawMessage `json:"examples,omitempty"`
}

type StartAttributeDefinition struct {
	AttributeDefinition
	Required bool `json:"required"`
}

type EventAttributeDefinition = StartAttributeDefinition

type EventDefinition struct {
	Description string                              `json:"description"`
	Attributes  map[string]EventAttributeDefinition `json:"attributes"`
}

type ParentDefinition struct {
	Kind  string   `json:"kind"`
	Spans []string `json:"spans,omitempty"`
}

// An explicit empty spans list is distinct from an absent list in the union.
func (p ParentDefinition) MarshalJSON() ([]byte, error) {
	if p.Kind == "spans" {
		names := p.Spans
		if names == nil {
			names = []string{}
		}
		return json.Marshal(struct {
			Kind  string   `json:"kind"`
			Spans []string `json:"spans"`
		}{p.Kind, names})
	}
	return json.Marshal(struct {
		Kind string `json:"kind"`
	}{p.Kind})
}

type StatusDefinition struct {
	Default   string `json:"default"`
	ErrorWhen string `json:"errorWhen"`
}

type SpanDefinition struct {
	Description     string                              `json:"description"`
	Parents         ParentDefinition                    `json:"parents"`
	StartAttributes map[string]StartAttributeDefinition `json:"startAttributes"`
	EndAttributes   map[string]AttributeDefinition      `json:"endAttributes"`
	Events          map[string]EventDefinition          `json:"events,omitempty"`
	Status          StatusDefinition                    `json:"status"`
}

func (s SpanDefinition) MarshalJSON() ([]byte, error) {
	type plain SpanDefinition
	var events *map[string]EventDefinition
	if s.Events != nil {
		events = &s.Events
	}
	return json.Marshal(struct {
		plain
		Events *map[string]EventDefinition `json:"events,omitempty"`
	}{plain(s), events})
}

type SchemaDefinition struct {
	Version float64                   `json:"version"`
	Spans   map[string]SpanDefinition `json:"spans"`
}

// DefineSchema is Pi's identity helper. No cloning, normalization, validation,
// or schema access occurs; the same pointer is returned.
func DefineSchema(schema *SchemaDefinition) *SchemaDefinition { return schema }

// SpanStarter binds one explicit parent context. Schema values are deliberately
// unread, as in createTypedSpanStarter. Go does not infer literal attribute
// types from schema values; use application-owned typed wrappers when needed.
// It must not introduce runtime validation absent from Pi's implementation.
type SpanStarter struct{ context Context }

func CreateTypedSpanStarter(parent Context, _ ...*SchemaDefinition) SpanStarter {
	return SpanStarter{context: parent}
}

// StartSpan supplies a child starter bound to the newly admitted span. Saving
// that starter does not extend the span's lifetime; late children remain no-op.
func (s SpanStarter) StartSpan(name string, attributes Attributes, callback func(*Span, SpanStarter) error) error {
	return s.context.StartSpan(SpanOptions{Name: name, Attributes: attributes}, func(span *Span) error {
		return callback(span, SpanStarter{context: span.Context()})
	})
}

// StartTypedSpan preserves a generic result without boxing it into an interface.
func StartTypedSpan[T any](starter SpanStarter, name string, attributes Attributes, callback func(*Span, SpanStarter) (T, error)) (value T, err error) {
	err = starter.StartSpan(name, attributes, func(span *Span, children SpanStarter) error {
		var callbackErr error
		value, callbackErr = callback(span, children)
		return callbackErr
	})
	return value, err
}
