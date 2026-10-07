package ai

// This file ports Pi api/constrained-sampling.ts. It operates on the native
// Tool contract; legacy tool-schema adaptation remains separate until callers
// migrate to native provider requests. See LICENSE.pi and port-status.json.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// UnsupportedStrictSchemaKeywordCheck adds a provider-specific strict-schema
// restriction. The value retains its original JSON, including opaque numbers.
type UnsupportedStrictSchemaKeywordCheck func(string, json.RawMessage) bool

type UnsupportedStrictJSONSchemaError struct{ Reason string }

func (e *UnsupportedStrictJSONSchemaError) Error() string { return e.Reason }
func unsupportedStrict(reason string) error               { return &UnsupportedStrictJSONSchemaError{Reason: reason} }

var unsupportedStrictSchemaKeys = []string{"$ref", "$defs", "definitions", "allOf", "oneOf", "patternProperties", "dependentSchemas", "dependencies", "unevaluatedProperties", "propertyNames", "contains", "prefixItems", "not", "if", "then", "else"}

// MakeStrictJSONSchema ports Pi's constrained-sampling conversion. It returns a
// detached schema, requires every object property, and makes optional fields
// nullable. Unsupported constructs fail instead of being silently discarded.
func MakeStrictJSONSchema(schema json.RawMessage, check UnsupportedStrictSchemaKeywordCheck) (json.RawMessage, error) {
	if _, ok := samplingObject(schema); !ok {
		return nil, unsupportedStrict("root schema must have type object")
	}
	result, err := makeStrictSchemaNode(schema, check)
	if err != nil {
		return nil, err
	}
	object, _ := samplingObject(result)
	if samplingString(object["type"]) != "object" {
		return nil, unsupportedStrict("root schema must have type object")
	}
	return result, nil
}

func makeStrictSchemaNode(raw json.RawMessage, check UnsupportedStrictSchemaKeywordCheck) (json.RawMessage, error) {
	schema, ok := samplingObject(raw)
	if !ok {
		return nil, unsupportedStrict("boolean schemas are unsupported")
	}
	for _, key := range unsupportedStrictSchemaKeys {
		if _, ok := schema[key]; ok {
			return nil, unsupportedStrict(key + " schemas are unsupported")
		}
	}
	if check != nil {
		for _, key := range samplingObjectKeys(raw) {
			if check(key, schema[key]) {
				var compact bytes.Buffer
				_ = json.Compact(&compact, schema[key])
				return nil, unsupportedStrict(key + ": " + compact.String() + " is unsupported")
			}
		}
	}
	if raw, exists := schema["anyOf"]; exists {
		var variants []json.RawMessage
		if json.Unmarshal(raw, &variants) != nil || len(variants) == 0 {
			return nil, unsupportedStrict("anyOf must contain at least one schema")
		}
		for i, variant := range variants {
			if isStructuredSamplingSchema(variant) {
				return nil, unsupportedStrict("object and array unions are unsupported")
			}
			next, err := makeStrictSchemaNode(variant, check)
			if err != nil {
				return nil, err
			}
			variants[i] = next
		}
		schema["anyOf"], _ = json.Marshal(variants)
	}
	if items, exists := schema["items"]; exists {
		if bytes.HasPrefix(bytes.TrimSpace(items), []byte("[")) {
			return nil, unsupportedStrict("tuple schemas are unsupported")
		}
		next, err := makeStrictSchemaNode(items, check)
		if err != nil {
			return nil, err
		}
		schema["items"] = next
	}
	isObject := samplingString(schema["type"]) == "object"
	if _, exists := schema["properties"]; exists && !isObject {
		return nil, unsupportedStrict("properties require type object")
	}
	if !isObject {
		return marshalSamplingObject(schema, samplingObjectKeys(raw)), nil
	}
	if value, exists := schema["additionalProperties"]; exists && string(bytes.TrimSpace(value)) != "false" {
		return nil, unsupportedStrict("schema-valued or true additionalProperties is unsupported")
	}
	properties := map[string]json.RawMessage{}
	propertyNames := []string{}
	if value, exists := schema["properties"]; exists {
		var ok bool
		properties, ok = samplingObject(value)
		if !ok {
			return nil, unsupportedStrict("object properties must be a schema map")
		}
		propertyNames = samplingObjectKeys(value)
	}
	required := []string{}
	if value, exists := schema["required"]; exists {
		var fields []json.RawMessage
		if json.Unmarshal(value, &fields) != nil || fields == nil {
			return nil, unsupportedStrict("object required must be a string array")
		}
		for _, field := range fields {
			var name string
			if json.Unmarshal(field, &name) != nil || string(field) == "null" {
				return nil, unsupportedStrict("object required must be a string array")
			}
			required = append(required, name)
		}
	}
	requiredSet := map[string]bool{}
	for _, name := range required {
		if _, ok := properties[name]; !ok {
			return nil, unsupportedStrict("required contains an unknown property")
		}
		requiredSet[name] = true
	}
	for _, name := range propertyNames {
		property, err := makeStrictSchemaNode(properties[name], check)
		if err != nil {
			return nil, err
		}
		if !requiredSet[name] && !samplingSchemaAllowsNull(property) {
			property, _ = json.Marshal(map[string]any{"anyOf": []json.RawMessage{property, json.RawMessage(`{"type":"null"}`)}})
		}
		properties[name] = property
	}
	// Pi leaves an absent properties field absent, even for an empty object.
	if _, exists := schema["properties"]; exists {
		schema["properties"] = marshalSamplingObject(properties, propertyNames)
	}
	schema["required"], _ = json.Marshal(propertyNames)
	schema["additionalProperties"] = json.RawMessage(`false`)
	return marshalSamplingObject(schema, samplingObjectKeys(raw)), nil
}

func isStructuredSamplingSchema(raw json.RawMessage) bool {
	schema, ok := samplingObject(raw)
	if !ok {
		return false
	}
	if _, ok := schema["properties"]; ok {
		return true
	}
	if _, ok := schema["items"]; ok {
		return true
	}
	typ := samplingString(schema["type"])
	if typ == "object" || typ == "array" {
		return true
	}
	var types []string
	_ = json.Unmarshal(schema["type"], &types)
	for _, typ := range types {
		if typ == "object" || typ == "array" {
			return true
		}
	}
	return false
}
func samplingSchemaAllowsNull(raw json.RawMessage) bool {
	schema, ok := samplingObject(raw)
	if !ok {
		return false
	}
	if samplingString(schema["type"]) == "null" {
		return true
	}
	var types []string
	_ = json.Unmarshal(schema["type"], &types)
	for _, typ := range types {
		if typ == "null" {
			return true
		}
	}
	if value, exists := schema["const"]; exists && string(bytes.TrimSpace(value)) == "null" {
		return true
	}
	var enum []json.RawMessage
	_ = json.Unmarshal(schema["enum"], &enum)
	for _, value := range enum {
		if string(bytes.TrimSpace(value)) == "null" {
			return true
		}
	}
	var variants []json.RawMessage
	_ = json.Unmarshal(schema["anyOf"], &variants)
	for _, variant := range variants {
		if samplingSchemaAllowsNull(variant) {
			return true
		}
	}
	return false
}

// GetJSONSchemaToolParameters only applies strict conversion when explicitly
// selected. Otherwise the original parameter bytes remain authoritative.
func GetJSONSchemaToolParameters(tool Tool, strict *bool) (json.RawMessage, error) {
	if strict != nil && *strict {
		return MakeStrictJSONSchema(tool.Parameters, nil)
	}
	return tool.Parameters, nil
}
func ResolveJSONSchemaStrictSampling(tool Tool, supported bool, check UnsupportedStrictSchemaKeywordCheck) (*bool, error) {
	config, _ := samplingObject(tool.ConstrainedSampling)
	if samplingString(config["type"]) != "json_schema" {
		return nil, nil
	}
	require := samplingString(config["strict"]) == "require"
	if supported {
		_, err := MakeStrictJSONSchema(tool.Parameters, check)
		if err == nil {
			strict := true
			return &strict, nil
		}
		var unsupported *UnsupportedStrictJSONSchemaError
		if !errors.As(err, &unsupported) {
			return nil, err
		}
		if !require {
			return nil, nil
		}
		return nil, fmt.Errorf("Tool \"%s\" requires JSON-schema constrained sampling, but %s.", tool.Name, err.Error())
	}
	if require {
		return nil, fmt.Errorf("Tool \"%s\" requires JSON-schema constrained sampling, but strict tools are unsupported.", tool.Name)
	}
	return nil, nil
}

type GrammarConstrainedSampling struct {
	Format        string `json:"format"`
	Definition    string `json:"definition"`
	InputProperty string `json:"inputProperty"`
}

func ResolveGrammarConstrainedSampling(tool Tool, supported bool) (*GrammarConstrainedSampling, error) {
	config, _ := samplingObject(tool.ConstrainedSampling)
	if !supported || samplingString(config["type"]) != "grammar" {
		return nil, nil
	}
	variants, _ := samplingObject(config["variants"])
	lark, regex := samplingString(variants["openai_lark"]), samplingString(variants["openai_regex"])
	format, definition := "lark", lark
	if strings.TrimFunc(lark, jsWhitespace) == "" {
		format, definition = "regex", regex
	}
	if strings.TrimFunc(definition, jsWhitespace) == "" {
		return nil, fmt.Errorf("Tool \"%s\" cannot use grammar constrained sampling: no supported grammar variant was provided.", tool.Name)
	}
	property, err := inferGrammarInputProperty(tool.Parameters)
	if err != nil {
		return nil, fmt.Errorf("Tool \"%s\" cannot use grammar constrained sampling: %s.", tool.Name, err.Error())
	}
	return &GrammarConstrainedSampling{Format: format, Definition: definition, InputProperty: property}, nil
}
func inferGrammarInputProperty(raw json.RawMessage) (string, error) {
	schema, _ := samplingObject(raw)
	if samplingString(schema["type"]) != "object" {
		return "", errors.New("grammar constrained sampling requires an object parameter schema")
	}
	var required []json.RawMessage
	_ = json.Unmarshal(schema["required"], &required)
	var name string
	if len(required) != 1 || json.Unmarshal(required[0], &name) != nil || string(required[0]) == "null" {
		return "", errors.New("grammar constrained sampling requires exactly one required string property")
	}
	properties, _ := samplingObject(schema["properties"])
	value := bytes.TrimSpace(properties[name])
	if len(value) == 0 || string(value) == "null" || string(value) == "false" || string(value) == "0" || string(value) == `""` {
		return "", fmt.Errorf("grammar constrained sampling requires a properties entry for %s", name)
	}
	property, _ := samplingObject(value)
	if samplingString(property["type"]) != "string" {
		return "", fmt.Errorf("grammar constrained sampling property %s must have type string", name)
	}
	return name, nil
}
func CreateGrammarToolInputProperties(tools []Tool, supported bool) (map[string]string, error) {
	result := map[string]string{}
	for _, tool := range tools {
		grammar, err := ResolveGrammarConstrainedSampling(tool, supported)
		if err != nil {
			return nil, err
		}
		if grammar != nil {
			result[tool.Name] = grammar.InputProperty
		}
	}
	return result, nil
}
func GetGrammarToolInput(toolName string, arguments json.RawMessage, property string) (string, error) {
	object, _ := samplingObject(arguments)
	var value string
	raw := object[property]
	if len(raw) == 0 || string(bytes.TrimSpace(raw)) == "null" || json.Unmarshal(raw, &value) != nil {
		return "", fmt.Errorf("Grammar tool call \"%s\" requires argument \"%s\" to be a string.", toolName, property)
	}
	return value, nil
}

type GrammarToolInputJSONBuffer struct {
	Input   string `json:"input"`
	Started bool   `json:"started"`
	Closed  bool   `json:"closed"`
}

// AppendGrammarToolInputJSONDelta returns nil for an unchanged open input or a
// duplicate close. A non-nil delta is an exact JSON string fragment.
func AppendGrammarToolInputJSONDelta(buffer *GrammarToolInputJSONBuffer, property, next string, close bool) (*string, error) {
	if buffer.Closed {
		if close && next == buffer.Input {
			return nil, nil
		}
		return nil, fmt.Errorf("grammar tool input for property \"%s\" changed after it was closed", property)
	}
	if !strings.HasPrefix(next, buffer.Input) {
		return nil, fmt.Errorf("grammar tool input for property \"%s\" changed non-monotonically", property)
	}
	suffix := next[len(buffer.Input):]
	if !close && suffix == "" {
		return nil, nil
	}
	delta := ""
	if !buffer.Started {
		delta = "{" + marshalSamplingString(property) + ":" + `"`
		buffer.Started = true
	}
	encoded := marshalSamplingString(suffix)
	delta += encoded[1 : len(encoded)-1]
	buffer.Input = next
	if close {
		delta += `"}`
		buffer.Closed = true
	}
	return &delta, nil
}

func samplingObject(raw json.RawMessage) (map[string]json.RawMessage, bool) {
	var object map[string]json.RawMessage
	err := json.Unmarshal(raw, &object)
	return object, err == nil && object != nil
}
func samplingString(raw json.RawMessage) string {
	var value string
	_ = json.Unmarshal(raw, &value)
	return value
}
func samplingObjectKeys(raw json.RawMessage) []string {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	_, _ = decoder.Token()
	keys := []string{}
	seen := map[string]bool{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			break
		}
		name, ok := token.(string)
		if !ok {
			break
		}
		var value json.RawMessage
		if decoder.Decode(&value) != nil {
			break
		}
		if !seen[name] {
			keys = append(keys, name)
			seen[name] = true
		}
	}
	index := func(name string) (uint64, bool) {
		n, err := strconv.ParseUint(name, 10, 32)
		return n, err == nil && n < 4294967295 && strconv.FormatUint(n, 10) == name
	}
	sort.SliceStable(keys, func(i, j int) bool {
		a, ai := index(keys[i])
		b, bi := index(keys[j])
		if ai && bi {
			return a < b
		}
		return ai && !bi
	})
	return keys
}
func marshalSamplingString(value string) string {
	// JSON.stringify leaves HTML and U+2028/U+2029 intact. Build directly from
	// the string so a literal backslash-u sequence cannot be mistaken for them.
	var out strings.Builder
	out.WriteByte('"')
	for _, r := range value {
		switch r {
		case '"', '\\':
			out.WriteByte('\\')
			out.WriteRune(r)
		case '\b':
			out.WriteString(`\b`)
		case '\f':
			out.WriteString(`\f`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&out, `\u%04x`, r)
			} else {
				out.WriteRune(r)
			}
		}
	}
	out.WriteByte('"')
	return out.String()
}

func marshalSamplingObject(fields map[string]json.RawMessage, keys []string) json.RawMessage {
	seen := map[string]bool{}
	ordered := append([]string(nil), keys...)
	for _, key := range keys {
		seen[key] = true
	}
	// These are the only fields introduced by strict conversion, in Pi order.
	for _, key := range []string{"required", "additionalProperties"} {
		if _, ok := fields[key]; ok && !seen[key] {
			ordered = append(ordered, key)
			seen[key] = true
		}
	}
	var out bytes.Buffer
	out.WriteByte('{')
	for i, key := range ordered {
		if i > 0 {
			out.WriteByte(',')
		}
		out.WriteString(marshalSamplingString(key))
		out.WriteByte(':')
		out.Write(fields[key])
	}
	out.WriteByte('}')
	return out.Bytes()
}
