package ai

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/2found/2ai/ai/protocol"
)

type toolSchemaDialect uint8

const (
	toolSchemaGeneric toolSchemaDialect = iota
	toolSchemaOpenAIResponses
	toolSchemaCloudCodeAssist
)

var cloudCodeUnsupportedSchemaFields = map[string]struct{}{
	"$schema": {}, "$defs": {}, "definitions": {}, "$dynamicRef": {}, "$dynamicAnchor": {},
	"examples": {}, "prefixItems": {}, "unevaluatedProperties": {},
	"unevaluatedItems": {}, "patternProperties": {}, "additionalProperties": {},
	"propertyNames": {}, "minItems": {}, "maxItems": {}, "minLength": {},
	"maxLength": {}, "minimum": {}, "maximum": {}, "exclusiveMinimum": {},
	"exclusiveMaximum": {}, "multipleOf": {}, "pattern": {}, "format": {},
	"dependencies": {}, "dependentSchemas": {}, "dependentRequired": {},
	"x-mcp-header": {}, "deprecated": {}, "readOnly": {}, "writeOnly": {},
	"$comment": {},
}

// OpenAI's constrained tool grammar supports only a deliberately small schema
// subset. Removing validation-only keywords widens what the model may emit, but
// never what AgentRay executes: the canonical schema is validated locally.
var openAIStrictUnsupportedSchemaFields = map[string]struct{}{
	"$schema": {}, "format": {}, "pattern": {}, "minLength": {}, "maxLength": {},
	"minimum": {}, "maximum": {}, "exclusiveMinimum": {}, "exclusiveMaximum": {},
	"minItems": {}, "maxItems": {}, "uniqueItems": {}, "multipleOf": {},
	"examples": {}, "default": {}, "title": {}, "$comment": {}, "if": {},
	"then": {}, "else": {}, "not": {}, "unevaluatedProperties": {},
	"unevaluatedItems": {}, "patternProperties": {}, "propertyNames": {},
	"contains": {}, "minContains": {}, "maxContains": {}, "dependentRequired": {},
	"dependentSchemas": {}, "contentEncoding": {}, "contentMediaType": {},
	"contentSchema": {}, "deprecated": {}, "readOnly": {}, "writeOnly": {},
	"minProperties": {}, "maxProperties": {}, "$dynamicRef": {}, "$dynamicAnchor": {},
}

// toolParameters returns a provider-safe copy of the canonical tool schema.
// The canonical map remains the execution-time contract used by agentcore's
// argument validator; widening a wire projection can therefore improve model
// compatibility without weakening what is allowed to execute.
func toolParameters(parameters map[string]any, dialect toolSchemaDialect) map[string]any {
	if len(parameters) == 0 {
		return emptyObjectToolSchema()
	}
	cloned, ok := cloneJSONMap(parameters)
	if !ok {
		// Preserve the old behavior for a non-JSON/cyclic schema: json.Marshal on
		// the request will return the useful authoring error rather than silently
		// advertising an unrelated permissive schema.
		return parameters
	}
	switch dialect {
	case toolSchemaOpenAIResponses:
		value := normalizeResponsesSchemaNode(cloned)
		if normalized, ok := value.(map[string]any); ok && len(normalized) > 0 {
			return ensureObjectProperties(normalized)
		}
		return emptyObjectToolSchema()
	case toolSchemaCloudCodeAssist:
		value, representable := normalizeCloudCodeSchemaNode(cloned)
		if normalized, ok := value.(map[string]any); representable && ok && len(normalized) > 0 && !hasResidualCloudCodeSchema(normalized) {
			return ensureObjectProperties(normalized)
		}
		// CCA rejects an entire request when one declaration contains an unknown
		// schema shape. A permissive object keeps the tool available; agentcore
		// still validates the emitted arguments against the canonical schema.
		return emptyObjectToolSchema()
	default:
		return ensureObjectProperties(cloned)
	}
}

// projectedToolParameters returns the provider-facing schema and the optional
// strict wire value. Strict enforcement is intentionally conservative: unlike
// adapters that make every optional property required-and-nullable, AgentRay
// refuses that rewrite because its canonical validator would then reject a
// generated null. A schema that cannot preserve its argument semantics simply
// runs non-strict and retains full local validation.
func projectedToolParameters(schema protocol.ToolSchema, dialect toolSchemaDialect) (map[string]any, *bool) {
	parameters := toolParameters(schema.Parameters, dialect)
	switch schema.Strict {
	case protocol.ToolStrictDisabled:
		value := false
		return parameters, &value
	case protocol.ToolStrictEnabled:
		strict, ok := cloneJSONMap(parameters)
		if !ok {
			return parameters, nil
		}
		// Strict grammar engines accept anyOf but commonly reject oneOf and
		// lookaround regexes even when their non-strict endpoint accepts them.
		normalized, ok := normalizeResponsesSchemaNode(strict).(map[string]any)
		if !ok || !enforceOpenAIStrictSchema(normalized) {
			return parameters, nil
		}
		value := true
		return normalized, &value
	default:
		return parameters, nil
	}
}

// enforceOpenAIStrictSchema mutates only a provider-facing clone. It returns
// false rather than emitting a partially strict schema when a construct cannot
// be represented safely.
func enforceOpenAIStrictSchema(schema map[string]any) bool {
	if len(schema) == 0 {
		return false
	}
	// Strict mode cannot represent an open object keyspace. Silently replacing
	// an explicit open-map contract with false would narrow valid tool inputs.
	if pattern, exists := schema["patternProperties"]; exists {
		if entries, ok := pattern.(map[string]any); !ok || len(entries) > 0 {
			return false
		}
	}
	additional, explicitlyClosed := schema["additionalProperties"]
	objectLike := schema["type"] == "object" || schema["properties"] != nil
	if (objectLike && (!explicitlyClosed || additional != false)) ||
		(!objectLike && explicitlyClosed && additional != false) {
		return false
	}
	for key := range schema {
		if _, unsupported := openAIStrictUnsupportedSchemaFields[key]; unsupported {
			delete(schema, key)
		}
	}
	if constant, exists := schema["const"]; exists {
		delete(schema, "const")
		schema["enum"] = []any{constant}
	}

	for _, key := range []string{"$defs", "definitions", "properties"} {
		if raw, exists := schema[key]; exists {
			entries, ok := raw.(map[string]any)
			if !ok {
				return false
			}
			for name, child := range entries {
				node, ok := child.(map[string]any)
				if !ok || !enforceOpenAIStrictSchema(node) {
					return false
				}
				entries[name] = node
			}
		}
	}
	for _, key := range []string{"items"} {
		if raw, exists := schema[key]; exists {
			node, ok := raw.(map[string]any)
			if !ok || !enforceOpenAIStrictSchema(node) {
				return false
			}
			schema[key] = node
		}
	}
	for _, key := range []string{"anyOf", "allOf", "oneOf", "prefixItems"} {
		if raw, exists := schema[key]; exists {
			entries, ok := raw.([]any)
			if !ok || len(entries) == 0 {
				return false
			}
			for i, child := range entries {
				node, ok := child.(map[string]any)
				if !ok || !enforceOpenAIStrictSchema(node) {
					return false
				}
				entries[i] = node
			}
		}
	}

	object := schema["type"] == "object"
	if _, hasProperties := schema["properties"]; hasProperties {
		object = true
	}
	if object {
		properties, ok := schema["properties"].(map[string]any)
		if !ok {
			return false
		}
		required, ok := stringSet(schema["required"])
		if len(properties) > 0 && !ok {
			return false
		}
		// Making an optional field required-and-nullable changes the arguments
		// seen by tools. Only schemas already requiring every property qualify.
		for name := range properties {
			if !required[name] {
				return false
			}
		}
		for name := range required {
			if _, exists := properties[name]; !exists {
				return false
			}
		}
		// Reaching here proves the canonical object was explicitly closed; keep
		// the marker rather than changing JSON Schema's open-by-default meaning.
		schema["additionalProperties"] = false
	}

	_, hasType := schema["type"]
	_, hasRef := schema["$ref"]
	_, hasAny := schema["anyOf"]
	_, hasAll := schema["allOf"]
	_, hasOne := schema["oneOf"]
	return hasType || hasRef || hasAny || hasAll || hasOne
}

func stringSet(value any) (map[string]bool, bool) {
	if value == nil {
		return map[string]bool{}, true
	}
	values, ok := value.([]any)
	if !ok {
		return nil, false
	}
	out := make(map[string]bool, len(values))
	for _, raw := range values {
		name, ok := raw.(string)
		if !ok {
			return nil, false
		}
		out[name] = true
	}
	return out, true
}

func emptyObjectToolSchema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}}
}

func cloneJSONMap(value map[string]any) (map[string]any, bool) {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}
	var cloned map[string]any
	if json.Unmarshal(raw, &cloned) != nil {
		return nil, false
	}
	return cloned, true
}

func ensureObjectProperties(schema map[string]any) map[string]any {
	if schema["type"] == "object" {
		if _, ok := schema["properties"].(map[string]any); !ok {
			schema["properties"] = map[string]any{}
		}
	}
	return schema
}

// normalizeResponsesSchemaNode applies the non-strict Responses/Codex subset:
// oneOf becomes anyOf and every object declares properties. Recursion is
// limited to schema-valued positions so enum/default/example payloads remain
// literal data.
func normalizeResponsesSchemaNode(value any) any {
	schema, ok := value.(map[string]any)
	if !ok {
		return value
	}
	// An empty JSON Schema means "any JSON value". Responses-compatible
	// grammar engines distinguish that from an object with no properties.
	if len(schema) == 0 {
		return true
	}
	if pattern, _ := schema["pattern"].(string); containsRegexLookaround(pattern) {
		delete(schema, "pattern")
	}
	for key, child := range schema {
		switch key {
		case "properties", "patternProperties", "$defs", "definitions":
			if entries, ok := child.(map[string]any); ok {
				for name, entry := range entries {
					entries[name] = normalizeResponsesSchemaNode(entry)
				}
			}
		case "items", "additionalProperties", "not", "if", "then", "else", "contains":
			schema[key] = normalizeResponsesSchemaNode(child)
		case "anyOf", "allOf":
			if entries, ok := child.([]any); ok {
				for i := range entries {
					entries[i] = normalizeResponsesSchemaNode(entries[i])
				}
			}
		}
	}
	if oneOf, ok := schema["oneOf"].([]any); ok {
		converted := make([]any, len(oneOf))
		for i := range oneOf {
			converted[i] = normalizeResponsesSchemaNode(oneOf[i])
		}
		if anyOf, ok := schema["anyOf"].([]any); ok {
			schema["anyOf"] = append(anyOf, converted...)
		} else {
			schema["anyOf"] = converted
		}
		delete(schema, "oneOf")
	}
	return ensureObjectProperties(schema)
}

func containsRegexLookaround(pattern string) bool {
	return strings.Contains(pattern, "(?=") || strings.Contains(pattern, "(?!") ||
		strings.Contains(pattern, "(?<=") || strings.Contains(pattern, "(?<!")
}

// normalizeCloudCodeSchemaNode projects JSON Schema onto Cloud Code Assist's
// protobuf-shaped subset. It returns false for a shape that cannot be widened
// predictably; the caller then advertises an open object while retaining the
// canonical schema for local validation.
func normalizeCloudCodeSchemaNode(value any) (any, bool) {
	if boolean, ok := value.(bool); ok {
		if boolean {
			return map[string]any{}, true
		}
		return nil, false
	}
	schema, ok := value.(map[string]any)
	if !ok {
		return value, true
	}
	// Google SDKs accept these canonical camelCase names. Normalize only schema
	// nodes, never the user-authored names inside a properties map.
	for snake, camel := range map[string]string{
		"additional_properties": "additionalProperties",
		"any_of":                "anyOf",
		"prefix_items":          "prefixItems",
		"property_ordering":     "propertyOrdering",
	} {
		if child, exists := schema[snake]; exists {
			delete(schema, snake)
			schema[camel] = child
		}
	}
	if _, hasRef := schema["$ref"]; hasRef {
		return nil, false
	}
	if constant, exists := schema["const"]; exists {
		delete(schema, "const")
		schema["enum"] = []any{constant}
	}

	if variants, key, ok := schemaVariants(schema); ok {
		if nullable, ok := nullableVariant(variants); ok {
			base, representable := normalizeCloudCodeSchemaNode(nullable)
			if !representable {
				return nil, false
			}
			out, ok := base.(map[string]any)
			if !ok {
				return nil, false
			}
			out["nullable"] = true
			for sibling, siblingValue := range schema {
				if sibling != key {
					out[sibling] = siblingValue
				}
			}
			schema = out
		} else if key == "allOf" && len(variants) == 1 {
			base, representable := normalizeCloudCodeSchemaNode(variants[0])
			if !representable {
				return nil, false
			}
			out, ok := base.(map[string]any)
			if !ok {
				return nil, false
			}
			for sibling, siblingValue := range schema {
				if sibling != key {
					out[sibling] = siblingValue
				}
			}
			schema = out
		} else {
			return nil, false
		}
	}

	if types, ok := schema["type"].([]any); ok {
		var concrete string
		nullable := false
		for _, raw := range types {
			typeName, ok := raw.(string)
			if !ok {
				return nil, false
			}
			if typeName == "null" {
				nullable = true
				continue
			}
			if concrete != "" && concrete != typeName {
				return nil, false
			}
			concrete = typeName
		}
		if concrete == "" {
			return nil, false
		}
		schema["type"] = concrete
		if nullable {
			schema["nullable"] = true
		}
	}
	if _, typed := schema["type"]; !typed {
		if inferred := inferEnumType(schema["enum"]); inferred != "" {
			schema["type"] = inferred
		}
	}

	var lifted []string
	for key, child := range schema {
		if _, unsupported := cloudCodeUnsupportedSchemaFields[key]; unsupported {
			if isUsefulSchemaAnnotation(key) {
				lifted = append(lifted, fmt.Sprintf("%s: %v", key, child))
			}
			delete(schema, key)
			continue
		}
		switch key {
		case "properties":
			entries, ok := child.(map[string]any)
			if !ok {
				return nil, false
			}
			optional := make(map[string]bool)
			for name, entry := range entries {
				normalized, representable := normalizeCloudCodeSchemaNode(entry)
				if !representable {
					return nil, false
				}
				if property, ok := normalized.(map[string]any); ok && property["nullable"] == true {
					delete(property, "nullable")
					optional[name] = true
				}
				entries[name] = normalized
			}
			if required, ok := schema["required"].([]any); ok && len(optional) > 0 {
				kept := required[:0]
				for _, raw := range required {
					name, _ := raw.(string)
					if !optional[name] {
						kept = append(kept, raw)
					}
				}
				schema["required"] = kept
			}
		case "items":
			normalized, representable := normalizeCloudCodeSchemaNode(child)
			if !representable {
				return nil, false
			}
			schema[key] = normalized
		case "anyOf", "oneOf", "allOf", "not":
			// A supported nullable/single-allOf wrapper was consumed above.
			return nil, false
		}
	}
	if len(lifted) > 0 {
		sort.Strings(lifted)
		note := strings.Join(lifted, "; ")
		if description, _ := schema["description"].(string); description != "" {
			schema["description"] = description + " (" + note + ")"
		} else {
			schema["description"] = note
		}
	}
	return ensureObjectProperties(schema), true
}

func inferEnumType(value any) string {
	values, ok := value.([]any)
	if !ok || len(values) == 0 {
		return ""
	}
	kind := ""
	for _, value := range values {
		current := ""
		switch value.(type) {
		case string:
			current = "string"
		case bool:
			current = "boolean"
		case float64:
			current = "number"
		case nil:
			current = "null"
		default:
			return ""
		}
		if kind != "" && kind != current {
			return ""
		}
		kind = current
	}
	return kind
}

func hasResidualCloudCodeSchema(value any) bool {
	schema, ok := value.(map[string]any)
	if !ok {
		return false
	}
	if _, exists := schema["nullable"]; exists {
		return true
	}
	if _, array := schema["type"].([]any); array {
		return true
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf", "not"} {
		if _, exists := schema[key]; exists {
			return true
		}
	}
	for _, key := range []string{"properties", "patternProperties", "$defs", "definitions"} {
		entries, _ := schema[key].(map[string]any)
		for _, entry := range entries {
			if hasResidualCloudCodeSchema(entry) {
				return true
			}
		}
	}
	for _, key := range []string{"items", "additionalProperties", "if", "then", "else", "contains"} {
		if hasResidualCloudCodeSchema(schema[key]) {
			return true
		}
	}
	return false
}

func schemaVariants(schema map[string]any) ([]any, string, bool) {
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if values, ok := schema[key].([]any); ok {
			return values, key, true
		}
	}
	return nil, "", false
}

func nullableVariant(variants []any) (any, bool) {
	if len(variants) != 2 {
		return nil, false
	}
	for i, variant := range variants {
		if schema, ok := variant.(map[string]any); ok && schema["type"] == "null" {
			return variants[1-i], true
		}
	}
	return nil, false
}

func isUsefulSchemaAnnotation(key string) bool {
	switch key {
	case "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
		"exclusiveMinimum", "exclusiveMaximum", "multipleOf", "pattern", "format", "examples":
		return true
	default:
		return false
	}
}
