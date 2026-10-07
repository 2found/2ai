package engine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"

	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
	"github.com/santhosh-tekuri/jsonschema/v6"
)

// validateArguments follows Pi's serialized JSON-schema path. It clones the
// arguments before optional-null normalization and primitive coercion.
func validateArguments(tool ai.Tool, raw json.RawMessage) (any, error) {
	schema, err := jsonjs.DecodeJSON(tool.Parameters)
	if err != nil {
		return nil, err
	}
	args, err := jsonjs.DecodeJSON(raw)
	if err != nil {
		return nil, err
	}
	template, err := jsonjs.DecodeValue(raw)
	if err != nil {
		return nil, err
	}
	normalizeNulls(args, schema)
	compiled, err := compileSchema(schema, tool.Parameters, raw)
	if err != nil {
		return nil, err
	}
	coerced := coerce(args, schema)
	if !sameValue(coerced, args) {
		// Pi's primitive root replacement returns the original argument when
		// the converted primitive fails validation, without throwing.
		_, a := args.(map[string]any)
		_, b := coerced.(map[string]any)
		if !a || !b {
			if compiled.Validate(coerced) == nil {
				return orderedArgumentValue(coerced, template), nil
			}
			return orderedArgumentValue(args, template), nil
		}
		args = coerced
	}
	if err := compiled.Validate(args); err != nil {
		original, decodeErr := jsonjs.DecodeJSON(raw)
		if decodeErr != nil {
			return nil, decodeErr
		}
		canonical, marshalErr := marshalArguments(original, raw)
		if marshalErr != nil {
			return nil, marshalErr
		}
		var pretty bytes.Buffer
		if indentErr := json.Indent(&pretty, canonical, "", "  "); indentErr != nil {
			return nil, indentErr
		}
		orderedSchema, marshalErr := marshalArguments(schema, tool.Parameters)
		if marshalErr != nil {
			return nil, marshalErr
		}
		orderedArgs, marshalErr := marshalArguments(args, raw)
		if marshalErr != nil {
			return nil, marshalErr
		}
		lines := orderedValidationLines(err.(*jsonschema.ValidationError), orderedSchema, orderedArgs)
		return nil, fmt.Errorf("Validation failed for tool \"%s\":\n%s\n\nReceived arguments:\n%s", tool.Name, strings.Join(lines, "\n"), pretty.String())
	}
	return orderedArgumentValue(args, template), nil
}

func compileSchema(schema any, source ...json.RawMessage) (*argumentValidator, error) {
	compiler := jsonschema.NewCompiler()
	// Pi recognizes contains bounds and newer object keywords alongside
	// legacy tuple items. Draft 2019 preserves that tuple representation.
	compiler.DefaultDraft(jsonschema.Draft2019)
	compiler.UseLoader(nil)
	compiler.UseRegexpEngine(compileArgumentRegexp)
	registerArgumentNumbers(compiler)
	registerArgumentScalars(compiler)
	registerArgumentFormats(compiler)
	registerArgumentArrays(compiler)
	var schemaSource, argumentSource json.RawMessage
	if len(source) > 0 {
		schemaSource = source[0]
	}
	if len(source) > 1 {
		argumentSource = source[1]
	}
	orderedSchema, err := marshalArguments(schema, schemaSource)
	if err != nil {
		return nil, err
	}
	validator := &argumentValidator{source: argumentSource}
	registerArgumentObjects(compiler, orderedSchema, validator)
	const resource = "https://agentcore.local/pi-tool.json"
	locations := map[string]string{}
	if err := registerArgumentComposition(compiler, locations); err != nil {
		return nil, err
	}
	if err := compiler.AddResource(resource, compilerSchema(schema, "", locations)); err != nil {
		return nil, err
	}
	compiled, err := compiler.Compile(resource)
	if err != nil {
		return nil, err
	}
	validator.schema, validator.locations = compiled, locations
	return validator, nil
}

func matches(value, schema any) (bool, bool) {
	compiled, err := compileSchema(schema)
	if err != nil {
		return false, false
	}
	return compiled.Validate(value) == nil, true
}

func normalizeNulls(value, rawSchema any) {
	schema, ok := rawSchema.(map[string]any)
	if !ok {
		return
	}
	if array, ok := value.([]any); ok {
		for i, child := range array {
			if items, ok := schema["items"].([]any); ok {
				if i < len(items) {
					normalizeNulls(child, items[i])
				}
			} else if items, ok := schema["items"].(map[string]any); ok {
				normalizeNulls(child, items)
			}
		}
		return
	}
	object, ok := value.(map[string]any)
	if !ok {
		return
	}
	properties, _ := schema["properties"].(map[string]any)
	required := map[string]bool{}
	if names, ok := schema["required"].([]any); ok {
		for _, name := range names {
			if key, ok := name.(string); ok {
				required[key] = true
			}
		}
	}
	for key, childSchema := range properties {
		child, exists := object[key]
		if !exists {
			continue
		}
		definition, _ := childSchema.(map[string]any)
		_, isRef := definition["$ref"].(string)
		if child == nil && !required[key] && !isRef {
			if accepts, valid := matches(nil, childSchema); valid && !accepts {
				delete(object, key)
				continue
			}
		}
		normalizeNulls(child, childSchema)
	}
}

func coerce(value, rawSchema any) any {
	schema, ok := rawSchema.(map[string]any)
	if !ok {
		return value
	}
	if all, ok := schema["allOf"].([]any); ok {
		for _, child := range all {
			value = coerce(value, child)
		}
	}
	for _, name := range []string{"anyOf", "oneOf"} {
		if union, ok := schema[name].([]any); ok {
			value = coerceUnion(value, union)
		}
	}
	types := []string{}
	switch declared := schema["type"].(type) {
	case string:
		types = append(types, declared)
	case []any:
		for _, member := range declared {
			if name, ok := member.(string); ok {
				types = append(types, name)
			}
		}
	}
	unionMatch := false
	if len(types) > 1 {
		for _, name := range types {
			if typeMatches(value, name) {
				unionMatch = true
				break
			}
		}
	}
	if !unionMatch {
		for _, name := range types {
			candidate := coercePrimitive(value, name)
			if !sameValue(candidate, value) {
				value = candidate
				break
			}
		}
	}
	for _, name := range types {
		if name == "object" {
			if object, ok := value.(map[string]any); ok {
				properties, _ := schema["properties"].(map[string]any)
				for key, sub := range properties {
					if child, exists := object[key]; exists {
						object[key] = coerce(child, sub)
					}
				}
				if additional, ok := schema["additionalProperties"].(map[string]any); ok {
					for key, child := range object {
						if _, defined := properties[key]; !defined {
							object[key] = coerce(child, additional)
						}
					}
				}
			}
		}
		if name == "array" {
			if array, ok := value.([]any); ok {
				for i, child := range array {
					if items, ok := schema["items"].([]any); ok {
						if i < len(items) {
							array[i] = coerce(child, items[i])
						}
					} else if items, ok := schema["items"].(map[string]any); ok {
						array[i] = coerce(child, items)
					}
				}
			}
		}
	}
	return value
}

func coerceUnion(value any, schemas []any) any {
	for _, schema := range schemas {
		if accepts, valid := matches(value, schema); valid && accepts {
			return value
		}
	}
	for _, schema := range schemas {
		cloned := cloneArgument(value)
		candidate := coerce(cloned, schema)
		if accepts, valid := matches(candidate, schema); valid && accepts {
			return candidate
		}
	}
	return value
}

func typeMatches(value any, name string) bool {
	switch name {
	case "null":
		return value == nil
	case "string":
		_, ok := value.(string)
		return ok
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "number":
		_, ok := value.(float64)
		return ok
	case "integer":
		number, ok := value.(float64)
		return ok && !math.IsInf(number, 0) && math.Trunc(number) == number
	case "object":
		_, ok := value.(map[string]any)
		return ok
	case "array":
		_, ok := value.([]any)
		return ok
	}
	return false
}

func sameValue(a, b any) bool { return reflect.DeepEqual(a, b) }

func coercePrimitive(value any, name string) any {
	switch name {
	case "number", "integer":
		if value == nil {
			return float64(0)
		}
		if boolean, ok := value.(bool); ok {
			if boolean {
				return float64(1)
			}
			return float64(0)
		}
		if text, ok := value.(string); ok {
			text = strings.Trim(text, "\u0009\u000a\u000b\u000c\u000d \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff")
			if text == "" {
				return value
			}
			number := ai.ParseJSNumber(text)
			if !math.IsInf(number, 0) && !math.IsNaN(number) && (name != "integer" || math.Trunc(number) == number) {
				return number
			}
		}
	case "boolean":
		if value == nil {
			return false
		}
		if text, ok := value.(string); ok {
			if text == "true" {
				return true
			}
			if text == "false" {
				return false
			}
		}
		if number, ok := value.(float64); ok {
			if number == 1 {
				return true
			}
			if number == 0 {
				return false
			}
		}
	case "string":
		if value == nil {
			return ""
		}
		if boolean, ok := value.(bool); ok {
			return strconv.FormatBool(boolean)
		}
		if number, ok := value.(float64); ok {
			return jsNumberString(number)
		}
	case "null":
		if text, ok := value.(string); ok && text == "" {
			return nil
		}
		if number, ok := value.(float64); ok && number == 0 {
			return nil
		}
		if boolean, ok := value.(bool); ok && !boolean {
			return nil
		}
	}
	return value
}
