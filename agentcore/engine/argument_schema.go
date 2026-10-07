package engine

import (
	"encoding/json"
	"math"
	"net/url"
	"strconv"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

type argumentValidator struct {
	schema            *jsonschema.Schema
	locations         map[string]string
	source, arguments json.RawMessage
}

func (v *argumentValidator) Validate(value any) error {
	v.arguments, _ = marshalArguments(value, v.source)
	err := v.schema.Validate(argumentValidationValue(value))
	if failure, ok := err.(*jsonschema.ValidationError); ok {
		v.restoreLocations(failure)
	}
	return err
}

func (v *argumentValidator) restoreLocations(err *jsonschema.ValidationError) {
	if location, parseErr := url.Parse(err.SchemaURL); parseErr == nil {
		if original, ok := v.locations[location.Fragment]; ok {
			location.Fragment = original
			err.SchemaURL = location.String()
		}
	}
	for _, child := range err.Causes {
		v.restoreLocations(child)
	}
}

// compilerSchema projects Pi's schema metadata onto the Go compiler. TypeBox
// recognizes its keyword set independently of $schema; letting that annotation
// select a Go dialect would silently disable bounds or reject legacy tuples.
// Only schema-bearing keywords are traversed: const, enum and defaults are data.
func compilerSchema(value any, path string, locations map[string]string) any {
	switch schema := value.(type) {
	case []any:
		result := make([]any, len(schema))
		for i, child := range schema {
			result[i] = compilerSchema(child, path+"/"+strconv.Itoa(i), locations)
		}
		return result
	case map[string]any:
		result := make(map[string]any, len(schema))
		for key, child := range schema {
			switch key {
			case "$schema":
				continue
			case "properties", "patternProperties", "$defs", "definitions", "dependencies", "dependentSchemas":
				if entries, ok := child.(map[string]any); ok {
					children := make(map[string]any, len(entries))
					for name, entry := range entries {
						children[name] = compilerSchema(entry, path+"/"+key+"/"+strings.ReplaceAll(strings.ReplaceAll(name, "~", "~0"), "/", "~1"), locations)
					}
					result[key] = children
				} else {
					result[key] = child
				}
			case "items", "prefixItems", "additionalItems", "contains", "additionalProperties", "propertyNames", "unevaluatedItems", "unevaluatedProperties", "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "contentSchema":
				result[key] = compilerSchema(child, path+"/"+key, locations)
			default:
				result[key] = child
			}
		}
		// Go stops at type/const failures. Isolate these checks in allOf so
		// sibling checks still run, then restore their source schema locations.
		branches, valid := result["allOf"].([]any)
		if _, exists := result["allOf"]; !exists || valid {
			for _, keyword := range []string{"type", "const", "enum"} {
				if constraint, exists := result[keyword]; exists {
					locations[path+"/allOf/"+strconv.Itoa(len(branches))] = path
					branches = append(branches, map[string]any{keyword: constraint})
					delete(result, keyword)
				}
			}
			if len(branches) > 0 {
				result["allOf"] = branches
			}
		}
		return result
	default:
		return value
	}
}

// The Go validator rejects nonfinite float64 before visiting any schema.
// Keep them distinct from finite numbers in its private validation view.
// Concrete execution values are never replaced by this projection.
func argumentValidationValue(value any) any {
	switch value := value.(type) {
	case float64:
		if math.IsInf(value, 1) {
			return json.Number("1e400")
		}
		if math.IsInf(value, -1) {
			return json.Number("-1e400")
		}
	case map[string]any:
		result := make(map[string]any, len(value))
		for key, child := range value {
			result[key] = argumentValidationValue(child)
		}
		return result
	case []any:
		result := make([]any, len(value))
		for i, child := range value {
			result[i] = argumentValidationValue(child)
		}
		return result
	}
	return value
}
