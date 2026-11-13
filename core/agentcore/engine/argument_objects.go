package engine

import (
	"encoding/json"
	"slices"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// Mark object keys evaluated only after their schema succeeds. The validator's
// built-in object checks mark every visited key, including failed checks.
type argumentObjectRules struct {
	properties    map[string]*jsonschema.Schema
	patterns      map[jsonschema.Regexp]*jsonschema.Schema
	additional    *jsonschema.Schema
	propertyOrder []string
	patternOrder  []jsonschema.Regexp
	validator     *argumentValidator
}

func (rule *argumentObjectRules) Validate(ctx *jsonschema.ValidatorContext, value any) {
	object, ok := value.(map[string]any)
	if !ok {
		return
	}
	evaluated := map[string]bool{}
	check := func(schema *jsonschema.Schema, key string, value any) {
		if err := ctx.Validate(schema, value, []string{key}); err != nil {
			ctx.AddErr(err)
			// Pi leaves the failed child's evaluation frame on its stack.
			// Later checks therefore lose marks accumulated in the parent frame.
			clear(evaluated)
		} else {
			evaluated[key] = true
		}
	}
	keys, _ := validationObject(validationSchemaAt(rule.validator.arguments, ctx.ValueLocation()))
	if len(keys) != len(object) {
		keys = keys[:0]
		for key := range object {
			keys = append(keys, key)
		}
		slices.Sort(keys)
	}
	for _, key := range keys {
		_, declared := rule.properties[key]
		for _, pattern := range rule.patternOrder {
			if pattern.MatchString(key) {
				declared = true
			}
		}
		if !declared && rule.additional != nil {
			check(rule.additional, key, object[key])
		}
	}
	for _, pattern := range rule.patternOrder {
		for _, key := range keys {
			if pattern.MatchString(key) {
				check(rule.patterns[pattern], key, object[key])
			}
		}
	}
	for _, key := range rule.propertyOrder {
		if value, exists := object[key]; exists {
			check(rule.properties[key], key, value)
		}
	}
	for key := range evaluated {
		ctx.EvaluatedProp(key)
	}
}

func registerArgumentObjects(compiler *jsonschema.Compiler, source json.RawMessage, validator *argumentValidator) {
	compiler.AssertVocabs()
	compiler.RegisterVocabulary(&jsonschema.Vocabulary{
		URL: "https://agentcore.local/pi-object-semantics",
		Compile: func(ctx *jsonschema.CompilerContext, obj map[string]any) (jsonschema.SchemaExt, error) {
			current := ctx.Enqueue(nil)
			if len(current.Properties) == 0 && len(current.PatternProperties) == 0 && current.AdditionalProperties == nil {
				return nil, nil
			}
			rule := &argumentObjectRules{properties: current.Properties, patterns: current.PatternProperties, validator: validator}
			_, fields := validationObject(validationSchemaAt(source, validationPointer(current.Location)))
			rule.propertyOrder, _ = validationObject(fields["properties"])
			patterns, _ := validationObject(fields["patternProperties"])
			for _, name := range patterns {
				for pattern := range rule.patterns {
					if pattern.String() == name {
						rule.patternOrder = append(rule.patternOrder, pattern)
					}
				}
			}
			if _, exists := obj["additionalProperties"]; exists {
				rule.additional = ctx.Enqueue([]string{"additionalProperties"})
			}
			current.Properties = nil
			current.PatternProperties = nil
			current.AdditionalProperties = nil
			return rule, nil
		},
	})
}
