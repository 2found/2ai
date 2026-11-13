package engine

import (
	"strconv"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// Pi accepts prefixItems alongside both legacy tuple items and scalar items.
// Scalar items validates only the tail; tuple items independently checks index 0.
type argumentArrayRules struct {
	prefix           []*jsonschema.Schema
	tuple            []*jsonschema.Schema
	tail             *jsonschema.Schema
	additional       *jsonschema.Schema
	additionalOffset int
	contains         *jsonschema.Schema
	minContains      *int
}

func (rule *argumentArrayRules) Validate(ctx *jsonschema.ValidatorContext, value any) {
	array, ok := value.([]any)
	if !ok {
		return
	}
	evaluated := map[int]bool{}
	check := func(schema *jsonschema.Schema, index int) bool {
		if err := ctx.Validate(schema, array[index], []string{strconv.Itoa(index)}); err != nil {
			ctx.AddErr(err)
			clear(evaluated) // A failed Pi child retains its own evaluation frame.
			return false
		}
		evaluated[index] = true
		return true
	}
	if rule.additional != nil {
		for i := rule.additionalOffset; i < len(array); i++ {
			if !check(rule.additional, i) {
				break
			} // First invalid legacy tuple-tail item.
		}
	}
	// Contains checks use an isolated item value without reporting its errors.
	// Pi marks matching indices once for contains and again for minContains.
	matched := []int{}
	if rule.contains != nil {
		for i := range array {
			if ctx.Validate(rule.contains, array[i], []string{strconv.Itoa(i)}) == nil {
				matched = append(matched, i)
			}
		}
		if rule.minContains == nil || *rule.minContains != 0 {
			for _, i := range matched {
				evaluated[i] = true
			}
		}
	}
	if rule.tail != nil {
		for i := len(rule.prefix); i < len(array); i++ {
			check(rule.tail, i)
		}
	}
	for i, schema := range rule.tuple {
		if i >= len(array) {
			break
		}
		check(schema, i)
	}
	if rule.minContains != nil {
		for _, i := range matched {
			evaluated[i] = true
		}
	}
	for i, schema := range rule.prefix {
		if i >= len(array) {
			break
		}
		check(schema, i)
	}
	for index := range evaluated {
		ctx.EvaluatedItem(index)
	}
}

func registerArgumentArrays(compiler *jsonschema.Compiler) {
	compiler.AssertVocabs()
	compiler.RegisterVocabulary(&jsonschema.Vocabulary{
		URL:        "https://agentcore.local/pi-array-semantics",
		Subschemas: []jsonschema.SchemaPath{{jsonschema.Prop("prefixItems"), jsonschema.AllItem{}}},
		Compile: func(ctx *jsonschema.CompilerContext, obj map[string]any) (jsonschema.SchemaExt, error) {
			prefix, hasPrefix := obj["prefixItems"].([]any)
			tuple, hasTuple := obj["items"].([]any)
			_, hasAdditional := obj["additionalItems"]
			_, hasItems := obj["items"]
			_, hasContains := obj["contains"]
			if !hasPrefix && !hasItems && !hasContains {
				return nil, nil
			}
			rule := &argumentArrayRules{prefix: make([]*jsonschema.Schema, len(prefix))}
			for i := range prefix {
				rule.prefix[i] = ctx.Enqueue([]string{"prefixItems", strconv.Itoa(i)})
			}
			current := ctx.Enqueue(nil)
			rule.contains, rule.minContains = current.Contains, current.MinContains
			if hasTuple && hasAdditional {
				rule.additional = ctx.Enqueue([]string{"additionalItems"})
				rule.additionalOffset = len(tuple)
				current.AdditionalItems = nil
			}
			switch items := current.Items.(type) {
			case *jsonschema.Schema:
				rule.tail = items
			case []*jsonschema.Schema:
				rule.tuple = items
			}
			current.Items = nil
			return rule, nil
		},
	})
}
