package engine

import (
	"github.com/2found/2ai/jsonjs"
	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
)

// Pi's numeric type checks require finite numbers; strings count Unicode code
// points, including lone surrogates, instead of Go's invalid-UTF-8 rune count.
type argumentScalarRules struct {
	types                []string
	minLength, maxLength *int
}

func (rule *argumentScalarRules) Validate(ctx *jsonschema.ValidatorContext, value any) {
	if len(rule.types) > 0 {
		valid := false
		for _, name := range rule.types {
			valid = valid || typeMatches(value, name)
		}
		if !valid {
			ctx.AddError(&kind.Type{Want: rule.types})
		}
	}
	if text, ok := value.(string); ok {
		length := len(jsonjs.StringCodePoints(text))
		if rule.minLength != nil && length < *rule.minLength {
			ctx.AddError(&kind.MinLength{Got: length, Want: *rule.minLength})
		}
		if rule.maxLength != nil && length > *rule.maxLength {
			ctx.AddError(&kind.MaxLength{Got: length, Want: *rule.maxLength})
		}
	}
}

func registerArgumentScalars(compiler *jsonschema.Compiler) {
	compiler.RegisterVocabulary(&jsonschema.Vocabulary{
		URL: "https://agentcore.local/pi-scalar-semantics",
		Compile: func(ctx *jsonschema.CompilerContext, _ map[string]any) (jsonschema.SchemaExt, error) {
			current := ctx.Enqueue(nil)
			rule := &argumentScalarRules{minLength: current.MinLength, maxLength: current.MaxLength}
			if current.Types != nil {
				rule.types = current.Types.ToStrings()
			}
			if len(rule.types) == 0 && rule.minLength == nil && rule.maxLength == nil {
				return nil, nil
			}
			current.Types, current.MinLength, current.MaxLength = nil, nil, nil
			return rule, nil
		},
	})
}
