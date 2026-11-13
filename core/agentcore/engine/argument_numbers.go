package engine

import (
	"math"
	"math/big"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
)

// Pi applies numeric constraints only to finite binary64 values. Divisibility
// uses floating-point remainders with an absolute tolerance, not exact rationals.
type argumentNumberRules struct {
	divisor                                              float64
	want                                                 *big.Rat
	minimum, maximum, exclusiveMinimum, exclusiveMaximum *big.Rat
}

func (rule *argumentNumberRules) Validate(ctx *jsonschema.ValidatorContext, value any) {
	number, ok := value.(float64)
	if !ok || math.IsInf(number, 0) || math.IsNaN(number) {
		return
	}
	got := new(big.Rat).SetFloat64(number)
	if rule.minimum != nil {
		bound, _ := rule.minimum.Float64()
		if number < bound {
			ctx.AddError(&kind.Minimum{Got: got, Want: rule.minimum})
		}
	}
	if rule.maximum != nil {
		bound, _ := rule.maximum.Float64()
		if number > bound {
			ctx.AddError(&kind.Maximum{Got: got, Want: rule.maximum})
		}
	}
	if rule.exclusiveMinimum != nil {
		bound, _ := rule.exclusiveMinimum.Float64()
		if number <= bound {
			ctx.AddError(&kind.ExclusiveMinimum{Got: got, Want: rule.exclusiveMinimum})
		}
	}
	if rule.exclusiveMaximum != nil {
		bound, _ := rule.exclusiveMaximum.Float64()
		if number >= bound {
			ctx.AddError(&kind.ExclusiveMaximum{Got: got, Want: rule.exclusiveMaximum})
		}
	}
	if rule.want == nil {
		return
	}

	if math.Trunc(number) == number && math.Mod(1/rule.divisor, 1) == 0 {
		return
	}
	remainder := math.Mod(number, rule.divisor)
	distance := math.Min(math.Abs(remainder), math.Min(math.Abs(remainder-rule.divisor), math.Abs(remainder+rule.divisor)))
	if distance < 1e-10 {
		return
	}
	ctx.AddError(&kind.MultipleOf{Got: new(big.Rat).SetFloat64(number), Want: rule.want})
}

func registerArgumentNumbers(compiler *jsonschema.Compiler) {
	compiler.AssertVocabs()
	compiler.RegisterVocabulary(&jsonschema.Vocabulary{
		URL: "https://agentcore.local/pi-numeric-semantics",
		Compile: func(ctx *jsonschema.CompilerContext, obj map[string]any) (jsonschema.SchemaExt, error) {
			compiled := ctx.Enqueue(nil)
			divisor, _ := obj["multipleOf"].(float64)
			rule := &argumentNumberRules{divisor: divisor, want: compiled.MultipleOf, minimum: compiled.Minimum, maximum: compiled.Maximum, exclusiveMinimum: compiled.ExclusiveMinimum, exclusiveMaximum: compiled.ExclusiveMaximum}
			if rule.want == nil && rule.minimum == nil && rule.maximum == nil && rule.exclusiveMinimum == nil && rule.exclusiveMaximum == nil {
				return nil, nil
			}
			compiled.MultipleOf, compiled.Minimum, compiled.Maximum, compiled.ExclusiveMinimum, compiled.ExclusiveMaximum = nil, nil, nil, nil, nil
			return rule, nil
		},
	})
}
