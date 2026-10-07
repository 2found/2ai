package engine

import (
	"fmt"
	"math/big"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
)

func validationLines(err *jsonschema.ValidationError) []string {
	path := validationPath(err.InstanceLocation)
	message := ""
	switch k := err.ErrorKind.(type) {
	case *kind.Type:
		message = "must be " + strings.Join(k.Want, " or ")
		if len(k.Want) > 1 {
			message = "must be either " + strings.Join(k.Want, " or ")
		}
	case *kind.Required:
		if len(k.Missing) > 0 && k.Missing[0] != "" {
			if path != "" {
				path += "."
			}
			path += k.Missing[0]
		}
		message = "must have required properties " + strings.Join(k.Missing, ", ")
	case *kind.AdditionalProperties:
		message = "must not have additional properties"
	case *kind.Enum:
		message = "must be equal to one of the allowed values"
	case *kind.Const:
		message = "must be equal to constant"
	case *kind.MinLength:
		message = fmt.Sprintf("must not have fewer than %d characters", k.Want)
	case *kind.MaxLength:
		message = fmt.Sprintf("must not have more than %d characters", k.Want)
	case *kind.Pattern:
		message = "must match pattern \"" + k.Want + "\""
	case *kind.Format:
		message = "must match format \"" + k.Want + "\""
	case *kind.MinItems:
		message = fmt.Sprintf("must not have fewer than %d items", k.Want)
	case *kind.MaxItems:
		message = fmt.Sprintf("must not have more than %d items", k.Want)
	case *kind.MinProperties:
		message = fmt.Sprintf("must not have fewer than %d properties", k.Want)
	case *kind.MaxProperties:
		message = fmt.Sprintf("must not have more than %d properties", k.Want)
	case *kind.UniqueItems:
		message = "must not have duplicate items"
	case *kind.Contains, *kind.MinContains, *kind.MaxContains:
		message = "must contain at least 1 valid item"
	case *kind.Not:
		message = "must not be valid"
	case *kind.FalseSchema:
		message = "schema is false"
	case *kind.MultipleOf:
		message = "must be multiple of " + validationNumber(k.Want)
	case *kind.Minimum:
		message = "must be >= " + validationNumber(k.Want)
	case *kind.Maximum:
		message = "must be <= " + validationNumber(k.Want)
	case *kind.ExclusiveMinimum:
		message = "must be > " + validationNumber(k.Want)
	case *kind.ExclusiveMaximum:
		message = "must be < " + validationNumber(k.Want)
	case *kind.AnyOf:
		message = "must match a schema in anyOf"
	case *kind.OneOf:
		message = "must match exactly one schema in oneOf"
	}
	if message != "" {
		return []string{validationLine(path, message)}
	}
	lines := []string{}
	for _, cause := range err.Causes {
		lines = append(lines, validationLines(cause)...)
	}
	if len(lines) == 0 {
		lines = append(lines, "  - "+path+": "+err.Error())
	}
	return lines
}

func validationNumber(number *big.Rat) string {
	value, _ := number.Float64()
	raw, _ := marshalJSScalar(value)
	return string(raw)
}

func validationPath(location []string) string {
	// Pi replaces every slash in its instance path, including literal slashes
	// in property names. A required property's name is appended afterwards.
	return strings.ReplaceAll(strings.Join(location, "."), "/", ".")
}

func validationLine(path, message string) string {
	if path == "" {
		path = "root"
	}
	return "  - " + path + ": " + message
}
