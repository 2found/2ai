package jscompat

import (
	"math"
	"math/big"
	"regexp"
	"strconv"
	"strings"
)

var jsDecimalNumber = regexp.MustCompile(`^[+-]?(?:Infinity|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)$`)

// ParseJSNumber implements Number(string) for Go ports of Pi's coercion paths.
// Empty/whitespace input becomes zero; invalid syntax becomes NaN. It accepts
// unsigned hex, binary and octal integers of arbitrary width and rounds them to
// binary64. Overflow becomes infinity. Callers apply their own finite/integer
// and empty-input policies, as the upstream call sites do.
func ParseJSNumber(value string) float64 {
	value = strings.TrimFunc(value, IsWhitespace)
	if value == "" {
		return 0
	}
	if len(value) > 2 && value[0] == '0' {
		base := 0
		switch value[1] {
		case 'x', 'X':
			base = 16
		case 'o', 'O':
			base = 8
		case 'b', 'B':
			base = 2
		}
		if base != 0 {
			n, ok := new(big.Int).SetString(value[2:], base)
			if !ok || strings.ContainsAny(value[2:], "_+-") {
				return math.NaN()
			}
			f, _ := n.Float64()
			return f
		}
	}
	// strconv accepts Go-only hexadecimal floats and numeric separators.
	// Validate the complete JavaScript decimal grammar before parsing.
	if !jsDecimalNumber.MatchString(value) {
		return math.NaN()
	}
	n, err := strconv.ParseFloat(value, 64)
	if err != nil && !math.IsInf(n, 0) {
		return math.NaN()
	}
	return n
}

// ECMAScript trim includes BOM, excludes U+0085, and is not Go's TrimSpace.
func IsWhitespace(r rune) bool {
	return r == '\t' || r == '\n' || r == '\v' || r == '\f' || r == '\r' || r == ' ' ||
		r == '\u00a0' || r == '\u1680' || (r >= '\u2000' && r <= '\u200a') ||
		r == '\u2028' || r == '\u2029' || r == '\u202f' || r == '\u205f' || r == '\u3000' || r == '\ufeff'
}

var completionsFloatPrefix = regexp.MustCompile(`^[+-]?(?:Infinity|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)`)

func ParseJSFloat(value string) float64 {
	prefix := completionsFloatPrefix.FindString(strings.TrimLeftFunc(value, IsWhitespace))
	if prefix == "" {
		return math.NaN()
	}
	number, _ := strconv.ParseFloat(prefix, 64)
	return number
}
