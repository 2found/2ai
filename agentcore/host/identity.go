package host

import (
	"bytes"
	"encoding/json"
	"errors"
	"math/big"
	"strings"
)

// SameJSON compares opaque native JSON without modifying its replay bytes.
func SameJSON(a, b json.RawMessage) bool {
	left, leftErr := CanonicalJSON(a)
	right, rightErr := CanonicalJSON(b)
	return leftErr == nil && rightErr == nil && bytes.Equal(left, right)
}

// CanonicalJSON normalizes object ordering and decimal spelling without
// rounding opaque native numbers through float64. It is only for identity
// checks; the original messages remain the replay payload.
func CanonicalJSON(raw json.RawMessage) ([]byte, error) {
	if !json.Valid(raw) {
		return nil, errors.New("invalid native JSON")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	var normalize func(any) any
	normalize = func(value any) any {
		switch value := value.(type) {
		case json.Number:
			return canonicalNumber(value)
		case []any:
			for i := range value {
				value[i] = normalize(value[i])
			}
		case map[string]any:
			for key := range value {
				value[key] = normalize(value[key])
			}
		}
		return value
	}
	return json.Marshal(normalize(value))
}

func canonicalNumber(number json.Number) json.Number {
	text := string(number)
	sign := ""
	if strings.HasPrefix(text, "-") {
		sign, text = "-", text[1:]
	}
	var exponent big.Int
	if i := strings.IndexAny(text, "eE"); i >= 0 {
		exponent.SetString(text[i+1:], 10) // input already passed JSON validation
		text = text[:i]
	}
	fraction := 0
	if i := strings.IndexByte(text, '.'); i >= 0 {
		fraction = len(text) - i - 1
		text = text[:i] + text[i+1:]
	}
	text = strings.TrimLeft(text, "0")
	if text == "" {
		return "0"
	}
	digits := strings.TrimRight(text, "0")
	exponent.Add(&exponent, big.NewInt(int64(len(text)-len(digits)-fraction)))
	// Use encoding/json's ordinary/scientific thresholds so existing receipts
	// for native double values retain their digest. Only precision previously
	// lost to float64 changes identity. Never expand an unbounded exponent.
	point := new(big.Int).Add(&exponent, big.NewInt(int64(len(digits))))
	if point.IsInt64() && point.Int64() >= -5 && point.Int64() <= 21 {
		p := int(point.Int64())
		switch {
		case p <= 0:
			return json.Number(sign + "0." + strings.Repeat("0", -p) + digits)
		case p >= len(digits):
			return json.Number(sign + digits + strings.Repeat("0", p-len(digits)))
		default:
			return json.Number(sign + digits[:p] + "." + digits[p:])
		}
	}
	point.Sub(point, big.NewInt(1))
	suffix := point.String()
	if point.Sign() >= 0 {
		suffix = "+" + suffix
	}
	if len(digits) > 1 {
		digits = digits[:1] + "." + digits[1:]
	}
	return json.Number(sign + digits + "e" + suffix)
}
