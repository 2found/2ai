// Package jsonjs normalizes serialized JSON using JavaScript value semantics.
// It is shared by native AI, agent and telemetry code and uses only Go.
package jsonjs

import (
	"cmp"
	"encoding/json"
	"math"
	"slices"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"
)

// StringifyJSON normalizes serialized JSON using the value semantics of Pi's
// JSON.parse/JSON.stringify boundary. It preserves insertion order, applies JS
// index-key and duplicate-key rules, converts numbers to binary64, and retains
// lone UTF-16 surrogates. Invalid JSON returns the Go decoder's error.
func StringifyJSON(raw []byte) ([]byte, error) {
	if !json.Valid(raw) {
		var value json.RawMessage
		err := json.Unmarshal(raw, &value)
		return nil, err
	}
	parser := jsonValueParser{raw: raw}
	return parser.value(), nil
}

// ArrayIndex recognizes the own-property index keys sorted first by JS.
func ArrayIndex(key string) (uint64, bool) {
	n, err := strconv.ParseUint(key, 10, 32)
	return n, err == nil && n < 4294967295 && strconv.FormatUint(n, 10) == key
}

func stringifyJSONNumber(number float64) []byte {
	if math.IsNaN(number) || math.IsInf(number, 0) {
		return []byte("null")
	}
	if number == 0 {
		return []byte("0")
	}
	raw, _ := json.Marshal(number)
	return raw
}

type jsonValueParser struct {
	raw    []byte
	pos    int
	object func([]Property) any
	array  func([]any) any
}
type jsonObjectField struct {
	name, value []byte
	index       uint64
	isIndex     bool
}

// ObjectFields builds an object from canonical JSON keys, retaining UTF-16
// identity without converting lone surrogates into replacement characters.
type ObjectFields struct {
	fields    []jsonObjectField
	positions map[string]int
}

// Set replaces duplicate values without changing the first insertion position.
// The name must be a canonical JSON string produced by StringifyJSON.
func (o *ObjectFields) Set(name, value []byte) {
	key := string(name)
	if position, found := o.positions[key]; found {
		o.fields[position].value = value
		return
	}
	if o.positions == nil {
		o.positions = make(map[string]int)
	}
	o.positions[key] = len(o.fields)
	// Canonical index keys contain only unescaped ASCII digits.
	index, isIndex := ArrayIndex(key[1 : len(key)-1])
	o.fields = append(o.fields, jsonObjectField{name, value, index, isIndex})
}

// Marshal orders index keys numerically and retains other insertion positions.
func (o *ObjectFields) Marshal() []byte {
	slices.SortStableFunc(o.fields, func(a, b jsonObjectField) int {
		if a.isIndex && b.isIndex {
			return cmp.Compare(a.index, b.index)
		}
		if a.isIndex {
			return -1
		}
		if b.isIndex {
			return 1
		}
		return 0
	})
	out := []byte{'{'}
	for i, field := range o.fields {
		o.positions[string(field.name)] = i
		if i > 0 {
			out = append(out, ',')
		}
		out = append(out, field.name...)
		out = append(out, ':')
		out = append(out, field.value...)
	}
	return append(out, '}')
}

func (p *jsonValueParser) space() {
	for p.pos < len(p.raw) {
		switch p.raw[p.pos] {
		case ' ', '\n', '\r', '\t':
			p.pos++
		default:
			return
		}
	}
}

func (p *jsonValueParser) value() []byte {
	p.space()
	switch p.raw[p.pos] {
	case '"':
		return quoteJSONUTF16(p.stringUnits())
	case '{':
		p.pos++
		fields := ObjectFields{}
		p.space()
		for p.raw[p.pos] != '}' {
			name := quoteJSONUTF16(p.stringUnits())
			p.space()
			p.pos++ // colon
			fields.Set(name, p.value())
			p.space()
			if p.raw[p.pos] == '}' {
				break
			}
			p.pos++
			p.space() // comma
		}
		p.pos++
		return fields.Marshal()
	case '[':
		p.pos++
		p.space()
		out := []byte{'['}
		for p.raw[p.pos] != ']' {
			out = append(out, p.value()...)
			p.space()
			if p.raw[p.pos] == ']' {
				break
			}
			p.pos++
			out = append(out, ',')
			p.space()
		}
		p.pos++
		return append(out, ']')
	default:
		token := p.literal()
		if token[0] == 'n' || token[0] == 't' || token[0] == 'f' {
			return token
		}
		number, _ := strconv.ParseFloat(string(token), 64)
		return stringifyJSONNumber(number)
	}
}

func (p *jsonValueParser) literal() []byte {
	start := p.pos
	for p.pos < len(p.raw) {
		switch p.raw[p.pos] {
		case ',', '}', ']', ' ', '\r', '\n', '\t':
			return p.raw[start:p.pos]
		default:
			p.pos++
		}
	}
	return p.raw[start:p.pos]
}

func (p *jsonValueParser) stringUnits() []uint16 {
	p.pos++ // opening quote
	units := []uint16{}
	for p.raw[p.pos] != '"' {
		if p.raw[p.pos] == '\\' {
			p.pos++
			escape := p.raw[p.pos]
			p.pos++
			var unit uint16
			switch escape {
			case 'u':
				for range 4 {
					c := p.raw[p.pos]
					p.pos++
					var digit byte
					switch {
					case c <= '9':
						digit = c - '0'
					case c <= 'F':
						digit = c - 'A' + 10
					default:
						digit = c - 'a' + 10
					}
					unit = unit*16 + uint16(digit)
				}
			case 'b':
				unit = '\b'
			case 'f':
				unit = '\f'
			case 'n':
				unit = '\n'
			case 'r':
				unit = '\r'
			case 't':
				unit = '\t'
			default:
				unit = uint16(escape)
			}
			units = append(units, unit)
		} else {
			raw := p.raw[p.pos:]
			r, size := utf8.DecodeRune(raw)
			// In-memory JavaScript strings may contain unpaired UTF-16 units.
			// Preserve their WTF-8 representation, as StringCodePoints does.
			if len(raw) >= 3 && raw[0] == 0xed && raw[1] >= 0xa0 && raw[1] <= 0xbf && raw[2]&0xc0 == 0x80 {
				r, size = rune(raw[0]&0xf)<<12|rune(raw[1]&0x3f)<<6|rune(raw[2]&0x3f), 3
			}
			p.pos += size
			if r > 0xffff {
				high, low := utf16.EncodeRune(r)
				units = append(units, uint16(high), uint16(low))
			} else {
				units = append(units, uint16(r))
			}
		}
	}
	p.pos++ // closing quote
	return units
}

func quoteJSONUTF16(units []uint16) []byte {
	out := []byte{'"'}
	const hex = "0123456789abcdef"
	for i := 0; i < len(units); i++ {
		unit := units[i]
		switch unit {
		case '"', '\\':
			out = append(out, '\\', byte(unit))
		case '\b':
			out = append(out, '\\', 'b')
		case '\f':
			out = append(out, '\\', 'f')
		case '\n':
			out = append(out, '\\', 'n')
		case '\r':
			out = append(out, '\\', 'r')
		case '\t':
			out = append(out, '\\', 't')
		default:
			if unit >= 0xd800 && unit <= 0xdbff && i+1 < len(units) && units[i+1] >= 0xdc00 && units[i+1] <= 0xdfff {
				out = utf8.AppendRune(out, utf16.DecodeRune(rune(unit), rune(units[i+1])))
				i++
			} else if unit < 0x20 || (unit >= 0xd800 && unit <= 0xdfff) {
				out = append(out, '\\', 'u', hex[unit>>12], hex[unit>>8&15], hex[unit>>4&15], hex[unit&15])
			} else {
				out = utf8.AppendRune(out, rune(unit))
			}
		}
	}
	return append(out, '"')
}
