package jsonjs

import (
	"encoding/json"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"
)

// DecodeJSON reads JSON into nil, bool, float64, string, []any and map[string]any.
// Numbers use JavaScript's binary64 domain, including overflow. Strings retain
// lone UTF-16 surrogates as their three-byte WTF-8 encoding; valid pairs use
// ordinary UTF-8. QuoteString must be used to export those strings losslessly.
// Maps retain key identity and last duplicate values, but not insertion order.
func DecodeJSON(raw []byte) (any, error) {
	return DecodeJSONContainers(raw, nil, nil)
}

// Property is one JSON object entry. DecodeJSONContainers supplies entries in
// source order, including duplicates; the builder owns duplicate-key handling.
type Property struct {
	Name  string
	Value any
}

// DecodeJSONContainers supplies explicit object/array constructors. Nil
// constructors retain the ordinary Go map/slice representation. Children are
// constructed before their parent, after the entire input has been validated.
func DecodeJSONContainers(raw []byte, object func([]Property) any, array func([]any) any) (any, error) {
	if !json.Valid(raw) {
		var rejected json.RawMessage
		return nil, json.Unmarshal(raw, &rejected)
	}
	parser := jsonValueParser{raw: raw, object: object, array: array}
	return parser.decodedValue(), nil
}

func (p *jsonValueParser) decodedValue() any {
	p.space()
	switch p.raw[p.pos] {
	case '"':
		return p.decodedString()
	case '{':
		p.pos++
		object := map[string]any{}
		var properties []Property
		p.space()
		for p.raw[p.pos] != '}' {
			name := p.decodedString()
			p.space()
			p.pos++ // colon
			value := p.decodedValue()
			if p.object == nil {
				object[name] = value
			} else {
				properties = append(properties, Property{name, value})
			}
			p.space()
			if p.raw[p.pos] == '}' {
				break
			}
			p.pos++ // comma
			p.space()
		}
		p.pos++
		if p.object != nil {
			return p.object(properties)
		}
		return object
	case '[':
		p.pos++
		array := []any{}
		p.space()
		for p.raw[p.pos] != ']' {
			array = append(array, p.decodedValue())
			p.space()
			if p.raw[p.pos] == ']' {
				break
			}
			p.pos++ // comma
		}
		p.pos++
		if p.array != nil {
			return p.array(array)
		}
		return array
	default:
		token := p.literal()
		switch token[0] {
		case 'n':
			return nil
		case 't':
			return true
		case 'f':
			return false
		}
		// Syntax is already validated. The range-error result (infinity) is
		// intentional, and underflow preserves the sign of zero.
		number, _ := strconv.ParseFloat(string(token), 64)
		return number
	}
}

func stringFromUTF16(units []uint16) string {
	out := make([]byte, 0, len(units))
	for i := 0; i < len(units); i++ {
		u := units[i]
		if u >= 0xd800 && u <= 0xdbff && i+1 < len(units) && units[i+1] >= 0xdc00 && units[i+1] <= 0xdfff {
			out = utf8.AppendRune(out, utf16.DecodeRune(rune(u), rune(units[i+1])))
			i++
		} else if u >= 0xd800 && u <= 0xdfff {
			out = append(out, byte(0xe0|u>>12), byte(0x80|u>>6&0x3f), byte(0x80|u&0x3f))
		} else {
			out = utf8.AppendRune(out, rune(u))
		}
	}
	return string(out)
}

// QuoteString exports UTF-8 and decoded WTF-8 strings using JSON.stringify's
// escaping. Surrogate code units remain escaped; other invalid UTF-8 uses the
// replacement character, matching the shared serialized-JSON parser.
func QuoteString(value string) []byte {
	if utf8.ValidString(value) {
		out := make([]byte, 1, len(value)+2)
		out[0] = '"'
		const hex = "0123456789abcdef"
		for i := 0; i < len(value); i++ {
			c := value[i]
			switch c {
			case '"', '\\':
				out = append(out, '\\', c)
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
				if c < 0x20 {
					out = append(out, '\\', 'u', '0', '0', hex[c>>4], hex[c&15])
				} else {
					out = append(out, c)
				}
			}
		}
		return append(out, '"')
	}
	units := make([]uint16, 0, len(value))
	for _, r := range StringCodePoints(value) {
		if r > 0xffff {
			high, low := utf16.EncodeRune(r)
			units = append(units, uint16(high), uint16(low))
		} else {
			units = append(units, uint16(r))
		}
	}
	return quoteJSONUTF16(units)
}

// StringCodePoints follows JavaScript string iteration. A lone surrogate is
// one code point, represented by WTF-8 in Go; valid pairs are ordinary UTF-8.
func StringCodePoints(value string) []rune {
	points := make([]rune, 0, len(value))
	for i := 0; i < len(value); {
		if i+2 < len(value) && value[i] == 0xed && value[i+1] >= 0xa0 && value[i+1] <= 0xbf && value[i+2]&0xc0 == 0x80 {
			points = append(points, rune(value[i]&0xf)<<12|rune(value[i+1]&0x3f)<<6|rune(value[i+2]&0x3f))
			i += 3
		} else {
			r, size := utf8.DecodeRuneInString(value[i:])
			points = append(points, r)
			i += size
		}
	}
	return points
}

// RawProperty retains a decoded UTF-16 key and its original serialized value.
type RawProperty struct {
	Name  string
	Value json.RawMessage
}

// DecodeObjectProperties reads own properties in source order, including
// duplicates. Values borrow raw; callers retaining them must not mutate raw.
// Non-object JSON returns no properties. Invalid JSON returns a decoder error.
func DecodeObjectProperties(raw []byte) ([]RawProperty, error) {
	if !json.Valid(raw) {
		var rejected json.RawMessage
		return nil, json.Unmarshal(raw, &rejected)
	}
	p := jsonValueParser{raw: raw}
	p.space()
	if p.raw[p.pos] != '{' {
		return nil, nil
	}
	p.pos++
	p.space()
	properties := []RawProperty{}
	for p.raw[p.pos] != '}' {
		name := p.decodedString()
		p.space()
		p.pos++
		p.space()
		start := p.pos
		p.skipValue()
		properties = append(properties, RawProperty{Name: name, Value: raw[start:p.pos]})
		p.space()
		if p.raw[p.pos] == '}' {
			break
		}
		p.pos++
		p.space()
	}
	return properties, nil
}

// skipValue locates the end of a value in already validated JSON. Raw-property
// callers need its original bytes, not a normalized copy of every descendant.
// Keep this separate from value: numeric rounding, duplicate-key ordering and
// UTF-16 normalization must happen only when a caller actually decodes values.
func (p *jsonValueParser) skipValue() {
	p.space()
	switch p.raw[p.pos] {
	case '"':
		p.skipString()
	case '{', '[':
		depth := 0
		for {
			switch p.raw[p.pos] {
			case '"':
				p.skipString()
				continue
			case '{', '[':
				depth++
			case '}', ']':
				depth--
				if depth == 0 {
					p.pos++
					return
				}
			}
			p.pos++
		}
	default:
		p.literal()
	}
}

func (p *jsonValueParser) skipString() {
	p.pos++
	for p.raw[p.pos] != '"' {
		if p.raw[p.pos] == '\\' {
			p.pos++ // The escaped quote/backslash cannot end this string.
		}
		p.pos++
	}
	p.pos++
}
