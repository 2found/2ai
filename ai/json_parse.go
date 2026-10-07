package ai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/2found/2ai/jsonjs"
)

// RepairJSON ports Pi's repairJson: it only repairs control characters and
// invalid escapes inside quoted strings, leaving syntax outside them intact.
func RepairJSON(input string) string {
	var out strings.Builder
	inString := false
	for i := 0; i < len(input); i++ {
		c := input[i]
		if !inString {
			out.WriteByte(c)
			inString = c == '"'
			continue
		}
		if c == '"' {
			out.WriteByte(c)
			inString = false
			continue
		}
		if c == '\\' {
			if i+1 == len(input) {
				out.WriteString(`\\`)
				continue
			}
			next := input[i+1]
			if next == 'u' && i+6 <= len(input) && isJSONHex(input[i+2:i+6]) {
				out.WriteString(input[i : i+6])
				i += 5
				continue
			}
			if strings.ContainsRune(`"\/bfnrtu`, rune(next)) {
				out.WriteByte(c)
				out.WriteByte(next)
				i++
				continue
			}
			out.WriteString(`\\`)
			continue
		}
		switch c {
		case '\b':
			out.WriteString(`\b`)
		case '\f':
			out.WriteString(`\f`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		default:
			if c < 32 {
				fmt.Fprintf(&out, `\u%04x`, c)
			} else {
				out.WriteByte(c)
			}
		}
	}
	return out.String()
}

func isJSONHex(text string) bool {
	for _, c := range text {
		if !strings.ContainsRune("0123456789abcdefABCDEF", c) {
			return false
		}
	}
	return true
}

// ParseJSONWithRepair returns the parsed JSON representation, retaining object
// member order. Callers can unmarshal it into their desired Go type. Invalid
// syntax still fails; this function does not complete unfinished documents.
func ParseJSONWithRepair(input string) (json.RawMessage, error) {
	parsed, err := parseCompleteJSON(input)
	if err == nil {
		return parsed, nil
	}
	if repaired := RepairJSON(input); repaired != input {
		return parseCompleteJSON(repaired)
	}
	return nil, err
}

func parseCompleteJSON(input string) (json.RawMessage, error) {
	value := json.RawMessage(input)
	if err := jsonjs.ValidateJSON(value); err != nil {
		return nil, err
	}
	// Pi parses numbers as IEEE-754 and JSON.stringify renders overflow as
	// null. This also applies JS object-key and duplicate-key semantics.
	return StringifyJSON(value)
}

// ParseStreamingJSON ports Pi's parseStreamingJson, including the permissive
// partial-json 0.1.7 fallback. Its result is JSON, not necessarily an object:
// complete scalar/null/array inputs retain their values. Unparseable input and
// a partial null resolve to {}. No errors escape this streaming helper.
func ParseStreamingJSON(input string) json.RawMessage {
	if strings.TrimFunc(input, jsWhitespace) == "" {
		return json.RawMessage(`{}`)
	}
	if result, err := ParseJSONWithRepair(input); err == nil {
		return result
	}
	for _, candidate := range []string{input, RepairJSON(input)} {
		p := partialJSONParser{input: strings.TrimFunc(candidate, jsWhitespace)}
		if result, err := p.value(); err == nil {
			if bytes.Equal(bytes.TrimSpace(result), []byte("null")) && p.nullValue {
				return json.RawMessage(`{}`)
			}
			return result
		}
	}
	return json.RawMessage(`{}`)
}

// Algorithm translated from partial-json 0.1.7 (MIT, Promplate Dev Team).
// Its copyright notice is preserved in LICENSE.partial-json beside this file.
// The intentionally lenient colon/comma and exponent handling match that
// dependency; changing them to a stricter parser changes observable tool args.
type partialJSONParser struct {
	input     string
	index     int
	nullValue bool
}

var errPartialJSON = errors.New("incomplete or malformed JSON")

func (p *partialJSONParser) char() byte {
	if p.index >= len(p.input) {
		return 0
	}
	return p.input[p.index]
}
func (p *partialJSONParser) skipBlank() {
	for p.index < len(p.input) && strings.ContainsRune(" \n\r\t", rune(p.input[p.index])) {
		p.index++
	}
}
func (p *partialJSONParser) value() (json.RawMessage, error) {
	p.skipBlank()
	if p.index >= len(p.input) {
		return nil, errPartialJSON
	}
	p.nullValue = false
	switch p.char() {
	case '"':
		return p.str()
	case '{':
		return p.object()
	case '[':
		return p.array()
	}
	rest := p.input[p.index:]
	for _, literal := range []string{"null", "true", "false", "Infinity", "-Infinity", "NaN"} {
		partial := len(rest) < len(literal) && strings.HasPrefix(literal, rest)
		if literal == "-Infinity" && len(rest) <= 1 {
			partial = false
		}
		if strings.HasPrefix(rest, literal) || partial {
			p.index += len(literal)
			p.nullValue = literal == "null"
			if literal == "Infinity" || literal == "-Infinity" || literal == "NaN" {
				return json.RawMessage(`null`), nil
			}
			return json.RawMessage(literal), nil
		}
	}
	return p.number()
}

func (p *partialJSONParser) str() (json.RawMessage, error) {
	start := p.index
	escape := false
	p.index++
	for p.index < len(p.input) && (p.char() != '"' || (escape && p.input[p.index-1] == '\\')) {
		if p.char() == '\\' {
			escape = !escape
		} else {
			escape = false
		}
		p.index++
	}
	escaped := 0
	if escape {
		escaped = 1
	}
	if p.index < len(p.input) && p.char() == '"' {
		p.index++
		return parseCompleteJSON(p.input[start : p.index-escaped])
	}
	end := p.index - escaped
	if start > end || end > len(p.input) {
		return nil, errPartialJSON
	}
	if value, err := parseCompleteJSON(p.input[start:end] + `"`); err == nil {
		return value, nil
	}
	// JavaScript substring swaps reversed bounds and clamps negatives to zero.
	return parseCompleteJSON(jsSubstring(p.input, start, strings.LastIndex(p.input, `\`)) + `"`)
}

func jsSubstring(text string, start, end int) string {
	if start < 0 {
		start = 0
	}
	if end < 0 {
		end = 0
	}
	if start > len(text) {
		start = len(text)
	}
	if end > len(text) {
		end = len(text)
	}
	if end < start {
		start, end = end, start
	}
	return text[start:end]
}

func (p *partialJSONParser) object() (json.RawMessage, error) {
	p.index++
	p.skipBlank()
	fields := jsonjs.ObjectFields{}
	finish := func() (json.RawMessage, error) {
		p.nullValue = false
		return fields.Marshal(), nil
	}
	for p.char() != '}' {
		p.skipBlank()
		if p.index >= len(p.input) {
			return finish()
		}
		key, err := p.str()
		if err != nil {
			return finish()
		}
		if len(key) < 2 || key[0] != '"' {
			return finish()
		}
		p.skipBlank()
		p.index++ // The dependency skips one character for ':' without validating it.
		value, err := p.value()
		if err != nil {
			return finish()
		}
		// The JS dependency assigns onto {}, so __proto__ invokes its setter
		// rather than becoming a JSON property. Complete JSON.parse does not.
		if string(key) != `"__proto__"` {
			fields.Set(key, value)
		}
		p.skipBlank()
		if p.char() == ',' {
			p.index++
		}
	}
	p.index++
	return finish()
}

func (p *partialJSONParser) array() (json.RawMessage, error) {
	p.index++
	values := [][]byte{}
	for p.char() != ']' {
		value, err := p.value()
		if err != nil {
			break
		}
		values = append(values, value)
		p.skipBlank()
		if p.char() == ',' {
			p.index++
		}
	}
	if p.char() == ']' {
		p.index++
	}
	p.nullValue = false
	return append(append([]byte{'['}, bytes.Join(values, []byte{','})...), ']'), nil
}

func (p *partialJSONParser) number() (json.RawMessage, error) {
	if p.index == 0 {
		if p.input == "-" {
			return nil, errPartialJSON
		}
		if value, err := parseCompleteJSON(p.input); err == nil {
			return value, nil
		}
		return parseCompleteJSON(jsSubstring(p.input, 0, strings.LastIndex(p.input, "e")))
	}
	start := p.index
	if p.char() == '-' {
		p.index++
	}
	for p.index < len(p.input) && !strings.ContainsRune(",]}", rune(p.char())) {
		p.index++
	}
	value := p.input[start:p.index]
	if result, err := parseCompleteJSON(value); err == nil {
		return result, nil
	}
	if value == "-" {
		return nil, errPartialJSON
	}
	return parseCompleteJSON(jsSubstring(p.input, start, strings.LastIndex(p.input, "e")))
}
