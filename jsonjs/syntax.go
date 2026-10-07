package jsonjs

import (
	"encoding/json"
	"errors"
	"strings"
	"unicode/utf16"
)

// ValidateJSON checks strict JSON syntax using the pinned Bun/JavaScriptCore
// diagnostics. It does not repair input, coerce values, or construct a graph.
// The explicit parser stack also avoids recursive descent on nested input.
func ValidateJSON(raw []byte) error {
	if json.Valid(raw) {
		return nil
	}
	lexer := jsonSyntaxLexer{}
	for _, r := range StringCodePoints(string(raw)) {
		if r > 0xffff {
			high, low := utf16.EncodeRune(r)
			lexer.units = append(lexer.units, uint16(high), uint16(low))
		} else {
			lexer.units = append(lexer.units, uint16(r))
		}
	}
	const (
		rootValue byte = iota
		rootEnd
		arrayFirst
		arrayValue
		arrayEnd
		objectFirst
		objectKey
		objectColon
		objectValue
		objectEnd
	)
	stack := []byte{rootValue}
	for {
		token, err := lexer.next()
		if err != nil {
			return err
		}
		last := len(stack) - 1
		switch stack[last] {
		case rootEnd:
			if token.kind == 0 {
				return nil
			}
			return jsonSyntaxError("Unable to parse JSON string")
		case arrayEnd:
			if token.kind == ']' {
				stack = stack[:last]
			} else if token.kind == ',' {
				stack[last] = arrayValue
			} else {
				return jsonSyntaxError("Expected ']'")
			}
			continue
		case objectEnd:
			if token.kind == '}' {
				stack = stack[:last]
			} else if token.kind == ',' {
				stack[last] = objectKey
			} else {
				return jsonSyntaxError("Expected '}'")
			}
			continue
		case objectFirst, objectKey:
			if token.kind == '}' && stack[last] == objectFirst {
				stack = stack[:last]
				continue
			}
			if token.kind != 's' {
				if stack[last] == objectFirst {
					return jsonSyntaxError("Expected '}'")
				}
				return jsonSyntaxError("Property name must be a string literal")
			}
			stack[last] = objectColon
			continue
		case objectColon:
			if token.kind != ':' {
				return jsonSyntaxError("Expected ':' before value in object property definition")
			}
			stack[last] = objectValue
			continue
		case arrayFirst:
			if token.kind == ']' {
				stack = stack[:last]
				continue
			}
			stack[last] = arrayEnd
		case arrayValue:
			if token.kind == ']' {
				return jsonSyntaxError("Unexpected comma at the end of array expression")
			}
			stack[last] = arrayEnd
		case objectValue:
			stack[last] = objectEnd
		case rootValue:
			stack[last] = rootEnd
		}
		switch token.kind {
		case 's', 'v':
		case '[':
			stack = append(stack, arrayFirst)
		case '{':
			stack = append(stack, objectFirst)
		case 0:
			return jsonSyntaxError("Unexpected EOF")
		case 'i':
			return jsonSyntaxError(`Unexpected identifier "` + token.text + `"`)
		default:
			return jsonSyntaxError("Unexpected token '" + string(token.kind) + "'")
		}
	}
}

func jsonSyntaxError(message string) error { return errors.New("JSON Parse error: " + message) }

type jsonSyntaxToken struct {
	kind byte
	text string
}
type jsonSyntaxLexer struct {
	units []uint16
	pos   int
}

func (l *jsonSyntaxLexer) next() (jsonSyntaxToken, error) {
	for l.pos < len(l.units) && strings.ContainsRune(" \t\n\r", rune(l.units[l.pos])) {
		l.pos++
	}
	if l.pos == len(l.units) {
		return jsonSyntaxToken{}, nil
	}
	start, c := l.pos, l.units[l.pos]
	l.pos++
	if c == '"' {
		for l.pos < len(l.units) {
			c = l.units[l.pos]
			l.pos++
			if c == '"' {
				return jsonSyntaxToken{kind: 's'}, nil
			}
			if c < 32 {
				return jsonSyntaxToken{}, jsonSyntaxError("Unterminated string")
			}
			if c != '\\' {
				continue
			}
			if l.pos == len(l.units) {
				break
			}
			c = l.units[l.pos]
			l.pos++
			if strings.ContainsRune(`"\/bfnrt`, rune(c)) {
				continue
			}
			if c != 'u' {
				return jsonSyntaxToken{}, jsonSyntaxError("Invalid escape character " + stringFromUTF16([]uint16{c}))
			}
			if len(l.units)-l.pos < 4 {
				return jsonSyntaxToken{}, jsonSyntaxError(`\u must be followed by 4 hex digits`)
			}
			for _, hex := range l.units[l.pos : l.pos+4] {
				if !strings.ContainsRune("0123456789abcdefABCDEF", rune(hex)) {
					return jsonSyntaxToken{}, jsonSyntaxError(`"\u` + stringFromUTF16(l.units[l.pos:l.pos+4]) + `" is not a valid unicode escape`)
				}
			}
			l.pos += 4
		}
		return jsonSyntaxToken{}, jsonSyntaxError("Unterminated string")
	}
	if c == '-' || syntaxDigit(c) {
		if c == '-' {
			if l.pos == len(l.units) || !syntaxDigit(l.units[l.pos]) {
				return jsonSyntaxToken{}, jsonSyntaxError("Invalid number")
			}
			c = l.units[l.pos]
			l.pos++
		}
		if c != '0' {
			for l.pos < len(l.units) && syntaxDigit(l.units[l.pos]) {
				l.pos++
			}
		}
		if l.pos < len(l.units) && l.units[l.pos] == '.' {
			l.pos++
			if l.pos == len(l.units) || !syntaxDigit(l.units[l.pos]) {
				return jsonSyntaxToken{}, jsonSyntaxError("Invalid digits after decimal point")
			}
			for l.pos < len(l.units) && syntaxDigit(l.units[l.pos]) {
				l.pos++
			}
		}
		if l.pos < len(l.units) && (l.units[l.pos] == 'e' || l.units[l.pos] == 'E') {
			exponent := l.pos
			l.pos++
			if l.pos < len(l.units) && (l.units[l.pos] == '+' || l.units[l.pos] == '-') {
				l.pos++
			}
			digits := l.pos
			for l.pos < len(l.units) && syntaxDigit(l.units[l.pos]) {
				l.pos++
			}
			if l.pos == digits {
				l.pos = exponent
			}
		}
		return jsonSyntaxToken{kind: 'v'}, nil
	}
	if syntaxIdentifier(c) {
		for _, literal := range []string{"true", "false", "null"} {
			if len(l.units)-start >= len(literal) && stringFromUTF16(l.units[start:start+len(literal)]) == literal {
				l.pos = start + len(literal)
				return jsonSyntaxToken{kind: 'v'}, nil
			}
		}
		for l.pos < len(l.units) && (syntaxIdentifier(l.units[l.pos]) || syntaxDigit(l.units[l.pos])) {
			l.pos++
		}
		return jsonSyntaxToken{kind: 'i', text: stringFromUTF16(l.units[start:l.pos])}, nil
	}
	if c == '\'' {
		return jsonSyntaxToken{}, jsonSyntaxError("Single quotes (') are not allowed in JSON")
	}
	if strings.ContainsRune("{}[]():,;.=", rune(c)) {
		return jsonSyntaxToken{kind: byte(c)}, nil
	}
	return jsonSyntaxToken{}, jsonSyntaxError("Unrecognized token '" + stringFromUTF16([]uint16{c}) + "'")
}

func syntaxDigit(c uint16) bool { return c >= '0' && c <= '9' }
func syntaxIdentifier(c uint16) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '_' || c == '$'
}
