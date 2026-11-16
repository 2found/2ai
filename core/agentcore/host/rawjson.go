package host

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

// A native transcript is stored as the JSON the host persisted. Reading it on
// every run must cost one pass over those bytes, not a decode of every
// message: the scanner below validates JSON with encoding/json's acceptance
// rules (invalid UTF-8 inside strings is accepted, control characters are
// not, nesting stops at 10000) and hands back sub-slices of the input, never
// copies.

// maxJSONDepth is encoding/json's nesting limit.
const maxJSONDepth = 10000

var errInvalidJSON = errors.New("invalid native JSON")

func isSpace(c byte) bool { return c == ' ' || c == '\t' || c == '\n' || c == '\r' }

func skipSpace(b []byte, i int) int {
	for i < len(b) && isSpace(b[i]) {
		i++
	}
	return i
}

// scanValue validates the JSON value at b[i:] (no leading space) and returns
// the index just past it.
func scanValue(b []byte, i, depth int) (int, error) {
	if i >= len(b) {
		return 0, errInvalidJSON
	}
	switch c := b[i]; {
	case c == '"':
		return scanString(b, i)
	case c == '{':
		if depth >= maxJSONDepth {
			return 0, errInvalidJSON
		}
		i = skipSpace(b, i+1)
		if i < len(b) && b[i] == '}' {
			return i + 1, nil
		}
		for {
			if i >= len(b) || b[i] != '"' {
				return 0, errInvalidJSON
			}
			end, err := scanString(b, i)
			if err != nil {
				return 0, err
			}
			i = skipSpace(b, end)
			if i >= len(b) || b[i] != ':' {
				return 0, errInvalidJSON
			}
			if i, err = scanValue(b, skipSpace(b, i+1), depth+1); err != nil {
				return 0, err
			}
			i = skipSpace(b, i)
			if i >= len(b) {
				return 0, errInvalidJSON
			}
			if b[i] == '}' {
				return i + 1, nil
			}
			if b[i] != ',' {
				return 0, errInvalidJSON
			}
			i = skipSpace(b, i+1)
		}
	case c == '[':
		if depth >= maxJSONDepth {
			return 0, errInvalidJSON
		}
		i = skipSpace(b, i+1)
		if i < len(b) && b[i] == ']' {
			return i + 1, nil
		}
		for {
			var err error
			if i, err = scanValue(b, i, depth+1); err != nil {
				return 0, err
			}
			i = skipSpace(b, i)
			if i >= len(b) {
				return 0, errInvalidJSON
			}
			if b[i] == ']' {
				return i + 1, nil
			}
			if b[i] != ',' {
				return 0, errInvalidJSON
			}
			i = skipSpace(b, i+1)
		}
	case c == '-' || (c >= '0' && c <= '9'):
		return scanNumber(b, i)
	case c == 't':
		return scanLiteral(b, i, "true")
	case c == 'f':
		return scanLiteral(b, i, "false")
	case c == 'n':
		return scanLiteral(b, i, "null")
	}
	return 0, errInvalidJSON
}

func scanLiteral(b []byte, i int, literal string) (int, error) {
	if len(b)-i < len(literal) || string(b[i:i+len(literal)]) != literal {
		return 0, errInvalidJSON
	}
	return i + len(literal), nil
}

func scanNumber(b []byte, i int) (int, error) {
	if b[i] == '-' {
		i++
	}
	digits := func() bool {
		start := i
		for i < len(b) && b[i] >= '0' && b[i] <= '9' {
			i++
		}
		return i > start
	}
	switch {
	case i < len(b) && b[i] == '0':
		i++
	case !digits():
		return 0, errInvalidJSON
	}
	if i < len(b) && b[i] == '.' {
		i++
		if !digits() {
			return 0, errInvalidJSON
		}
	}
	if i < len(b) && (b[i] == 'e' || b[i] == 'E') {
		i++
		if i < len(b) && (b[i] == '+' || b[i] == '-') {
			i++
		}
		if !digits() {
			return 0, errInvalidJSON
		}
	}
	return i, nil
}

func hexDigit(c byte) bool {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
}

// stringStop marks the bytes scanString must look at: the closing quote, an
// escape, and the control characters JSON forbids in a string.
var stringStop = func() (stop [256]bool) {
	for c := 0; c < 0x20; c++ {
		stop[c] = true
	}
	stop['"'], stop['\\'] = true, true
	return stop
}()

// scanString validates the string starting at the quote b[i].
func scanString(b []byte, i int) (int, error) {
	for i++; i < len(b); i++ {
		c := b[i]
		if !stringStop[c] {
			continue
		}
		switch {
		case c == '"':
			return i + 1, nil
		case c == '\\':
			i++
			if i >= len(b) {
				return 0, errInvalidJSON
			}
			switch b[i] {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
			case 'u':
				if len(b)-i < 5 || !hexDigit(b[i+1]) || !hexDigit(b[i+2]) || !hexDigit(b[i+3]) || !hexDigit(b[i+4]) {
					return 0, errInvalidJSON
				}
				i += 4
			default:
				return 0, errInvalidJSON
			}
		default:
			return 0, errInvalidJSON
		}
	}
	return 0, errInvalidJSON
}

// skipValue returns the index just past the already validated value at b[i].
func skipValue(b []byte, i int) int {
	switch b[i] {
	case '"':
		return skipString(b, i)
	case '{', '[':
		depth := 0
		for ; i < len(b); i++ {
			switch b[i] {
			case '"':
				i = skipString(b, i) - 1
			case '{', '[':
				depth++
			case '}', ']':
				if depth--; depth == 0 {
					return i + 1
				}
			}
		}
		return len(b)
	}
	for i < len(b) && b[i] != ',' && b[i] != '}' && b[i] != ']' && !isSpace(b[i]) {
		i++
	}
	return i
}

// skipString returns the index just past the validated string at b[i]: the
// first quote not escaped by an odd run of backslashes.
func skipString(b []byte, i int) int {
	for i++; ; i++ {
		j := bytes.IndexByte(b[i:], '"')
		if j < 0 {
			return len(b)
		}
		i += j
		k := i - 1
		for b[k] == '\\' { // the opening quote ends the run
			k--
		}
		if (i-1-k)%2 == 0 {
			return i + 1
		}
	}
}

// eachMember calls visit with the raw key (between its quotes) and the value
// of every member of the validated object at b[i:], in order, until visit
// returns false.
func eachMember(b []byte, visit func(key, value []byte) bool) {
	i := skipSpace(b, 0)
	if i >= len(b) || b[i] != '{' {
		return
	}
	i = skipSpace(b, i+1)
	for i < len(b) && b[i] == '"' {
		end := skipString(b, i)
		key := b[i+1 : end-1]
		i = skipSpace(b, end)
		i = skipSpace(b, i+1) // ':'
		valueEnd := skipValue(b, i)
		if !visit(key, b[i:valueEnd]) {
			return
		}
		i = skipSpace(b, valueEnd)
		if i < len(b) && b[i] == ',' {
			i = skipSpace(b, i+1)
		}
	}
}

// eachElement calls visit with every element of the validated array b.
func eachElement(b []byte, visit func(value []byte) bool) {
	i := skipSpace(b, 0)
	if i >= len(b) || b[i] != '[' {
		return
	}
	i = skipSpace(b, i+1)
	for i < len(b) && b[i] != ']' {
		end := skipValue(b, i)
		if !visit(b[i:end]) {
			return
		}
		i = skipSpace(b, end)
		if i < len(b) && b[i] == ',' {
			i = skipSpace(b, i+1)
		}
	}
}

// SplitMessages validates raw as a JSON array and returns its elements as
// sub-slices of raw: the exact bytes each message was stored as, without
// copying or decoding them. It fails where json.Unmarshal into a
// []json.RawMessage fails, and on a null that is no array.
func SplitMessages(raw json.RawMessage) ([]json.RawMessage, error) {
	i := skipSpace(raw, 0)
	if i >= len(raw) || raw[i] != '[' {
		return nil, fmt.Errorf("native transcript must be an array")
	}
	// Count first so the result is one allocation of its final size.
	count := 0
	end, err := scanArrayElements(raw, i, func(int, int) { count++ })
	if err != nil {
		return nil, err
	}
	if skipSpace(raw, end) != len(raw) {
		return nil, errInvalidJSON
	}
	messages := make([]json.RawMessage, 0, count)
	eachElement(raw[i:end], func(value []byte) bool {
		messages = append(messages, value[:len(value):len(value)])
		return true
	})
	return messages, nil
}

// scanArrayElements validates the array at b[i] and reports each element's
// span.
func scanArrayElements(b []byte, i int, element func(start, end int)) (int, error) {
	i = skipSpace(b, i+1)
	if i < len(b) && b[i] == ']' {
		return i + 1, nil
	}
	for {
		start := i
		end, err := scanValue(b, i, 1)
		if err != nil {
			return 0, err
		}
		element(start, end)
		i = skipSpace(b, end)
		if i >= len(b) {
			return 0, errInvalidJSON
		}
		if b[i] == ']' {
			return i + 1, nil
		}
		if b[i] != ',' {
			return 0, errInvalidJSON
		}
		i = skipSpace(b, i+1)
	}
}

// LocateMember validates raw as one JSON object, except the value of its
// member name, and returns the span of that value. The caller must validate
// that value: when raw is not JSON the span may be anything. ok is false when
// the member is absent, appears more than once, or another key would also
// match it under encoding/json's case-insensitive field matching: decode raw
// with encoding/json then.
func LocateMember(raw json.RawMessage, name string) (start, end int, ok bool, err error) {
	i := skipSpace(raw, 0)
	if i >= len(raw) || raw[i] != '{' {
		if _, err := scanValue(raw, i, 0); err != nil {
			return 0, 0, false, err
		}
		return 0, 0, false, nil
	}
	found, clash := 0, false
	i = skipSpace(raw, i+1)
	if i < len(raw) && raw[i] == '}' {
		i++
	} else {
		for {
			if i >= len(raw) || raw[i] != '"' {
				return 0, 0, false, errInvalidJSON
			}
			keyEnd, err := scanString(raw, i)
			if err != nil {
				return 0, 0, false, err
			}
			key := raw[i+1 : keyEnd-1]
			i = skipSpace(raw, keyEnd)
			if i >= len(raw) || raw[i] != ':' {
				return 0, 0, false, errInvalidJSON
			}
			i = skipSpace(raw, i+1)
			if i >= len(raw) {
				return 0, 0, false, errInvalidJSON
			}
			var valueEnd int
			if string(key) == name {
				// The caller validates this value itself; skipping it keeps
				// the transcript to one validating pass.
				valueEnd = skipValue(raw, i)
				found++
				start, end = i, valueEnd
			} else {
				if valueEnd, err = scanValue(raw, i, 1); err != nil {
					return 0, 0, false, err
				}
				clash = clash || foldsTo(key, name)
			}
			i = skipSpace(raw, valueEnd)
			if i >= len(raw) {
				return 0, 0, false, errInvalidJSON
			}
			if raw[i] == '}' {
				i++
				break
			}
			if raw[i] != ',' {
				return 0, 0, false, errInvalidJSON
			}
			i = skipSpace(raw, i+1)
		}
	}
	if skipSpace(raw, i) != len(raw) {
		return 0, 0, false, errInvalidJSON
	}
	return start, end, found == 1 && !clash, nil
}

// foldsTo reports whether key, which is not spelled exactly as name, might
// still match name under encoding/json: any escape or non-ASCII byte, or an
// ASCII case-insensitive match.
func foldsTo(key []byte, name string) bool {
	for _, c := range key {
		if c == '\\' || c >= 0x80 {
			return true
		}
	}
	if len(key) != len(name) {
		return false
	}
	for i := range key {
		a, b := key[i], name[i]
		if 'A' <= a && a <= 'Z' {
			a += 'a' - 'A'
		}
		if 'A' <= b && b <= 'Z' {
			b += 'a' - 'A'
		}
		if a != b {
			return false
		}
	}
	return true
}
