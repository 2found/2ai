package host

import (
	"bytes"
	"encoding/json"
	"strconv"
	"unicode/utf8"
)

// factFields are the message members decodeFacts reads, as 2ai writes them.
// encoding/json matches struct fields case-insensitively; any other spelling
// of these names leaves the message to the decoding path.
var factFields = [...]string{"role", "toolCallId", "toolName", "stopReason", "isError", "timestamp", "agentrayContextReducedAt", "agentrayContextSummary", "content", "usage"}

const (
	fieldRole = iota
	fieldToolCallID
	fieldToolName
	fieldStopReason
	fieldIsError
	fieldTimestamp
	fieldReducedAt
	fieldSummary
	fieldContent
	fieldUsage
)

var usageFields = [...]string{"input", "output", "cacheRead", "cacheWrite"}

var blockFields = [...]string{"type", "id", "arguments"}

// fieldIndex is the index of key in names, -1 for an unrelated key, or -2 for a
// key encoding/json might match to one of names although it is spelled
// differently.
func fieldIndex(key []byte, names []string) int {
	for i, name := range names {
		if string(key) == name {
			return i
		}
	}
	for _, name := range names {
		if foldsTo(key, name) {
			return -2
		}
	}
	return -1
}

func isNull(value []byte) bool { return len(value) == 4 && value[0] == 'n' }

// plainString is the content of a JSON string value that needs no decoding:
// no escape and valid UTF-8, so encoding/json would read exactly these bytes.
func plainString(value []byte) ([]byte, bool) {
	if len(value) < 2 || value[0] != '"' {
		return nil, false
	}
	body := value[1 : len(value)-1]
	if bytes.IndexByte(body, '\\') >= 0 || !utf8.Valid(body) {
		return nil, false
	}
	return body, true
}

// internString avoids allocating the strings every message repeats.
func internString(body []byte) string {
	switch string(body) {
	case "":
		return ""
	case "user":
		return "user"
	case "assistant":
		return "assistant"
	case "toolResult":
		return "toolResult"
	case "system":
		return "system"
	case "stop":
		return "stop"
	case "toolUse":
		return "toolUse"
	case "length":
		return "length"
	case "error":
		return "error"
	case "aborted":
		return "aborted"
	}
	return string(body)
}

// plainInt parses an integer literal as encoding/json reads it into an int64.
func plainInt(value []byte) (int64, bool) {
	if isNull(value) {
		return 0, true
	}
	i, negative := 0, false
	if len(value) > 0 && value[0] == '-' {
		i, negative = 1, true
	}
	if i >= len(value) {
		return 0, false
	}
	var n uint64
	for ; i < len(value); i++ {
		c := value[i]
		if c < '0' || c > '9' {
			return 0, false
		}
		if n > (1<<63)/10 {
			return 0, false
		}
		n = n*10 + uint64(c-'0')
	}
	if negative {
		if n > 1<<63 {
			return 0, false
		}
		return -int64(n), true
	}
	if n > 1<<63-1 {
		return 0, false
	}
	return int64(n), true
}

// plainFloat parses a number as encoding/json reads it into a float64.
func plainFloat(value []byte) (float64, bool) {
	if isNull(value) {
		return 0, true
	}
	if n, ok := plainInt(value); ok && n > -(1<<53) && n < 1<<53 {
		return float64(n), true
	}
	if len(value) == 0 || (value[0] != '-' && (value[0] < '0' || value[0] > '9')) {
		return 0, false
	}
	f, err := strconv.ParseFloat(string(value), 64)
	return f, err == nil
}

// scanFacts is decodeFacts without decoding: one pass over a message in the
// shape 2ai writes, reading sub-slices of raw. ok is false for anything that
// shape does not cover (escaped or differently spelled member names, a
// repeated member, a value of another type), and the caller decodes the
// message instead. For an ok message, validateMessage's decode cannot fail.
func scanFacts(raw json.RawMessage) (f messageFacts, ok bool) {
	if len(raw) == 0 || raw[0] != '{' {
		return f, false
	}
	var values [len(factFields)][]byte
	ok = true
	eachMember(raw, func(key, value []byte) bool {
		index := fieldIndex(key, factFields[:])
		switch {
		case index == -1:
			return true
		case index < 0 || values[index] != nil:
			ok = false
			return false
		}
		values[index] = value
		return true
	})
	if !ok {
		return f, false
	}
	text := func(index int) (string, bool) {
		value := values[index]
		if value == nil || isNull(value) {
			return "", true
		}
		body, ok := plainString(value)
		if !ok {
			return "", false
		}
		return internString(body), true
	}
	if f.role, ok = text(fieldRole); !ok {
		return f, false
	}
	if f.toolCallID, ok = text(fieldToolCallID); !ok {
		return f, false
	}
	if f.toolName, ok = text(fieldToolName); !ok {
		return f, false
	}
	if f.stopReason, ok = text(fieldStopReason); !ok {
		return f, false
	}
	switch value := values[fieldIsError]; {
	case value == nil || isNull(value) || string(value) == "false":
	case string(value) == "true":
		f.isError = true
	default:
		return f, false
	}
	if value := values[fieldTimestamp]; value != nil {
		if f.timestamp, ok = plainInt(value); !ok {
			return f, false
		}
	}
	if value := values[fieldReducedAt]; value != nil {
		if f.reducedAt, ok = plainInt(value); !ok {
			return f, false
		}
	}
	switch value := values[fieldSummary]; {
	case value == nil || isNull(value):
	case value[0] == '"':
		f.summary = len(value) > 2
	default:
		return f, false
	}
	if value := values[fieldUsage]; value != nil && !isNull(value) {
		if value[0] != '{' {
			return f, false
		}
		var parts [len(usageFields)][]byte
		eachMember(value, func(key, value []byte) bool {
			index := fieldIndex(key, usageFields[:])
			switch {
			case index == -1:
				return true
			case index < 0 || parts[index] != nil:
				ok = false
				return false
			}
			parts[index] = value
			return true
		})
		if !ok {
			return f, false
		}
		for _, part := range parts {
			if part != nil {
				n, ok := plainFloat(part)
				if !ok {
					return f, false
				}
				f.usage += n
			}
		}
		f.hasUsage = true
	}
	if f.role == "assistant" && f.stopReason != "error" && f.stopReason != "aborted" {
		content := values[fieldContent]
		if content == nil || (!isNull(content) && content[0] != '[') {
			return f, false
		}
		eachElement(content, func(block []byte) bool {
			if isNull(block) {
				return true
			}
			if block[0] != '{' {
				ok = false
				return false
			}
			var parts [len(blockFields)][]byte
			eachMember(block, func(key, value []byte) bool {
				index := fieldIndex(key, blockFields[:])
				switch {
				case index == -1:
					return true
				case index < 0 || parts[index] != nil:
					ok = false
					return false
				}
				parts[index] = value
				return true
			})
			if !ok {
				return false
			}
			var kind, id []byte
			if value := parts[0]; value != nil && !isNull(value) {
				if kind, ok = plainString(value); !ok {
					return false
				}
			}
			if value := parts[1]; value != nil && !isNull(value) {
				if id, ok = plainString(value); !ok {
					return false
				}
			}
			if string(kind) == "toolCall" {
				callID := string(id)
				f.calls = append(f.calls, callID)
				if f.callArgs == nil {
					f.callArgs = map[string]json.RawMessage{}
				}
				var args json.RawMessage
				if parts[2] != nil {
					args = append(json.RawMessage(nil), parts[2]...)
				}
				f.callArgs[callID] = args
			}
			return true
		})
		if !ok {
			return messageFacts{}, false
		}
	}
	f.bytes, f.raw = len(raw), raw
	return f, true
}

// factsOf is decodeFacts, read without decoding when the message allows it.
func factsOf(raw json.RawMessage) messageFacts {
	if f, ok := scanFacts(raw); ok {
		return f
	}
	return decodeFacts(raw)
}
