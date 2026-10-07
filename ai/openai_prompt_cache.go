package ai

import (
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"strings"

	"github.com/2found/2ai/jsonjs"
)

const OpenAIPromptCacheKeyMaxLength = 64

// ClampOpenAIPromptCacheKey counts Unicode code points as Pi does with Array.from.
// A nil raw input represents an absent key. For serialized non-string inputs it
// retains the original value when the array-like length does not exceed the limit.
func ClampOpenAIPromptCacheKey(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	value, err := jsonjs.DecodeJSON(raw)
	if err != nil {
		return nil, err
	}
	if value == nil {
		return nil, errors.New("Array.from requires an array-like object - not null or undefined")
	}
	var values []any
	switch value := value.(type) {
	case string:
		chars := jsonjs.StringCodePoints(value)
		if len(chars) <= OpenAIPromptCacheKeyMaxLength {
			return raw, nil
		}
		var out strings.Builder
		for _, c := range chars[:OpenAIPromptCacheKeyMaxLength] {
			if c >= 0xd800 && c <= 0xdfff {
				out.Write([]byte{0xe0 | byte(c>>12), 0x80 | byte(c>>6&0x3f), 0x80 | byte(c&0x3f)})
			} else {
				out.WriteRune(c)
			}
		}
		return jsonjs.QuoteString(out.String()), nil
	case []any:
		if len(value) <= OpenAIPromptCacheKeyMaxLength {
			return raw, nil
		}
		values = value[:OpenAIPromptCacheKeyMaxLength]
	case map[string]any:
		var length float64
		switch v := value["length"].(type) {
		case float64:
			length = v
		case string:
			length = ParseJSNumber(v)
		case []any:
			length = ParseJSNumber(promptCacheElementString(v))
		}
		if math.IsNaN(length) || math.Floor(length) <= OpenAIPromptCacheKeyMaxLength {
			return raw, nil
		}
		values = make([]any, OpenAIPromptCacheKeyMaxLength)
		for i := range values {
			values[i] = value[strconv.Itoa(i)]
		}
	default:
		return raw, nil
	}
	var out strings.Builder
	for _, value := range values {
		if value != nil {
			out.WriteString(promptCacheElementString(value))
		}
	}
	return jsonjs.QuoteString(out.String()), nil
}

func promptCacheElementString(value any) string {
	switch value := value.(type) {
	case nil:
		return ""
	case string:
		return value
	case map[string]any:
		return "[object Object]"
	case []any:
		parts := make([]string, len(value))
		for i, item := range value {
			parts[i] = promptCacheElementString(item)
		}
		return strings.Join(parts, ",")
	case float64:
		if math.IsInf(value, 1) {
			return "Infinity"
		}
		if math.IsInf(value, -1) {
			return "-Infinity"
		}
	}
	raw, _ := jsonjs.StringifyValue(value)
	return string(raw)
}
