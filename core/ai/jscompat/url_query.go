package jscompat

import (
	"strconv"
	"strings"

	"golang.org/x/text/encoding/unicode"
)

// URLSearchParams replaces plus before percent decoding and uses the WHATWG
// UTF-8 replacement algorithm, preserving malformed percent escapes literally.
func QueryComponent(value string) string {
	var decoded strings.Builder
	for i := 0; i < len(value); i++ {
		if value[i] == '+' {
			decoded.WriteByte(' ')
			continue
		}
		if value[i] == '%' && i+2 < len(value) {
			if n, err := strconv.ParseUint(value[i+1:i+3], 16, 8); err == nil {
				decoded.WriteByte(byte(n))
				i += 2
				continue
			}
		}
		decoded.WriteByte(value[i])
	}
	text, _ := unicode.UTF8.NewDecoder().String(decoded.String())
	return text
}

func QueryValues(query string) map[string]string {
	values := make(map[string]string)
	for _, pair := range strings.Split(query, "&") {
		if pair == "" {
			continue
		}
		key, value, _ := strings.Cut(pair, "=")
		key = QueryComponent(key)
		if _, exists := values[key]; !exists {
			values[key] = QueryComponent(value)
		}
	}
	return values
}
