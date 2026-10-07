package ai

import (
	"net/url"
	"strings"

	"github.com/2found/2ai/jsonjs"
)

// oauthFormEncode retains insertion order and URLSearchParams' USVString
// conversion, including one replacement character per lone UTF-16 surrogate.
func oauthFormEncode(fields ...Property) (string, error) {
	parts := make([]string, 0, len(fields))
	encode := func(s string) string {
		return strings.NewReplacer("%2A", "*", "~", "%7E").Replace(url.QueryEscape(string(jsonjs.StringCodePoints(s))))
	}
	for _, field := range fields {
		value, err := catalogKey(field.Value, make(map[*Array]bool))
		if err != nil {
			return "", err
		}
		parts = append(parts, encode(field.Name)+"="+encode(value))
	}
	return strings.Join(parts, "&"), nil
}
