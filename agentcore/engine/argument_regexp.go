package engine

import (
	"errors"
	"fmt"
	"strings"

	"github.com/2found/2ai/jsonjs"

	"github.com/dlclark/regexp2"
	"github.com/santhosh-tekuri/jsonschema/v6"
)

// Pi creates RegExp(pattern, "u"). Keep the original pattern for diagnostics
// while enabling ECMAScript syntax and Unicode code-point matching in Go.
type argumentRegexp struct {
	compiled *regexp2.Regexp
	source   string
}

func (pattern *argumentRegexp) MatchString(value string) bool {
	matched, err := pattern.compiled.MatchRunes(jsonjs.StringCodePoints(value))
	return err == nil && matched
}

func (pattern *argumentRegexp) String() string { return pattern.source }

func compileArgumentRegexp(pattern string) (jsonschema.Regexp, error) {
	// regexp2's string decoder cannot represent lone surrogate code points.
	// Escape literal surrogates before compilation, but keep their original
	// spelling for diagnostics. The pinned runtime also accepts a backslash
	// before a non-ASCII identity character, including a lone surrogate.
	var normalized strings.Builder
	escaped := false
	for _, point := range jsonjs.StringCodePoints(pattern) {
		if point >= 0xd800 && point <= 0xdfff {
			if !escaped {
				normalized.WriteByte('\\')
			}
			fmt.Fprintf(&normalized, "u%04x", point)
		} else {
			normalized.WriteRune(point)
		}
		if point == '\\' {
			escaped = !escaped
		} else {
			escaped = false
		}
	}
	syntax := argumentRegexpSyntax{text: normalized.String()}
	if !syntax.parse() {
		return nil, errors.New("invalid ECMAScript Unicode regular expression")
	}
	compiled, err := regexp2.Compile(syntax.lower(), regexp2.ECMAScript|regexp2.Unicode)
	if err != nil {
		return nil, err
	}
	return &argumentRegexp{compiled: compiled, source: pattern}, nil
}
