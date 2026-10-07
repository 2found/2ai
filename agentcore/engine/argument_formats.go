package engine

import (
	"strconv"
	"strings"

	"github.com/dlclark/regexp2"
	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
	"golang.org/x/text/unicode/norm"
)

// Formats are assertions in Pi regardless of a schema's declared dialect.
// Use its checks rather than enabling the Go library's different format set.
type argumentFormat struct{ name string }

func (rule *argumentFormat) Validate(ctx *jsonschema.ValidatorContext, value any) {
	text, ok := value.(string)
	if !ok || argumentFormatMatches(rule.name, text) {
		return
	}
	ctx.AddError(&kind.Format{Got: value, Want: rule.name})
}

func registerArgumentFormats(compiler *jsonschema.Compiler) {
	compiler.RegisterVocabulary(&jsonschema.Vocabulary{
		URL: "https://agentcore.local/pi-format-semantics",
		Compile: func(_ *jsonschema.CompilerContext, obj map[string]any) (jsonschema.SchemaExt, error) {
			name, ok := obj["format"].(string)
			if !ok {
				return nil, nil
			}
			return &argumentFormat{name: name}, nil
		},
	})
}

func argumentFormatMatches(name, value string) bool {
	switch name {
	case "url", "iri", "iri-reference":
		return argumentURL(name, value)
	case "hostname":
		return argumentHostname(value, false)
	case "idn-hostname":
		return argumentHostname(value, true)
	case "regex":
		_, err := compileArgumentRegexp(value)
		return err == nil
	case "date-time":
		parts := strings.FieldsFunc(value, func(r rune) bool { return r == 'T' || r == 't' })
		return len(parts) == 2 && strings.Count(value, "T")+strings.Count(value, "t") == 1 && argumentFormatMatches("date", parts[0]) && argumentFormatMatches("time", parts[1])
	case "idn-email":
		value = norm.NFC.String(value)
	}
	pattern := argumentFormatPatterns[name]
	if pattern == nil {
		// Pi ignores unknown format names rather than rejecting the schema.
		return true
	}
	match, err := pattern.FindStringMatch(value)
	if err != nil || match == nil {
		return false
	}
	number := func(index int) int {
		value, _ := strconv.Atoi(match.GroupByNumber(index).String())
		return value
	}
	switch name {
	case "date":
		year, month, day := number(1), number(2), number(3)
		if month < 1 || month > 12 || day < 1 {
			return false
		}
		days := [...]int{0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
		if month == 2 && year%4 == 0 && (year%100 != 0 || year%400 == 0) {
			return day <= 29
		}
		return day <= days[month]
	case "time":
		hour, minute, second, zoneHour, zoneMinute := number(1), number(2), number(3), number(6), number(7)
		zone, sign := match.GroupByNumber(4).String(), match.GroupByNumber(5).String()
		if zone == "" && sign == "" || hour > 23 || minute > 59 || second > 60 || zoneHour > 23 || zoneMinute > 59 {
			return false
		}
		if second < 60 {
			return true
		}
		offset := zoneHour*60 + zoneMinute
		if sign == "-" {
			offset = -offset
		}
		utc := hour*60 + minute - offset
		return (utc%1440+1440)%1440 == 1439
	}
	return true
}

// Unicode-mode format literals need the same property, dot and anchor lowering
// as schema patterns. Preserve their outer flags (notably ignore-case).
func mustCompileArgumentUnicodeFormat(pattern string, options regexp2.RegexOptions) *regexp2.Regexp {
	syntax := argumentRegexpSyntax{text: pattern, ignoreCase: options&regexp2.IgnoreCase != 0}
	if !syntax.parse() {
		panic("invalid pinned Unicode format expression")
	}
	return regexp2.MustCompile(syntax.lower(), options)
}
