package engine

import (
	"regexp"
	"strings"

	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

// These guards follow TypeBox 1.3.27's URL/IRI formats (LICENSE.typebox).
// URL.canParse permits nonfatal WHATWG validation errors; net/url and a
// parser configured to reject those errors would change tool admission.
var (
	argumentIPvFuture       = regexp.MustCompile(`\[[vV][0-9a-fA-F]+\.[^\]]+\]`)
	argumentMalformedScheme = regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9+\-.]*//`)
)

func argumentURL(format, value string) bool {
	if format != "url" {
		for i, r := range value {
			if r <= 0x20 || r == '\\' {
				return false
			}
			if format == "iri" && strings.ContainsRune("<>^`{|}", r) || format == "iri-reference" && r == 0x7f {
				return false
			}
			if r == '%' && (i+2 >= len(value) || !argumentURLHex(value[i+1]) || !argumentURLHex(value[i+2])) {
				return false
			}
		}
	}
	if format == "iri-reference" {
		if argumentMalformedScheme.MatchString(value) {
			return false
		}
		_, err := whatwg.ParseRef("http://example.com", value)
		return err == nil
	}
	if format == "iri" && hostnameUTF16Length(value) < 2048 {
		// JavaScript replace with a non-global expression changes only the first
		// match, including when that match occurs outside the URL authority.
		if match := argumentIPvFuture.FindStringIndex(value); match != nil {
			value = value[:match[0]] + "[::1]" + value[match[1]:]
		}
	}
	_, err := whatwg.Parse(value)
	return err == nil
}

func argumentURLHex(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
}
