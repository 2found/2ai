package ai

import (
	"strings"
	"unicode/utf16"

	"github.com/2found/2ai/jsonjs"
)

// SanitizeSurrogates ports Pi's provider-text filter. Transcripts retain lone
// UTF-16 units as WTF-8; only the provider fields selected by Pi drop them.
// Valid pairs, including pairs assembled by concatenating WTF-8 strings, survive.
func SanitizeSurrogates(text string) string {
	points := jsonjs.StringCodePoints(text)
	var out strings.Builder
	for i := 0; i < len(points); i++ {
		point := points[i]
		if point >= 0xd800 && point <= 0xdbff {
			if i+1 < len(points) && points[i+1] >= 0xdc00 && points[i+1] <= 0xdfff {
				out.WriteRune(utf16.DecodeRune(point, points[i+1]))
				i++
			}
		} else if point < 0xdc00 || point > 0xdfff {
			out.WriteRune(point)
		}
	}
	return out.String()
}
