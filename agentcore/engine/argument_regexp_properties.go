package engine

import (
	"sort"
	"strconv"
	"strings"
)

// Expand the pinned property set explicitly rather than using regexp2's
// different property vocabulary or the Go toolchain's Unicode version.
func argumentRegexpPropertyClass(property string, negate, inClass bool) string {
	ranges := argumentUnicodePropertyRanges[property]
	var out strings.Builder
	if !inClass {
		out.WriteByte('[')
	}
	interval := func(start, end rune) {
		out.WriteString(`\u{` + strconv.FormatInt(int64(start), 16) + "}")
		if start != end {
			out.WriteString(`-\u{` + strconv.FormatInt(int64(end), 16) + "}")
		}
	}
	if negate {
		next := rune(0)
		for i := 0; i < len(ranges); i += 2 {
			if next < ranges[i] {
				interval(next, ranges[i]-1)
			}
			next = ranges[i+1] + 1
		}
		if next <= 0x10ffff {
			interval(next, 0x10ffff)
		}
	} else {
		for i := 0; i < len(ranges); i += 2 {
			interval(ranges[i], ranges[i+1])
		}
	}
	if !inClass {
		out.WriteByte(']')
	}
	return out.String()
}

// Use the same pinned scalar sets for non-regex validation. A toolchain upgrade
// must not silently change hostname admission or capture-name parsing.
func argumentUnicodeHas(property string, r rune) bool {
	ranges := argumentUnicodePropertyRanges[property]
	i := sort.Search(len(ranges)/2, func(i int) bool { return ranges[2*i+1] >= r })
	return i < len(ranges)/2 && ranges[2*i] <= r
}
