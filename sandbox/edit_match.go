package sandbox

// edit_match.go — matching machinery for edit_file, ported from pi's
// edit-diff.ts. Exact match runs first; when it finds nothing, a fuzzy pass
// retries in a normalized view that tolerates the mangling models introduce
// when quoting code back (smart quotes, unicode dashes, non-breaking spaces,
// trailing whitespace, NFKC-foldable characters). The write-back preserves the
// original bytes of every line the edit does not touch: only lines inside a
// matched span are rewritten from the normalized text.

import (
	"fmt"
	"strings"

	"golang.org/x/text/unicode/norm"
)

// stripBOM removes a leading UTF-8 byte-order mark and reports whether one was
// present, so it can be restored on write.
func stripBOM(s string) (string, bool) {
	if strings.HasPrefix(s, "\uFEFF") {
		return strings.TrimPrefix(s, "\uFEFF"), true
	}
	return s, false
}

// detectCRLF reports whether the file's first line ends with CRLF; the whole
// file is then treated as CRLF and restored that way after editing in LF space.
func detectCRLF(s string) bool {
	if i := strings.Index(s, "\n"); i > 0 {
		return s[i-1] == '\r'
	}
	return false
}

// normalizeToLF folds CRLF and lone CR to LF so matching is line-ending
// agnostic; detectCRLF decides whether to restore CRLF on write.
func normalizeToLF(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	return strings.ReplaceAll(s, "\r", "\n")
}

// normalizeFuzzyRune maps the unicode characters models most often substitute
// for ASCII when re-typing code: smart quotes, dash variants, and exotic
// spaces.
func normalizeFuzzyRune(r rune) rune {
	switch {
	case r >= '‘' && r <= '‛': // single smart quotes
		return '\''
	case r >= '“' && r <= '‟': // double smart quotes
		return '"'
	case (r >= '‐' && r <= '―') || r == '−': // hyphen…horizontal bar, minus sign
		return '-'
	case r == ' ' || (r >= ' ' && r <= ' ') || r == ' ' || r == ' ' || r == '　': // nbsp, en quad…hair space, narrow nbsp, mmsp, ideographic space
		return ' '
	}
	return r
}

// normalizeFuzzyLine is the per-line normalization: NFKC fold, character map,
// then trailing-whitespace strip. It never adds or removes newlines, which is
// the invariant fuzzyReplace's line mapping depends on.
func normalizeFuzzyLine(line string) string {
	line = norm.NFKC.String(line)
	line = strings.Map(normalizeFuzzyRune, line)
	return strings.TrimRight(line, " \t")
}

// normalizeFuzzy normalizes a multi-line string line by line, preserving the
// line structure (same line count as the input).
func normalizeFuzzy(s string) string {
	lines := strings.Split(s, "\n")
	for i := range lines {
		lines[i] = normalizeFuzzyLine(lines[i])
	}
	return strings.Join(lines, "\n")
}

// lineStarts returns the byte offset of each line's first character in s
// (lines delimited by '\n').
func lineStarts(s string) []int {
	starts := []int{0}
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}

// lineOf maps a byte offset to its line index given lineStarts output. The
// '\n' separator is attributed to the line it terminates.
func lineOf(starts []int, off int) int {
	lo, hi := 0, len(starts)-1
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if starts[mid] <= off {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

// fuzzyReplace replaces oldS with newS in content by matching in the
// normalized view. rel is the workspace-relative path, used only in error
// messages. Untouched lines keep their original bytes; lines inside a matched
// span are rewritten from the normalized text with the replacement applied.
// Without replaceAll the match must be unique, mirroring the exact path.
func fuzzyReplace(content, oldS, newS, rel string, replaceAll bool) (string, int, error) {
	normContent := normalizeFuzzy(content)
	normOld := normalizeFuzzy(oldS)
	if normOld == "" {
		return "", 0, fmt.Errorf("edit_file: old_string not found in %s", rel)
	}

	// Non-overlapping match offsets in the normalized view.
	var matches []int
	for from := 0; ; {
		i := strings.Index(normContent[from:], normOld)
		if i < 0 {
			break
		}
		matches = append(matches, from+i)
		from += i + len(normOld)
	}
	if len(matches) == 0 {
		return "", 0, fmt.Errorf("edit_file: old_string not found in %s", rel)
	}
	if len(matches) > 1 && !replaceAll {
		return "", 0, fmt.Errorf("edit_file: old_string matches %d locations in %s after fuzzy normalization; add context to make it unique or set replace_all", len(matches), rel)
	}

	// Normalization is per-line, so both views have identical line structure.
	origStarts := lineStarts(content)
	normStarts := lineStarts(normContent)

	// spanEnd is the exclusive end of line i's span in the given view,
	// including its trailing '\n' when present.
	spanEnd := func(starts []int, view string, i int) int {
		if i+1 < len(starts) {
			return starts[i+1]
		}
		return len(view)
	}

	// Widen each match to whole lines, then merge groups sharing lines.
	type group struct {
		startLine, endLine int
		matches            []int
	}
	var groups []group
	for _, m := range matches {
		s := lineOf(normStarts, m)
		e := lineOf(normStarts, m+len(normOld)-1)
		if n := len(groups); n > 0 && s <= groups[n-1].endLine {
			if e > groups[n-1].endLine {
				groups[n-1].endLine = e
			}
			groups[n-1].matches = append(groups[n-1].matches, m)
		} else {
			groups = append(groups, group{startLine: s, endLine: e, matches: []int{m}})
		}
	}

	var out strings.Builder
	prevLine := 0
	for _, g := range groups {
		// Untouched lines before the group: original bytes.
		out.WriteString(content[origStarts[prevLine]:origStarts[g.startLine]])
		// Group text: normalized bytes with the matched spans replaced.
		ge := spanEnd(normStarts, normContent, g.endLine)
		pos := normStarts[g.startLine]
		for _, m := range g.matches {
			out.WriteString(normContent[pos:m])
			out.WriteString(newS)
			pos = m + len(normOld)
		}
		out.WriteString(normContent[pos:ge])
		prevLine = g.endLine + 1
	}
	if prevLine < len(origStarts) {
		out.WriteString(content[origStarts[prevLine]:])
	}
	return out.String(), len(matches), nil
}
