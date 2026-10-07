package engine

import (
	"slices"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

type argumentRegexpEdit struct {
	start, end  int
	kind        byte // Capture parenthesis, named reference, or literal replacement.
	name        string
	replacement string
	group       *argumentRegexpGroup
	priority    int
}

// JavaScript numbers all captures in opening-parenthesis order. regexp2 puts
// named captures after unnamed ones. Lower names to plain captures and their
// references to the corresponding numbers, so both reference forms agree.
// Property, dot and anchor replacements share the same source-position pass.
func (p *argumentRegexpSyntax) lower() string {
	if p.trackProgress {
		p.lowerProgressAtoms()
	}
	if len(p.edits) == 0 {
		return p.text
	}
	var out strings.Builder
	last := 0
	for _, edit := range p.edits {
		out.WriteString(p.text[last:edit.start])
		last = edit.end
		if edit.kind == 'r' {
			out.WriteString(edit.replacement)
			continue
		}
		if edit.group != nil {
			if edit.kind == 'q' {
				if edit.group.reverse {
					out.WriteString(edit.group.repetitionInit())
				}
				out.WriteByte(')')
				continue
			}
			opening, closing := p.lowerGroup(edit.group)
			if edit.kind == '(' {
				out.WriteString(opening)
			} else {
				out.WriteString(closing)
			}
			continue
		}
		slots := p.namedSlots[edit.name]
		duplicate := len(slots) > 1
		// An auxiliary named slot records the last participating duplicate.
		// regexp2 allocates these after all the plain (JavaScript) captures,
		// so the wrapper cannot shift any source numeric reference.
		backendName := "piCapture" + strconv.Itoa(slots[0])
		if duplicate {
			out.WriteString(`\k<` + backendName + ">")
		} else {
			// Separate the slot number from literal digits after the name.
			out.WriteString(`(?:\` + strconv.Itoa(slots[0]) + ")")
		}
	}
	out.WriteString(p.text[last:])
	if p.trackProgress {
		// These declarations never execute. They make always-empty source
		// captures' flags available to conditions even when no atom sets them.
		out.WriteString("(?:(?!)")
		for _, flag := range p.progressFlags() {
			out.WriteString("(?<" + flag + ">)")
		}
		out.WriteString("|)")
	}
	return out.String()
}

func (p *argumentRegexpSyntax) lowerGroup(group *argumentRegexpGroup) (string, string) {
	opening, closing := p.text[group.start:group.bodyStart], ")"
	if group.slot > 0 {
		opening = "("
		if slots := p.namedSlots[group.name]; len(slots) > 1 {
			opening = "(?<piCapture" + strconv.Itoa(slots[0]) + ">("
			closing += ")"
		}
	}
	first, last := group.slot, group.slot
	if group.quantified {
		first, last = group.firstCapture, group.lastCapture
	}
	if first == 0 || first > last {
		return opening, closing
	}
	// Keep each capture stack at one value, and reset every descendant at the
	// start of an iteration. Balancing operations are undone by regexp2 during
	// backtracking, restoring the captures of the previous successful path.
	var reset strings.Builder
	clear := func(name string) {
		reset.WriteString("(?(" + name + ")(?<-" + name + ">)|)")
	}
	for slot := first; slot <= last; slot++ {
		clear(strconv.Itoa(slot))
		if p.trackProgress {
			clear("piNonEmpty" + strconv.Itoa(slot))
		}
	}
	var auxiliary []string
	for _, slots := range p.namedSlots {
		if len(slots) < 2 {
			continue
		}
		for _, slot := range slots {
			if slot >= first && slot <= last {
				auxiliary = append(auxiliary, "piCapture"+strconv.Itoa(slots[0]))
				break
			}
		}
	}
	slices.Sort(auxiliary)
	for _, name := range auxiliary {
		clear(name)
		if p.trackProgress {
			clear("piNonEmpty" + strings.TrimPrefix(name, "pi"))
		}
	}
	entry, exit := reset.String(), ""
	if group.repeatMinimum != "" {
		before, after := group.repetitionGuard()
		entry += before
		exit = after
		if group.reverse {
			opening = "(?:" + opening
		} else {
			opening = "(?:" + group.repetitionInit() + opening
		}
	}
	// Wrap alternatives so the reset applies to every branch. Lookbehind
	// executes right-to-left, so its entry operations belong at the other end.
	if group.reverse {
		return opening + exit + "(?:", ")" + entry + closing
	}
	return opening + entry + "(?:", ")" + exit + closing
}

// Group names are ECMAScript IdentifierNames, including Unicode escapes.
// Normalize escapes before checking duplicate declarations or named references;
// the matching engine never needs to parse an identifier in another grammar.
func argumentRegexpCaptureName(raw string) (string, bool) {
	var chars []rune
	for pos := 0; pos < len(raw); {
		if raw[pos] != '\\' {
			r, size := utf8.DecodeRuneInString(raw[pos:])
			chars = append(chars, r)
			pos += size
			continue
		}
		code, end, valid := argumentRegexpNameEscape(raw, pos)
		if !valid {
			return "", false
		}
		chars = append(chars, code)
		pos = end
	}
	var out strings.Builder
	for i := 0; i < len(chars); i++ {
		r := chars[i]
		if r >= 0xd800 && r <= 0xdbff && i+1 < len(chars) && chars[i+1] >= 0xdc00 && chars[i+1] <= 0xdfff {
			r = utf16.DecodeRune(r, chars[i+1])
			i++
		}
		// The pinned JavaScriptCore grammar uses the letter/mark/number
		// categories below. It rejects Letter_Number and the Other_ID_Start /
		// Other_ID_Continue exceptions admitted by Unicode ID_Start/Continue.
		start := r == '$' || r == '_' || argumentUnicodeHas("General_Category=Letter", r)
		continuation := argumentUnicodeHas("General_Category=Nonspacing_Mark", r) || argumentUnicodeHas("General_Category=Spacing_Mark", r) || argumentUnicodeHas("General_Category=Decimal_Number", r) || argumentUnicodeHas("General_Category=Connector_Punctuation", r) || r == 0x200c || r == 0x200d
		if !start && (out.Len() == 0 || !continuation) {
			return "", false
		}
		out.WriteRune(r)
	}
	return out.String(), out.Len() > 0
}

// Decode one Unicode escape; fixed-width surrogate units are joined by the caller.
func argumentRegexpNameEscape(raw string, pos int) (rune, int, bool) {
	if pos+2 > len(raw) || raw[pos+1] != 'u' {
		return 0, pos, false
	}
	pos += 2
	start, end := pos, pos+4
	braced := pos < len(raw) && raw[pos] == '{'
	if braced {
		start = pos + 1
		length := strings.IndexByte(raw[start:], '}')
		if length < 0 {
			return 0, pos, false
		}
		end = start + length
		pos = end + 1
	} else {
		pos = end
	}
	if end > len(raw) || start == end {
		return 0, pos, false
	}
	for _, c := range raw[start:end] {
		if c > 127 || !argumentURLHex(byte(c)) {
			return 0, pos, false
		}
	}
	code, err := strconv.ParseUint(raw[start:end], 16, 32)
	if err != nil || code > utf8.MaxRune {
		return 0, pos, false
	}
	// Only consecutive fixed-width escapes form a surrogate pair.
	// Braced escapes denote individual code points, which cannot be
	// surrogate code points in an IdentifierName.
	if braced && code >= 0xd800 && code <= 0xdfff {
		return 0, pos, false
	}
	return rune(code), pos, true
}
