package engine

import "strings"

// In Unicode mode, ignore-case adds the two scalars whose simple fold enters
// the ASCII word set. Disable backend case folding: it otherwise admits U+0130
// and omits U+017F. This set also defines both sides of a word boundary.
func argumentRegexpWordClass(ignoreCase, negate bool) string {
	body := `A-Za-z0-9_`
	if ignoreCase {
		body += `\u017f\u212a`
	}
	if negate {
		body = "^" + body
	}
	return "(?-i:[" + body + "])"
}

func argumentRegexpWordBoundary(ignoreCase, negate bool) string {
	word := argumentRegexpWordClass(ignoreCase, false)
	if negate {
		return "(?:(?<=" + word + ")(?=" + word + ")|(?<!" + word + ")(?!" + word + "))"
	}
	return "(?:(?<=" + word + ")(?!" + word + ")|(?<!" + word + ")(?=" + word + "))"
}

// A word escape inside a class must keep its own folding policy without
// changing the other class members. Split that union into matching branches;
// for a negated class, reject the union before consuming one scalar. In RTL
// matching the scalar is consumed first and the lookahead checks that scalar.
func (p *argumentRegexpSyntax) lowerWordClass(start, editStart int) {
	edits := p.edits[editStart:]
	found := false
	for _, edit := range edits {
		found = found || edit.kind == 'w'
	}
	if !found {
		return
	}
	last := start + 1
	negate := p.text[last] == '^'
	if negate {
		last++
	}
	var body strings.Builder
	var branches []string
	for _, edit := range edits {
		body.WriteString(p.text[last:edit.start])
		if edit.kind == 'w' {
			branches = append(branches, edit.replacement)
		} else {
			body.WriteString(edit.replacement)
		}
		last = edit.end
	}
	body.WriteString(p.text[last : p.pos-1])
	if body.Len() > 0 {
		text := body.String()
		// A literal caret can become the first member after removing a word escape.
		if text[0] == '^' {
			text = `\` + text
		}
		branches = append(branches, "["+text+"]")
	}
	union := "(?:" + strings.Join(branches, "|") + ")"
	replacement := union
	if negate {
		replacement = "(?!" + union + `)[\s\S]`
	}
	p.edits = append(p.edits[:editStart], argumentRegexpEdit{start: start, end: p.pos, kind: 'r', replacement: replacement})
}
