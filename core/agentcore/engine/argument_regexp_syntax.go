package engine

import (
	"strconv"
	"strings"
	"unicode/utf8"
)

// regexp2's ECMAScript mode still admits legacy escapes and .NET constructs.
// Check the Unicode grammar before handing a pattern to that matcher. The
// matcher remains responsible for backtracking and character range ordering.
type argumentRegexpSyntax struct {
	text          string
	pos           int
	captures      int
	references    []string
	namedRefs     []string
	names         map[string][]map[int]int
	groups        []*argumentRegexpGroup
	nextGroup     int
	namedSlots    map[string][]int
	edits         []argumentRegexpEdit
	atoms         []argumentRegexpAtom
	trackProgress bool
	ignoreCase    bool
}

type argumentRegexpGroup struct {
	id, branch          int
	assertion           bool
	name                string
	multiline, dotAll   bool
	ignoreCase          bool
	start, bodyStart    int
	slot                int
	firstCapture        int
	lastCapture         int
	quantified, reverse bool
	branchConsumes      bool
	alternativesConsume bool
	nullable            bool
	repeatMinimum       string // Nonempty when optional empty iterations need a guard.
}

func (p *argumentRegexpSyntax) parse() bool {
	text := p.text
	p.names, p.namedSlots = map[string][]map[int]int{}, map[string][]int{}
	p.groups = []*argumentRegexpGroup{{alternativesConsume: true, ignoreCase: p.ignoreCase}}
	atom := false
	pendingConsumes := false
	var lastGroup *argumentRegexpGroup
	for p.pos < len(text) {
		quantifiedGroup := lastGroup
		lastGroup = nil
		c := text[p.pos]
		p.pos++
		if !strings.ContainsRune("*+?{", rune(c)) {
			group := p.groups[len(p.groups)-1]
			group.branchConsumes = group.branchConsumes || pendingConsumes
			pendingConsumes = false
		}
		switch c {
		case '\\':
			start := p.pos - 1
			escapeStart := p.pos
			if _, ok := p.escape(false); !ok {
				return false
			}
			atom = text[escapeStart] != 'b' && text[escapeStart] != 'B'
			code := text[escapeStart]
			pendingConsumes = atom && code != 'k' && (code < '1' || code > '9')
			if atom {
				reference, named := "", false
				if code == 'k' {
					reference, named = p.namedRefs[len(p.namedRefs)-1], true
				} else if code >= '1' && code <= '9' {
					reference = p.references[len(p.references)-1]
				}
				p.addAtom(start, p.pos, reference, named)
			}
		case '[':
			start := p.pos - 1
			editStart := len(p.edits)
			if !p.class() {
				return false
			}
			p.lowerWordClass(start, editStart)
			atom = true
			pendingConsumes = true
			p.addAtom(start, p.pos, "", false)
		case '(':
			if !p.group() {
				return false
			}
			atom = false
		case ')':
			if len(p.groups) == 1 {
				return false
			}
			group := p.groups[len(p.groups)-1]
			atom = !group.assertion
			group.lastCapture = p.captures
			group.nullable = !group.branchConsumes || !group.alternativesConsume
			p.edits = append(p.edits, argumentRegexpEdit{start: p.pos - 1, end: p.pos, kind: ')', group: group})
			lastGroup = group
			pendingConsumes = !group.assertion && !group.nullable
			p.groups = p.groups[:len(p.groups)-1]
		case '|':
			group := p.groups[len(p.groups)-1]
			group.branch++
			group.alternativesConsume = group.alternativesConsume && group.branchConsumes
			group.branchConsumes = false
			atom = false
		case '^', '$':
			// JavaScript multiline anchors recognize all four line terminators.
			// Absolute backend anchors also stay absolute inside an m scope.
			group := p.groups[len(p.groups)-1]
			replacement := `\A`
			if c == '$' {
				replacement = `\z`
			}
			if group.multiline {
				if c == '^' {
					replacement = `(?:\A|(?<=[\n\r\u2028\u2029]))`
				} else {
					replacement = `(?:\z|(?=[\n\r\u2028\u2029]))`
				}
			}
			p.edits = append(p.edits, argumentRegexpEdit{start: p.pos - 1, end: p.pos, kind: 'r', replacement: replacement})
			atom = false
		case '.':
			// regexp2's ECMAScript dot ignores its Singleline option, so lower
			// both modes explicitly using the source group's effective flags.
			replacement := `[^\n\r\u2028\u2029]`
			if p.groups[len(p.groups)-1].dotAll {
				replacement = `[\s\S]`
			}
			p.edits = append(p.edits, argumentRegexpEdit{start: p.pos - 1, end: p.pos, kind: 'r', replacement: replacement})
			atom = true
			pendingConsumes = true
			p.addAtom(p.pos-1, p.pos, "", false)
		case '*', '+', '?', '{':
			quantifierStart := p.pos - 1
			if !atom || c == '{' && !p.quantifier() {
				return false
			}
			minimum, optional := argumentRegexpRepetition(text[quantifierStart:p.pos])
			pendingConsumes = pendingConsumes && minimum != "0"
			if p.pos < len(text) && text[p.pos] == '?' {
				p.pos++
			}
			if quantifiedGroup != nil {
				quantifiedGroup.quantified = true
				if optional && quantifiedGroup.nullable && quantifiedGroup.firstCapture <= quantifiedGroup.lastCapture {
					quantifiedGroup.repeatMinimum = minimum
					p.trackProgress = true
					p.edits = append(p.edits, argumentRegexpEdit{start: p.pos, end: p.pos, kind: 'q', group: quantifiedGroup, priority: -3})
				}
			}
			atom = false
		case '}', ']':
			return false
		default:
			atom = true
			pendingConsumes = true
			start := p.pos - 1
			if c >= utf8.RuneSelf {
				_, size := utf8.DecodeRuneInString(text[start:])
				p.pos = start + size
			}
			p.addAtom(start, p.pos, "", false)
		}
	}
	if len(p.groups) != 1 {
		return false
	}
	for _, ref := range p.references {
		n, err := strconv.Atoi(ref)
		if err != nil || n > p.captures {
			return false
		}
	}
	for _, name := range p.namedRefs {
		if len(p.names[name]) == 0 {
			return false
		}
	}
	return true
}

func (p *argumentRegexpSyntax) group() bool {
	start := p.pos - 1
	assertion := false
	captureName := ""
	parent := p.groups[len(p.groups)-1]
	multiline, dotAll := parent.multiline, parent.dotAll
	ignoreCase := parent.ignoreCase
	reverse := parent.reverse
	firstCapture, slot := p.captures+1, 0
	if p.pos == len(p.text) || p.text[p.pos] != '?' {
		p.captures++
		slot = p.captures
	} else {
		p.pos++
		if p.pos == len(p.text) {
			return false
		}
		c := p.text[p.pos]
		p.pos++
		switch c {
		case ':':
		case '=', '!':
			assertion = true
			reverse = false
		case '<':
			if p.pos < len(p.text) && (p.text[p.pos] == '=' || p.text[p.pos] == '!') {
				assertion = true
				reverse = true
				p.pos++
			} else {
				name, ok := p.name()
				if !ok {
					return false
				}
				path := map[int]int{}
				for _, group := range p.groups {
					path[group.id] = group.branch
				}
				for _, previous := range p.names[name] {
					disjoint := false
					for id, branch := range path {
						if old, exists := previous[id]; exists && old != branch {
							disjoint = true
						}
					}
					if !disjoint {
						return false
					}
				}
				p.names[name] = append(p.names[name], path)
				p.captures++
				slot = p.captures
				p.namedSlots[name] = append(p.namedSlots[name], p.captures)
				captureName = name
			}
		default:
			// Scoped modifiers are limited to i/m/s. Standalone modifiers,
			// comments, atomic groups and conditionals are not ECMAScript.
			p.pos--
			seen := map[byte]bool{}
			minus, count := false, 0
			for p.pos < len(p.text) && p.text[p.pos] != ':' {
				flag := p.text[p.pos]
				p.pos++
				if flag == '-' && !minus {
					minus = true
					continue
				}
				if !strings.ContainsRune("ims", rune(flag)) || seen[flag] {
					return false
				}
				seen[flag], count = true, count+1
				switch flag {
				case 'i':
					ignoreCase = !minus
				case 'm':
					multiline = !minus
				case 's':
					dotAll = !minus
				}
			}
			if count == 0 || p.pos == len(p.text) {
				return false
			}
			p.pos++
		}
	}
	p.nextGroup++
	group := &argumentRegexpGroup{id: p.nextGroup, assertion: assertion, name: captureName, multiline: multiline, dotAll: dotAll, ignoreCase: ignoreCase, start: start, bodyStart: p.pos, slot: slot, firstCapture: firstCapture, reverse: reverse, alternativesConsume: true}
	p.groups = append(p.groups, group)
	p.edits = append(p.edits, argumentRegexpEdit{start: start, end: p.pos, kind: '(', group: group})
	return true
}

func (p *argumentRegexpSyntax) name() (string, bool) {
	start := p.pos
	for p.pos < len(p.text) {
		end := p.pos
		r, size := utf8.DecodeRuneInString(p.text[p.pos:])
		if r == '\\' {
			var valid bool
			r, p.pos, valid = argumentRegexpNameEscape(p.text, p.pos)
			if !valid {
				return "", false
			}
		} else {
			p.pos += size
		}
		// JavaScriptCore also treats an escaped greater-than character as the
		// name delimiter. Any literal '>' after it is part of the pattern body.
		if r == '>' {
			return argumentRegexpCaptureName(p.text[start:end])
		}
	}
	return "", false
}

func (p *argumentRegexpSyntax) quantifier() bool {
	start := p.pos
	p.digits()
	if start == p.pos {
		return false
	}
	if p.pos < len(p.text) && p.text[p.pos] == ',' {
		p.pos++
		p.digits()
	}
	if p.pos == len(p.text) || p.text[p.pos] != '}' {
		return false
	}
	p.pos++
	return true
}

func (p *argumentRegexpSyntax) digits() {
	for p.pos < len(p.text) && p.text[p.pos] >= '0' && p.text[p.pos] <= '9' {
		p.pos++
	}
}

// escape reports whether this class atom denotes one character (rather than
// a set), allowing class() to reject shorthand sets used as range endpoints.
func (p *argumentRegexpSyntax) escape(inClass bool) (single, ok bool) {
	if p.pos == len(p.text) {
		return false, false
	}
	c := p.text[p.pos]
	p.pos++
	switch {
	case c >= utf8.RuneSelf:
		// The pinned Pi oracle's RegExp implementation accepts non-ASCII
		// identity escapes even in Unicode mode. Keep that observed behavior.
		_, size := utf8.DecodeRuneInString(p.text[p.pos-1:])
		p.pos += size - 1
		return true, true
	case strings.ContainsRune(`^$\.*+?()[]{}/`, rune(c)):
		return true, true
	case c == '-':
		return true, inClass
	case c == 'b' || c == 'B':
		if inClass {
			return c == 'b', c == 'b'
		}
		replacement := argumentRegexpWordBoundary(p.groups[len(p.groups)-1].ignoreCase, c == 'B')
		p.edits = append(p.edits, argumentRegexpEdit{start: p.pos - 2, end: p.pos, kind: 'r', replacement: replacement})
		return false, true
	case c == 'w' || c == 'W':
		kind := byte('r')
		if inClass {
			kind = 'w'
		}
		replacement := argumentRegexpWordClass(p.groups[len(p.groups)-1].ignoreCase, c == 'W')
		p.edits = append(p.edits, argumentRegexpEdit{start: p.pos - 2, end: p.pos, kind: kind, replacement: replacement})
		return false, true
	case strings.ContainsRune("dDsS", rune(c)):
		return false, true
	case strings.ContainsRune("fnrtv", rune(c)):
		return true, true
	case c == '0':
		return true, p.pos == len(p.text) || p.text[p.pos] < '0' || p.text[p.pos] > '9'
	case c >= '1' && c <= '9':
		start := p.pos - 1
		p.digits()
		p.references = append(p.references, p.text[start:p.pos])
		return false, !inClass
	case c == 'k' && !inClass:
		start := p.pos - 2
		if p.pos == len(p.text) || p.text[p.pos] != '<' {
			return false, false
		}
		p.pos++
		name, valid := p.name()
		p.namedRefs = append(p.namedRefs, name)
		p.edits = append(p.edits, argumentRegexpEdit{start: start, end: p.pos, kind: '\\', name: name})
		return false, valid
	case c == 'c':
		if p.pos == len(p.text) {
			return false, false
		}
		letter := p.text[p.pos]
		p.pos++
		return true, letter >= 'a' && letter <= 'z' || letter >= 'A' && letter <= 'Z'
	case c == 'x' || c == 'u':
		size := 2
		if c == 'u' {
			size = 4
			if p.pos < len(p.text) && p.text[p.pos] == '{' {
				p.pos++
				start := p.pos
				for p.pos < len(p.text) && argumentURLHex(p.text[p.pos]) {
					p.pos++
				}
				code, err := strconv.ParseUint(p.text[start:p.pos], 16, 32)
				if err != nil || code > 0x10ffff || p.pos == len(p.text) || p.text[p.pos] != '}' {
					return false, false
				}
				p.pos++
				return true, true
			}
		}
		for range size {
			if p.pos == len(p.text) || !argumentURLHex(p.text[p.pos]) {
				return false, false
			}
			p.pos++
		}
		return true, true
	case c == 'p' || c == 'P':
		start := p.pos - 2
		if p.pos == len(p.text) || p.text[p.pos] != '{' {
			return false, false
		}
		end := strings.IndexByte(p.text[p.pos:], '}')
		if end < 0 {
			return false, false
		}
		property := p.text[p.pos+1 : p.pos+end]
		canonical, ok := argumentUnicodePropertyAliases[property]
		if !ok {
			return false, false
		}
		p.pos += end + 1
		p.edits = append(p.edits, argumentRegexpEdit{start: start, end: p.pos, kind: 'r', replacement: argumentRegexpPropertyClass(canonical, c == 'P', inClass)})
		return false, true
	}
	return false, false
}

func (p *argumentRegexpSyntax) class() bool {
	if p.pos < len(p.text) && p.text[p.pos] == '^' {
		p.pos++
	}
	for p.pos < len(p.text) && p.text[p.pos] != ']' {
		single, ok := p.classAtom()
		if !ok {
			return false
		}
		if p.pos+1 < len(p.text) && p.text[p.pos] == '-' && p.text[p.pos+1] != ']' {
			p.pos++
			endSingle, endOK := p.classAtom()
			if !single || !endSingle || !endOK {
				return false
			}
		}
	}
	if p.pos == len(p.text) {
		return false
	}
	p.pos++
	return true
}

func (p *argumentRegexpSyntax) classAtom() (bool, bool) {
	c := p.text[p.pos]
	_, size := utf8.DecodeRuneInString(p.text[p.pos:])
	p.pos += size
	if c == '\\' {
		return p.escape(true)
	}
	return true, true
}
