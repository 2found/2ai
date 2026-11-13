package engine

import (
	"slices"
	"strconv"
	"strings"
)

type argumentRegexpAtom struct {
	start, end int
	groups     []*argumentRegexpGroup
	reference  string
	named      bool
	reverse    bool
}

func (p *argumentRegexpSyntax) addAtom(start, end int, reference string, named bool) {
	first := 1 // The root is neither a capture nor a repeated group.
	for i, group := range p.groups {
		if group.assertion {
			// Lookarounds consume only within their own scope. Their inner
			// captures can be nonempty without advancing an outer repetition.
			first = i + 1
		}
	}
	if first == len(p.groups) {
		return
	}
	p.atoms = append(p.atoms, argumentRegexpAtom{start: start, end: end,
		groups: slices.Clone(p.groups[first:]), reference: reference, named: named,
		reverse: p.groups[len(p.groups)-1].reverse})
}

func (p *argumentRegexpSyntax) lowerProgressAtoms() {
	for _, atom := range p.atoms {
		var flags []string
		for _, group := range atom.groups {
			if group.slot > 0 {
				flags = append(flags, "piNonEmpty"+strconv.Itoa(group.slot))
				if slots := p.namedSlots[group.name]; len(slots) > 1 {
					flags = append(flags, "piNonEmptyCapture"+strconv.Itoa(slots[0]))
				}
			}
			if group.repeatMinimum != "" {
				flags = append(flags, "piProgress"+strconv.Itoa(group.id))
			}
		}
		if len(flags) == 0 {
			continue
		}
		var marks strings.Builder
		for _, flag := range flags {
			clear := "(?(" + flag + ")(?<-" + flag + ">)|)"
			set := "(?<" + flag + ">)"
			if atom.reverse {
				marks.WriteString(set + clear)
			} else {
				marks.WriteString(clear + set)
			}
		}
		mark := marks.String()
		if atom.reference != "" {
			slot := atom.reference
			if atom.named {
				slots := p.namedSlots[atom.reference]
				slot = strconv.Itoa(slots[0])
				if len(slots) > 1 {
					slot = "piCapture" + slot
				}
			}
			flag := "piNonEmpty" + strings.TrimPrefix(slot, "pi")
			// Forward/self references and captures with an empty value do not
			// advance the cursor. A flag alone is insufficient while its source
			// capture is still open; check the committed capture as well.
			mark = "(?(" + slot + ")(?(" + flag + ")" + mark + "|)|)"
		}
		p.edits = append(p.edits,
			argumentRegexpEdit{start: atom.start, end: atom.start, kind: 'r', priority: -1, replacement: "(?:"},
			argumentRegexpEdit{start: atom.end, end: atom.end, kind: 'r', priority: -2, replacement: mark + ")"})
	}
	// At a shared position, close the preceding atom/quantifier wrapper before
	// opening the next atom, and open an atom before replacing its source text.
	slices.SortStableFunc(p.edits, func(a, b argumentRegexpEdit) int {
		if a.start != b.start {
			return a.start - b.start
		}
		return a.priority - b.priority
	})
}

func (p *argumentRegexpSyntax) progressFlags() []string {
	var flags []string
	for slot := 1; slot <= p.captures; slot++ {
		flags = append(flags, "piNonEmpty"+strconv.Itoa(slot))
	}
	for _, slots := range p.namedSlots {
		if len(slots) > 1 {
			flags = append(flags, "piNonEmptyCapture"+strconv.Itoa(slots[0]))
		}
	}
	for _, edit := range p.edits {
		if edit.kind == '(' && edit.group.repeatMinimum != "" {
			flags = append(flags, "piProgress"+strconv.Itoa(edit.group.id))
		}
	}
	slices.Sort(flags)
	return flags
}
