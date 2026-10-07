package engine

import (
	"strconv"
	"strings"
)

func argumentRegexpRepetition(quantifier string) (minimum string, optional bool) {
	switch quantifier {
	case "*", "?":
		return "0", true
	case "+":
		return "1", true
	}
	bounds := strings.Split(quantifier[1:len(quantifier)-1], ",")
	normalize := func(value string) string {
		value = strings.TrimLeft(value, "0")
		if value == "" {
			return "0"
		}
		return value
	}
	minimum = normalize(bounds[0])
	return minimum, len(bounds) == 2 && (bounds[1] == "" || normalize(bounds[1]) != minimum)
}

func (group *argumentRegexpGroup) repetitionInit() string {
	if group.repeatMinimum == "" || group.repeatMinimum == "0" {
		return ""
	}
	return "(?<piMinimum" + strconv.Itoa(group.id) + ">){" + group.repeatMinimum + "}"
}

// JS RepeatMatcher rejects an empty iteration after satisfying the minimum.
// Consuming atoms set a private progress capture, which is undone when the
// matcher backtracks. This avoids rescanning the input suffix per iteration.
// Only nullable groups containing captures need this guard: nonempty bodies
// always advance, and empty iterations without captures cannot affect matching.
func (group *argumentRegexpGroup) repetitionGuard() (entry, exit string) {
	id := strconv.Itoa(group.id)
	progress := "piProgress" + id
	clear := func(name string) string { return "(?(" + name + ")(?<-" + name + ">)|)" }
	entry = clear(progress)
	exit = "(?(" + progress + ")|(?!))"
	if group.repeatMinimum != "0" {
		quota, required := "piMinimum"+id, "piRequired"+id
		mark := "(?(" + quota + ")(?<-" + quota + ">)(?<" + required + ">)|)"
		if group.reverse {
			entry = mark + clear(required) + entry
		} else {
			entry += clear(required) + mark
		}
		exit = "(?(" + required + ")|" + exit + ")"
	}
	return entry, exit
}
