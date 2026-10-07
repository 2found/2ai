package engine

import (
	"strings"

	"golang.org/x/text/unicode/norm"
)

// These checks follow TypeBox 1.3.27's IDNA rules, including its permitted
// punctuation and partial contextual/bidi checks. A stricter DNS/IDNA library
// would reject inputs Pi accepts. See LICENSE.typebox.
func argumentHostname(value string, international bool) bool {
	if value == "" || international && strings.Contains(value, " ") {
		return false
	}
	if international {
		value = strings.Map(func(r rune) rune {
			if r >= 0xff01 && r <= 0xff5e {
				return r - 0xfee0
			}
			return r
		}, value)
		value = norm.NFC.String(value)
		value = strings.Map(func(r rune) rune {
			switch {
			case r == 0x00ad || r == 0x034f || r >= 0x180b && r <= 0x180d || r == 0x200b || r >= 0xfe00 && r <= 0xfe0f || r >= 0xe0100 && r <= 0xe01ef:
				return -1
			case r == 0x3002 || r == 0xff0e || r == 0xff61:
				return '.'
			}
			return r
		}, value)
	}
	if hostnameUTF16Length(value) > 253 {
		return false
	}
	labels := strings.Split(value, ".")
	hasBidi := false
	if international {
		for _, label := range labels {
			decoded := label
			if strings.HasPrefix(strings.ToLower(label), "xn--") {
				var ok bool
				decoded, ok = hostnamePunyDecode(strings.ToLower(label[4:]))
				if !ok {
					continue
				}
			}
			hasBidi = hasBidi || hostnameHasRTL(decoded)
		}
	}
	for _, label := range labels {
		length := hostnameUTF16Length(label)
		if length == 0 || length > 63 {
			return false
		}
		valid := hostnamePunyLabel(label)
		if international {
			valid = valid || hostnameUnicodeLabel(label)
		} else {
			valid = valid || hostnameASCIILabel(label)
		}
		if !valid || hasBidi && !hostnameBidiRule(label) {
			return false
		}
	}
	return true
}

func hostnameUTF16Length(value string) int {
	length := 0
	for _, r := range value {
		length++
		if r > 0xffff {
			length++
		}
	}
	return length
}

func hostnameASCIILabel(value string) bool {
	if value == "" || strings.HasPrefix(value, "-") || strings.HasSuffix(value, "-") || len(value) >= 4 && value[2:4] == "--" {
		return false
	}
	for _, r := range value {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-') {
			return false
		}
	}
	return true
}

func hostnamePunyLabel(value string) bool {
	if !strings.HasPrefix(strings.ToLower(value), "xn--") {
		return false
	}
	body := strings.ToLower(value[4:])
	if strings.LastIndexByte(body, '-') == 0 {
		return false
	}
	decoded, ok := hostnamePunyDecode(body)
	return ok && hostnameNonASCII(decoded) && hostnameUnicodeLabel(decoded)
}

func hostnameNonASCII(value string) bool {
	for _, r := range value {
		if r > 127 {
			return true
		}
	}
	return false
}

func hostnameUnicodeLabel(value string) bool {
	if hostnameNonASCII(value) && len(hostnamePunyEncode(value))+4 > 63 {
		return false
	}
	if hostnameHasRTL(value) && !hostnameBidiRule(value) {
		return false
	}
	chars := []rune(value)
	if len(chars) == 0 || chars[0] == '-' || chars[len(chars)-1] == '-' || len(chars) >= 4 && chars[2] == '-' && chars[3] == '-' || argumentUnicodeHas("General_Category=Mark", chars[0]) {
		return false
	}
	hasJapanese := false
	for i, r := range chars {
		if strings.ContainsRune("\u0640\u07fa\u302e\u302f\u3031\u3032\u3033\u3034\u3035\u303b", r) {
			return false
		}
		if !(argumentUnicodeHas("General_Category=Letter", r) || argumentUnicodeHas("General_Category=Decimal_Number", r) || argumentUnicodeHas("General_Category=Nonspacing_Mark", r) || argumentUnicodeHas("General_Category=Spacing_Mark", r) || strings.ContainsRune("-+.,:/\u00b7\u0375\u05f3\u05f4\u200c\u200d\u30fb\u00df\u03c2\u06fd\u06fe\u0f0b\u3007", r)) {
			return false
		}
		hasJapanese = hasJapanese || argumentUnicodeHas("Script=Hiragana", r) || argumentUnicodeHas("Script=Katakana", r) || argumentUnicodeHas("Script=Han", r)
		var prev, next rune
		if i > 0 {
			prev = chars[i-1]
		}
		if i+1 < len(chars) {
			next = chars[i+1]
		}
		switch r {
		case 0x00b7:
			if prev != 'l' || next != 'l' {
				return false
			}
		case 0x0375:
			if next == 0 || !argumentUnicodeHas("Script=Greek", next) {
				return false
			}
		case 0x05f3, 0x05f4:
			if prev == 0 || !argumentUnicodeHas("Script=Hebrew", prev) {
				return false
			}
		case 0x200c:
			if prev == 0 || prev < 128 && !hostnameVirama(prev) {
				return false
			}
		case 0x200d:
			if prev == 0 || !hostnameVirama(prev) {
				return false
			}
		}
	}
	return !strings.ContainsRune(value, 0x30fb) || hasJapanese
}

func hostnameVirama(r rune) bool {
	return strings.ContainsRune("\u094d\u09cd\u0a4d\u0acd\u0b4d\u0bcd\u0c4d\u0ccd\u0d3b\u0d3c\u0d4d\u0dca\u1b44\u1baa\u1bab\ua9c0\U00011046\U0001107f\U000110b9\U00011133\U00011134\U000111c0\U00011235\U0001134d\U00011442\U000114c2\U000115bf\U0001163f\U000116b6\U00011c3f\U00011d44\U00011d45", r)
}

func hostnameBidiClass(r rune) string {
	switch {
	case r >= '0' && r <= '9' || r >= 0x06f0 && r <= 0x06f9:
		return "EN"
	case r >= 0x0660 && r <= 0x0669:
		return "AN"
	case argumentUnicodeHas("General_Category=Nonspacing_Mark", r):
		return "NSM"
	case argumentUnicodeHas("Script=Hebrew", r):
		return "R"
	case argumentUnicodeHas("Script=Arabic", r) || argumentUnicodeHas("Script=Syriac", r) || argumentUnicodeHas("Script=Thaana", r) || argumentUnicodeHas("Script=Mandaic", r):
		return "AL"
	case argumentUnicodeHas("General_Category=Letter", r):
		return "L"
	default:
		return "ON"
	}
}

func hostnameHasRTL(value string) bool {
	for _, r := range value {
		switch hostnameBidiClass(r) {
		case "R", "AL", "AN":
			return true
		}
	}
	return false
}

func hostnameBidiRule(value string) bool {
	rtl, first, sawEN, sawAN := false, true, false, false
	for _, r := range value {
		class := hostnameBidiClass(r)
		if first {
			if class != "L" && class != "R" && class != "AL" {
				return false
			}
			rtl, first = class != "L", false
		}
		if rtl && class == "L" || !rtl && (class == "R" || class == "AL" || class == "AN") {
			return false
		}
		sawEN, sawAN = sawEN || class == "EN", sawAN || class == "AN"
	}
	return !rtl || !sawEN || !sawAN
}
