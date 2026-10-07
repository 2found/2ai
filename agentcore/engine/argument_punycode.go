package engine

import (
	"math"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// Raw RFC 3492 operations used by TypeBox's hostname format, without the
// mappings or extra registration checks an IDNA profile would introduce.
// Ported from TypeBox 1.3.27; see LICENSE.typebox.
func hostnamePunyAdapt(delta, count int64, first bool) int64 {
	if first {
		delta /= 700
	} else {
		delta /= 2
	}
	delta += delta / count
	k := int64(0)
	for delta > 455 {
		delta /= 35
		k += 36
	}
	return k + 36*delta/(delta+38)
}

func hostnamePunyThreshold(k, bias int64) int64 {
	if k <= bias {
		return 1
	}
	if k >= bias+26 {
		return 26
	}
	return k - bias
}

func hostnamePunyDecode(value string) (string, bool) {
	output := []rune{}
	hasSurrogates := false
	n, index, bias := int64(128), int64(0), int64(72)
	delim := strings.LastIndexByte(value, '-')
	if delim > 0 {
		for _, r := range value[:delim] {
			if r >= 128 {
				return "", false
			}
			output = append(output, r)
		}
	}
	for input := delim + 1; input < len(value); {
		old, weight := index, int64(1)
		for k := int64(36); ; k += 36 {
			if input >= len(value) {
				return "", false
			}
			ch := value[input]
			input++
			var digit int64
			switch {
			case ch >= 'a' && ch <= 'z':
				digit = int64(ch - 'a')
			case ch >= '0' && ch <= '9':
				digit = int64(ch-'0') + 26
			default:
				return "", false
			}
			if digit > (math.MaxInt64-index)/weight {
				return "", false
			}
			index += digit * weight
			threshold := hostnamePunyThreshold(k, bias)
			if digit < threshold {
				break
			}
			if weight > math.MaxInt64/(36-threshold) {
				return "", false
			}
			weight *= 36 - threshold
		}
		count := int64(len(output) + 1)
		if index/count > utf8.MaxRune-n {
			return "", false
		}
		bias = hostnamePunyAdapt(index-old, count, old == 0)
		n += index / count
		if n >= 0xd800 && n <= 0xdfff {
			hasSurrogates = true
		}
		index %= count
		output = append(output, 0)
		copy(output[index+1:], output[index:])
		output[index] = rune(n)
		index++
	}
	if !hasSurrogates {
		return string(output), true
	}
	// String.fromCodePoint accepts individual surrogate values. Adjacent
	// pairs combine when Pi later iterates the string by Unicode code point.
	// Lone surrogates become U+FFFD here, which the label category check also
	// rejects. Do not reject a valid pair before that check.
	units := make([]uint16, 0, len(output))
	for _, r := range output {
		if r <= 0xffff {
			units = append(units, uint16(r))
		} else {
			high, low := utf16.EncodeRune(r)
			units = append(units, uint16(high), uint16(low))
		}
	}
	return string(utf16.Decode(units)), true
}

func hostnamePunyEncode(value string) string {
	chars := []rune(value)
	output := []byte{}
	for _, r := range chars {
		if r < 128 {
			output = append(output, byte(r))
		}
	}
	basic := int64(len(output))
	if basic > 0 {
		output = append(output, '-')
	}
	n, delta, bias, handled := int64(128), int64(0), int64(72), basic
	digitChar := func(digit int64) byte {
		if digit < 26 {
			return byte(digit) + 'a'
		}
		return byte(digit-26) + '0'
	}
	for handled < int64(len(chars)) {
		m := int64(utf8.MaxRune)
		for _, r := range chars {
			if int64(r) >= n && int64(r) < m {
				m = int64(r)
			}
		}
		delta += (m - n) * (handled + 1)
		n = m
		for _, r := range chars {
			if int64(r) < n {
				delta++
			}
			if int64(r) != n {
				continue
			}
			q := delta
			for k := int64(36); ; k += 36 {
				threshold := hostnamePunyThreshold(k, bias)
				if q < threshold {
					break
				}
				output = append(output, digitChar(threshold+(q-threshold)%(36-threshold)))
				q = (q - threshold) / (36 - threshold)
			}
			output = append(output, digitChar(q))
			bias = hostnamePunyAdapt(delta, handled+1, handled == basic)
			delta = 0
			handled++
		}
		delta++
		n++
	}
	return string(output)
}
