package ai

import (
	"math"
	"strings"
	"testing"
)

func TestParseJSNumberPreservesNumberSemanticsBeforeCallerPolicy(t *testing.T) {
	for _, tc := range []struct {
		input string
		want  float64
	}{
		{"", 0}, {" \ufeff\n", 0}, {"-0", math.Copysign(0, -1)},
		{"-1e-9999", math.Copysign(0, -1)}, {"1e-9999", 0},
		{"Infinity", math.Inf(1)}, {"+Infinity", math.Inf(1)}, {"-Infinity", math.Inf(-1)},
		{"1e9999", math.Inf(1)}, {"0x1" + strings.Repeat("0", 256), math.Inf(1)},
		{"0x20000000000001", 9007199254740992}, {"0x20000000000003", 9007199254740996},
		{"0b1" + strings.Repeat("0", 64), math.Exp2(64)},
		{"\u00851", math.NaN()}, {"Inf", math.NaN()}, {"-0x1p2", math.NaN()},
		{"1_0", math.NaN()}, {"0x-1", math.NaN()}, {"0x+1", math.NaN()}, {"0b1_0", math.NaN()},
	} {
		got := ParseJSNumber(tc.input)
		if math.IsNaN(tc.want) {
			if !math.IsNaN(got) {
				t.Errorf("%q: got %v, want NaN", tc.input, got)
			}
			continue
		}
		if math.Float64bits(got) != math.Float64bits(tc.want) {
			t.Errorf("%q: got %v (%x), want %v (%x)", tc.input, got, math.Float64bits(got), tc.want, math.Float64bits(tc.want))
		}
	}
}
