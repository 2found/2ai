package engine

import (
	"encoding/json"
	"os"
	"testing"
)

func TestPiRegexpCharactersOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-regexp-characters.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, Bun, Unicode string
		WordSets                     []struct {
			IgnoreCase bool
			Codepoints []rune
		}
		Inputs []string
		Cases  []struct {
			Pattern  string
			Valid    bool
			Accepted []int
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || fixture.Bun != "1.3.14" || fixture.Unicode != "15.1" || len(fixture.Cases) != 71 || len(fixture.Inputs) != 255 || len(fixture.WordSets) != 2 {
		t.Fatal("unexpected character oracle")
	}
	for _, row := range fixture.WordSets {
		pattern := `^\w$`
		if row.IgnoreCase {
			pattern = `^(?i:\w)$`
		}
		t.Run("complete word set/"+pattern, func(t *testing.T) {
			compiled, err := compileArgumentRegexp(pattern)
			if err != nil {
				t.Fatal(err)
			}
			accepted := map[rune]bool{}
			for _, code := range row.Codepoints {
				accepted[code] = true
			}
			for code := rune(0); code <= 0x10ffff; code++ {
				// Go strings cannot represent lone UTF-16 surrogate units.
				if code >= 0xd800 && code <= 0xdfff {
					continue
				}
				if got := compiled.MatchString(string(code)); got != accepted[code] {
					t.Fatalf("%U: Go=%v Pi=%v", code, got, accepted[code])
				}
			}
		})
	}
	for _, row := range fixture.Cases {
		t.Run(row.Pattern, func(t *testing.T) {
			compiled, err := compileArgumentRegexp(row.Pattern)
			if (err == nil) != row.Valid {
				t.Fatalf("admission Go=%v Pi=%v: %v", err == nil, row.Valid, err)
			}
			if !row.Valid {
				return
			}
			accepted := map[int]bool{}
			for _, i := range row.Accepted {
				accepted[i] = true
			}
			for i, input := range fixture.Inputs {
				if got := compiled.MatchString(input); got != accepted[i] {
					t.Errorf("%q: Go=%v Pi=%v", input, got, accepted[i])
				}
			}
		})
	}
}
