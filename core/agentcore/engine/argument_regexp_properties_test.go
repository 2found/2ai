package engine

import (
	"encoding/json"
	"os"
	"strconv"
	"testing"
)

func TestPiUnicodePropertyOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-unicode-properties.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, Bun, Unicode string
		CaptureNames                 [][3]json.RawMessage
		Cases                        []struct {
			Property string
			Aliases  []string
			Probes   [][2]json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || fixture.Bun != "1.3.14" || fixture.Unicode != "15.1" || len(fixture.Cases) != 419 {
		t.Fatal("unexpected Unicode property oracle")
	}
	if len(fixture.CaptureNames) != 4640 {
		t.Fatal("unexpected capture-name probe coverage")
	}
	t.Run("CaptureNames", func(t *testing.T) {
		for _, row := range fixture.CaptureNames {
			var code rune
			var start, continuation bool
			if err := json.Unmarshal(row[0], &code); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(row[1], &start); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(row[2], &continuation); err != nil {
				t.Fatal(err)
			}
			for _, text := range []string{string(code), `\u{` + strconv.FormatInt(int64(code), 16) + `}`} {
				for _, position := range []struct {
					prefix string
					want   bool
				}{{"", start}, {"a", continuation}} {
					name, got := argumentRegexpCaptureName(position.prefix + text)
					if got != position.want || got && name != position.prefix+string(code) {
						t.Fatalf("capture name %q: Go=%q/%v Pi=%v", position.prefix+text, name, got, position.want)
					}
				}
			}
		}
	})
	aliases := 0
	for _, row := range fixture.Cases {
		t.Run(row.Property, func(t *testing.T) {
			type probe struct {
				text string
				want bool
			}
			probes := make([]probe, len(row.Probes))
			for i, raw := range row.Probes {
				var code rune
				if err := json.Unmarshal(raw[0], &code); err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(raw[1], &probes[i].want); err != nil {
					t.Fatal(err)
				}
				probes[i].text = string(code)
				if got := argumentUnicodeHas(row.Property, code); got != probes[i].want {
					t.Fatalf("property lookup %s on %U: Go=%v Pi=%v", row.Property, code, got, probes[i].want)
				}
			}
			for _, form := range []struct {
				prefix, suffix string
				negate         bool
			}{{`^\p{`, `}$`, false}, {`^\P{`, `}$`, true}, {`^[\p{`, `}]$`, false}, {`^[\P{`, `}]$`, true}} {
				pattern := form.prefix + row.Property + form.suffix
				compiled, err := compileArgumentRegexp(pattern)
				if err != nil {
					t.Fatal(err)
				}
				for _, probe := range probes {
					want := probe.want != form.negate
					if got := compiled.MatchString(probe.text); got != want {
						t.Fatalf("%s on %U: Go=%v Pi=%v", pattern, []rune(probe.text)[0], got, want)
					}
				}
			}
			// Exercise every accepted alias through the parser and matcher, with
			// one accepting and one rejecting probe when the set permits them.
			for _, alias := range row.Aliases {
				compiled, err := compileArgumentRegexp(`^\p{` + alias + `}$`)
				if err != nil {
					t.Fatalf("alias %s: %v", alias, err)
				}
				seen := map[bool]bool{}
				for _, probe := range probes {
					if seen[probe.want] {
						continue
					}
					if got := compiled.MatchString(probe.text); got != probe.want {
						t.Fatalf("alias %s on %U: Go=%v Pi=%v", alias, []rune(probe.text)[0], got, probe.want)
					}
					seen[probe.want] = true
				}
			}
		})
		aliases += len(row.Aliases)
	}
	if aliases != 1627 {
		t.Fatalf("unexpected alias coverage: %d", aliases)
	}
}
