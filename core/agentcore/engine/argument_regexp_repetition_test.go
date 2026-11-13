package engine

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
)

func BenchmarkPiNullableRepetition(b *testing.B) {
	for _, size := range []int{128, 1024, 4096, 8192} {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			pattern, err := compileArgumentRegexp(`^(a?)*\1$`)
			if err != nil {
				b.Fatal(err)
			}
			input := strings.Repeat("a", size)
			if !pattern.MatchString(input) {
				b.Fatal("expected a match")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if !pattern.MatchString(input) {
					b.Fatal("expected a match")
				}
			}
		})
	}
}

func TestPiRegexpRepetitionOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-arguments.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Repetitions    struct {
			Inputs []string
			Cases  []struct {
				Name, Pattern string
				Accepted      []int
			}
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Repetitions.Inputs) != 364 || len(fixture.Repetitions.Cases) != 32 {
		t.Fatal("unexpected repetition oracle coverage")
	}
	for _, row := range fixture.Repetitions.Cases {
		t.Run(row.Name, func(t *testing.T) {
			pattern, err := compileArgumentRegexp(row.Pattern)
			if err != nil {
				t.Fatal(err)
			}
			accepted := map[int]bool{}
			for _, index := range row.Accepted {
				accepted[index] = true
			}
			for index, input := range fixture.Repetitions.Inputs {
				if got := pattern.MatchString(input); got != accepted[index] {
					t.Fatalf("%s on %q: Go=%v Pi=%v", row.Pattern, input, got, accepted[index])
				}
			}
		})
	}
}
