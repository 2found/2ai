package jsonjs

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
)

func TestPiJSONSyntax(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-json-syntax.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct{ Input, Error json.RawMessage }
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 2064 {
		t.Fatal("unexpected JSON syntax coverage")
	}
	for i, tc := range fixture.Cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			input, err := DecodeJSON(tc.Input)
			if err != nil {
				t.Fatal(err)
			}
			expected, err := DecodeJSON(tc.Error)
			if err != nil {
				t.Fatal(err)
			}
			var actual any
			if err := ValidateJSON([]byte(input.(string))); err != nil {
				actual = err.Error()
			}
			if actual != expected {
				t.Fatalf("input %q: Go %q, Pi %q", input, actual, expected)
			}
		})
	}
}

func TestJSONSyntaxDeepNesting(t *testing.T) {
	// This exceeds encoding/json's nesting limit, without recursion in our validator.
	input := strings.Repeat("[", 20000) + "0" + strings.Repeat("]", 20000)
	if err := ValidateJSON([]byte(input)); err != nil {
		t.Fatal(err)
	}
	if err := ValidateJSON([]byte(input[:len(input)-1])); err == nil || err.Error() != "JSON Parse error: Expected ']'" {
		t.Fatal(err)
	}
}
