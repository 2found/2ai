package ai

import (
	"encoding/json"
	"os"
	"testing"
)

func TestPiCodexContinuationOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-codex-continuation.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name         string
				Body         json.RawMessage
				Continuation *codexWebSocketContinuation
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 24 {
		t.Fatal("unexpected continuation oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			before, _ := json.Marshal(tc.Input)
			continuation := tc.Input.Continuation
			body, err := codexCachedWebSocketBody(tc.Input.Body, &continuation)
			if err != nil {
				t.Fatal(err)
			}
			assertPiJSON(t, tc.Expected, map[string]any{"body": body, "retained": continuation != nil})
			after, _ := json.Marshal(tc.Input)
			if string(before) != string(after) {
				t.Fatal("continuation preparation mutated caller state")
			}
		})
	}
}
