package ai

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestPiBlockReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-block-references.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name      string
				Messages  []Message
				Model     Model
				Normalize bool
			}
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 18 {
		t.Fatal("unexpected block reference coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			before, err := json.Marshal(tc.Input.Messages)
			if err != nil {
				t.Fatal(err)
			}
			originals := map[*ContentBlock]int{}
			for i, block := range tc.Input.Messages[0].Content.Blocks.Values() {
				originals[block] = i
			}
			var normalize ToolCallIDNormalizer
			if tc.Input.Normalize {
				normalize = func(id string, _ *Model, _ *Message) string { return "normalized-" + id }
			}
			messages := transformMessagesAt(tc.Input.Messages, tc.Input.Model, normalize, func() int64 { return 100 })
			indices := make([][]int, len(messages))
			for i, message := range messages {
				indices[i] = []int{}
				for _, block := range message.Content.Blocks.Values() {
					index, found := originals[block]
					if !found {
						index = -1
					}
					indices[i] = append(indices[i], index)
				}
			}
			after, err := json.Marshal(tc.Input.Messages)
			if err != nil {
				t.Fatal(err)
			}
			if string(after) != string(before) {
				t.Fatalf("transformation mutated source: %s", after)
			}
			actual, err := json.Marshal(map[string]any{"messages": messages, "indices": indices})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err = json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
