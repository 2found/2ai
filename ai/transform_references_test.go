package ai

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

func TestPiTransformReferences(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-transform-references.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name, Phase string
				Messages    []*Message
				Model       *Model
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 56 {
		t.Fatal("unexpected transform reference coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			input := tc.Input
			messages, model := input.Messages, input.Model
			originalMessages, originalBlocks := map[*Message]int{}, map[*ContentBlock]int{}
			for i, message := range messages {
				originalMessages[message] = i
				for _, block := range message.Content.Blocks.Values() {
					originalBlocks[block] = len(originalBlocks)
				}
			}
			modelJSON := func(value *Model) any {
				return map[string]any{"id": value.ID, "api": value.API, "provider": value.Provider, "input": value.Input}
			}
			callbacks := []json.RawMessage{}
			normalize := func(id string, target *Model, source *Message) string {
				raw, err := json.Marshal(map[string]any{"id": id, "sameModel": target == model, "sameSource": source == messages[1] || source == messages[5], "source": source, "model": modelJSON(target)})
				if err != nil {
					t.Fatal(err)
				}
				callbacks = append(callbacks, raw)
				if id != "call" {
					return id
				}
				block := source.Content.Blocks.Get(0)
				switch input.Phase {
				case "id_equal":
					block.ID = "changed"
					return "changed"
				case "id_different":
					block.ID = "changed"
					return "normalized"
				case "fields":
					block.Name = "edited"
					block.Arguments = json.RawMessage(`{"edited":true}`)
					signature := "changed"
					block.ThoughtSignature = &signature
				case "source_fields":
					source.Timestamp = 9
					source.Provider = "changed"
				case "source_replace":
					source.Content = BlockContent(ContentBlock{Type: "text", Text: "replacement"})
				case "source_stop":
					source.StopReason = "error"
				case "next_slot":
					source.Content.Blocks.Set(1, &ContentBlock{Type: "text", Text: "replacement"})
				case "next_grow":
					for i := 0; i < 32; i++ {
						call := &ContentBlock{Type: "toolCall", ID: fmt.Sprintf("extra-%d", i), Name: "echo", Arguments: json.RawMessage(`{}`), ThoughtSignature: block.ThoughtSignature}
						source.Content.Blocks.Append(call)
					}
				case "model_input":
					if len(target.Input) == 2 {
						target.Input = []string{"text"}
					} else {
						target.Input = []string{"text", "image"}
					}
				case "model_id":
					target.ID = "source"
				case "prior_message":
					messages[0].Timestamp = 9
					messages[0].Content.Blocks.Get(1).Text = "changed"
				case "later_message":
					messages[4].Timestamp = 9
					messages[4].Content.Blocks.Get(1).Text = "changed"
				case "later_content":
					messages[4].Content = BlockContent(ContentBlock{Type: "text", Text: "replacement"})
				}
				return id
			}
			output := transformMessageReferencesAt(messages, model, normalize, func() int64 { return 100 })
			messageIndices, blockIndices := []int{}, [][]int{}
			for _, message := range output {
				index, found := originalMessages[message]
				if !found {
					index = -1
				}
				messageIndices = append(messageIndices, index)
				indices := []int{}
				for _, block := range message.Content.Blocks.Values() {
					index, found := originalBlocks[block]
					if !found {
						index = -1
					}
					indices = append(indices, index)
				}
				blockIndices = append(blockIndices, indices)
			}
			assertPiJSON(t, tc.Expected, map[string]any{"messages": output, "source": messages, "model": modelJSON(model), "callbacks": callbacks, "messageIndices": messageIndices, "blockIndices": blockIndices})
		})
	}
}
