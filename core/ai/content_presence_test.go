package ai

import (
	"encoding/json"
	"testing"
)

func TestDecodedContentPresenceSurvivesReplacement(t *testing.T) {
	var original Message
	if err := json.Unmarshal([]byte(`{"role":"system","timestamp":1}`), &original); err != nil {
		t.Fatal(err)
	}
	if original.HasContent() {
		t.Fatal("absent content became present")
	}
	for _, mode := range []string{"null", "decoded_null", "text_field", "blocks_field"} {
		t.Run(mode, func(t *testing.T) {
			message := original
			expected := `null`
			switch mode {
			case "null":
				message.Content = MessageContent{}
			case "decoded_null":
				if err := json.Unmarshal([]byte(`null`), &message.Content); err != nil {
					t.Fatal(err)
				}
			case "text_field":
				text := "body"
				message.Content.Text = &text
				expected = `"body"`
			case "blocks_field":
				message.Content.Blocks = NewBlockList()
				expected = `[]`
			}
			if !message.HasContent() {
				t.Fatal("replacement kept content absent")
			}
			raw, err := json.Marshal(message)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &fields); err != nil {
				t.Fatal(err)
			}
			if string(fields["content"]) != expected {
				t.Fatalf("content: %s", raw)
			}
			if original.HasContent() {
				t.Fatal("replacement mutated original presence")
			}
		})
	}
	raw, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	if _, present := fields["content"]; present {
		t.Fatalf("original presence changed: %s", raw)
	}
}
