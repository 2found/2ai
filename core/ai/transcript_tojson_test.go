package ai

import (
	"encoding/json"
	"testing"
)

func TestMessageDetailsToJSONKeyAndOmission(t *testing.T) {
	for _, omit := range []bool{false, true} {
		calls := 0
		details := NewObject()
		details.Set("toJSON", JSONMethod(func(receiver any, key string) (any, error) {
			calls++
			if receiver != details || key != "details" {
				t.Fatalf("wrong receiver or key: %T %q", receiver, key)
			}
			if omit {
				return Undefined, nil
			}
			return nil, nil
		}))
		var message Message
		if err := json.Unmarshal([]byte(`{"role":"toolResult","details":{"old":true},"content":[],"toolCallId":"call","toolName":"tool","isError":false,"timestamp":0}`), &message); err != nil {
			t.Fatal(err)
		}
		message.Details = details
		raw, err := json.Marshal(message)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("hook called %d times", calls)
		}
		if omit {
			if _, exists := fields["details"]; exists {
				t.Fatalf("omitted details restored: %s", raw)
			}
		} else if string(fields["details"]) != "null" {
			t.Fatalf("explicit null omitted: %s", raw)
		}
	}
}
