package engine_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
)

func TestToolResultToJSONKeysOrderAndOmission(t *testing.T) {
	for _, omit := range []bool{false, true} {
		calls := []string{}
		structured := engine.NewObject(engine.Property{Name: "value", Value: 1})
		details := engine.NewObject()
		details.Set("toJSON", engine.JSONMethod(func(receiver any, key string) (any, error) {
			if receiver != details {
				t.Fatal("wrong details receiver")
			}
			calls = append(calls, key)
			structured.Set("value", 2)
			if omit {
				return engine.Undefined, nil
			}
			return "metadata", nil
		}))
		structured.Set("toJSON", engine.JSONMethod(func(receiver any, key string) (any, error) {
			if receiver != structured {
				t.Fatal("wrong structured content receiver")
			}
			calls = append(calls, key)
			return structured.Get("value"), nil
		}))
		raw, err := json.Marshal(engine.ToolResult{Details: details, StructuredContent: structured})
		if err != nil {
			t.Fatal(err)
		}
		expected := `{"details":"metadata","structuredContent":2}`
		if omit {
			expected = `{"structuredContent":2}`
		}
		if string(raw) != expected || !reflect.DeepEqual(calls, []string{"details", "structuredContent"}) {
			t.Fatalf("export: %s calls=%v", raw, calls)
		}
	}
}
