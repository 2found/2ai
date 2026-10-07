package ai_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/2found/2ai/ai"
)

func TestMessageModelIdentityJSONAndReplacement(t *testing.T) {
	for _, role := range []string{"assistant", "custom"} {
		for _, value := range []string{`null`, `false`, `42`, `["x"]`, `{"name":"p"}`, `""`, `"test"`} {
			t.Run(role+"/"+value, func(t *testing.T) {
				raw := []byte(`{"role":"` + role + `","content":[],"api":` + value + `,"provider":` + value + `,"model":` + value + `}`)
				var message ai.Message
				if err := json.Unmarshal(raw, &message); err != nil {
					t.Fatal(err)
				}
				decode := func(raw []byte) map[string]any {
					var result map[string]any
					if err := json.Unmarshal(raw, &result); err != nil {
						t.Fatal(err)
					}
					return result
				}
				check := func(message ai.Message, expected map[string]any) {
					encoded, err := json.Marshal(message)
					if err != nil {
						t.Fatal(err)
					}
					if actual := decode(encoded); !reflect.DeepEqual(actual, expected) {
						t.Fatalf("got %s, want %v", encoded, expected)
					}
				}
				original := decode(raw)
				check(message, original)
				copy := message
				copy.API, copy.Provider, copy.Model = "new-api", "new-provider", "new-model"
				updated := decode(raw)
				updated["api"], updated["provider"], updated["model"] = "new-api", "new-provider", "new-model"
				check(copy, updated)
				check(message, original)
			})
		}
	}
}
