package ai

import "testing"

func TestNativeClientDeveloperRoleOnlyForOpenAIAPI(t *testing.T) {
	for _, tc := range []struct {
		baseURL   string
		developer bool
	}{
		{"https://api.openai.com/v1", true},
		{"https://ai.example.com/v1", false},
		{"http://127.0.0.1:7812/v1", false},
	} {
		client, err := NewNativeClient(ClientSpec{Name: "openai", APIKey: "key", BaseURL: tc.baseURL})
		if err != nil {
			t.Fatal(err)
		}
		candidate, err := client.Candidate("m", 100000)
		if err != nil {
			t.Fatal(err)
		}
		compat, err := ResolveOpenAICompletionsCompat(candidate.Model)
		if err != nil {
			t.Fatal(err)
		}
		if compat.SupportsDeveloperRole != tc.developer {
			t.Fatalf("%s: supportsDeveloperRole=%v, want %v", tc.baseURL, compat.SupportsDeveloperRole, tc.developer)
		}
	}
}
