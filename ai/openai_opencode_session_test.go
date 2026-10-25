package ai

import (
	"encoding/json"
	"regexp"
	"testing"
)

func TestOpenCodeGoSendsSessionHeader(t *testing.T) {
	raw := json.RawMessage(`{"id":"deepseek-v4.1-flash","provider":"opencode-go","baseUrl":"https://opencode.ai/zen/go/v1"}`)
	compat, err := ResolveOpenAICompletionsCompat(raw)
	if err != nil {
		t.Fatal(err)
	}
	first := TranscriptContext{messages: []Message{{Role: "user"}}}
	headers, err := completionsHeaders(raw, first, map[string]json.RawMessage{}, compat, "key")
	if err != nil {
		t.Fatal(err)
	}
	derived := headers.Get("x-opencode-session")
	if derived == "" {
		t.Fatal("opencode-go request has no x-opencode-session")
	}
	again, _ := completionsHeaders(raw, first, map[string]json.RawMessage{}, compat, "key")
	if again.Get("x-opencode-session") != derived {
		t.Fatal("derived session changed between turns of one transcript")
	}
	// A caller session (the prompt cache key) fixes the header for the whole
	// conversation, whatever its transcript, in the derived shape.
	conv := map[string]json.RawMessage{"sessionId": json.RawMessage(`"conv-1"`), "cacheRetention": json.RawMessage(`"none"`)}
	explicit, _ := completionsHeaders(raw, first, conv, compat, "key")
	other := TranscriptContext{messages: []Message{{Role: "system", Content: TextContent("changed prompt")}}}
	same, _ := completionsHeaders(raw, other, conv, compat, "key")
	got := explicit.Get("x-opencode-session")
	if got == derived || got != same.Get("x-opencode-session") || !regexp.MustCompile(`^ses_[0-9a-f]{24}$`).MatchString(got) {
		t.Fatalf("x-opencode-session = %q (other transcript %q, derived %q)", got, same.Get("x-opencode-session"), derived)
	}

	zen := json.RawMessage(`{"id":"deepseek-v4.1-flash","provider":"opencode-zen","baseUrl":"https://opencode.ai/zen/v1"}`)
	zenCompat, _ := ResolveOpenAICompletionsCompat(zen)
	zenHeaders, _ := completionsHeaders(zen, first, map[string]json.RawMessage{}, zenCompat, "key")
	if zenHeaders.Get("x-opencode-session") != "" {
		t.Fatal("opencode-zen does not need the Go session header")
	}
}
