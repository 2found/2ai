package agentcore_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/core/ai"
)

// A resumed run appends a system message only when the prompt really changed:
// the same definition resumes on the transcript's own system message, and a
// changed one supersedes it with exactly one new message.
func TestNativeResumeAppendsTheSystemPromptOnlyOnChange(t *testing.T) {
	var views [][]ai.Message
	run := func(soul string, state json.RawMessage) agentcore.RunResult {
		t.Helper()
		provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"m"}`), Stream: func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
			views = append(views, view.Messages())
			return ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "ok"})})(ctx, model, view, options)
		}}}}
		a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "m", Definition: agentcore.AgentDefinition{Soul: soul}, Tools: agentcore.NewToolSet(), Policy: agentcore.NewAllowList()})
		if err != nil {
			t.Fatal(err)
		}
		result, err := a.RunNative(context.Background(), agentcore.NativeRun{State: state, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "go"}}})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	systems := func(view []ai.Message) (n int, last string) {
		for _, message := range view {
			if message.Role == "system" {
				n++
				for _, section := range message.Sections {
					if section.Value != nil {
						last = *section.Value
					}
				}
			}
		}
		return n, last
	}
	first := run("You are Ada.", nil)
	second := run("You are Ada.", first.NativeState)
	if n, last := systems(views[1]); n != 1 || !strings.Contains(last, "You are Ada.") {
		t.Fatalf("an unchanged prompt was appended again on resume: %d system messages", n)
	}
	run("You are Grace.", second.NativeState)
	if n, last := systems(views[2]); n != 2 || !strings.Contains(last, "You are Grace.") {
		t.Fatalf("a changed prompt did not supersede the old one: %d system messages, last %q", n, last)
	}
}
