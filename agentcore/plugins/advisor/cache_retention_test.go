package advisor_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/advisor"
	"github.com/2found/2ai/ai"
)

// A review is a one-shot request whose prompt is never sent again. Paying the
// provider's cache-write premium for it buys nothing, so every reviewer attempt,
// including a fallback, opts out of caching. The primary run keeps its own
// setting.
func TestReviewerRequestsDisableCaching(t *testing.T) {
	var reviewer []any
	record := func(reply ai.Message) ai.StreamFn {
		return func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
			reviewer = append(reviewer, options["cacheRetention"])
			return ai.ScriptedStream(reply)(ctx, model, view, options)
		}
	}
	reviews := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"failed"}`), Stream: record(ai.Message{Role: "assistant", StopReason: "error"})},
		{Model: json.RawMessage(`{"id":"reviewer"}`), Stream: record(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: `{"notes":[]}`})})},
	}}
	var primary []any
	a, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{Model: "test", NativeProvider: provider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		primary = append(primary, options["cacheRetention"])
		return ai.ScriptedStream(nativeAnswer("primary answer"))(ctx, model, view, options)
	})}), advisor.Native(reviews))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Prompt(context.Background(), "work"); err != nil {
		t.Fatal(err)
	}
	if len(reviewer) != 2 {
		t.Fatalf("reviewer attempts = %d, want 2 (failure then fallback)", len(reviewer))
	}
	for i, retention := range reviewer {
		if retention != "none" {
			t.Fatalf("reviewer attempt %d cacheRetention = %v, want \"none\"", i+1, retention)
		}
	}
	for _, retention := range primary {
		if retention == "none" {
			t.Fatal("the reviewer's cache opt-out leaked into the primary run")
		}
	}
}
