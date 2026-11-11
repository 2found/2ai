package memory

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

// The consolidation pass is a one-shot request whose prompt is never sent
// again. Paying the provider's cache-write premium for it buys nothing, so it
// must opt out of caching on every candidate, including a fallback.
func TestConsolidationRequestsDisableCaching(t *testing.T) {
	var seen []any
	record := func(reply ai.Message) ai.StreamFn {
		return func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
			seen = append(seen, options["cacheRetention"])
			return ai.ScriptedStream(reply)(ctx, model, view, options)
		}
	}
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"failed"}`), Stream: record(ai.Message{Role: "assistant", StopReason: "error"})},
		{Model: json.RawMessage(`{"id":"memory"}`), Stream: record(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: `{"changes":[]}`})})},
	}}
	if _, err := nativeConsolidator(provider, agentcore.RunInfo{})(context.Background(), Consolidation{}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 {
		t.Fatalf("attempts = %d, want 2 (failure then fallback)", len(seen))
	}
	for i, retention := range seen {
		if retention != "none" {
			t.Fatalf("attempt %d cacheRetention = %v, want \"none\"", i+1, retention)
		}
	}
}
