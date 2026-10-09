package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

func TestNativeConsolidationBoundsCompleteJSONWithoutMutatingEvidence(t *testing.T) {
	review := Consolidation{}
	for i := 0; i < 4; i++ {
		r := Rollout{ID: fmt.Sprintf("%064d", i), Final: strings.Repeat("final ", 1000)}
		for j := 0; j < 32; j++ {
			r.Messages = append(r.Messages, agentcore.Message{Role: agentcore.RoleUser, Content: strings.Repeat("Tiếng Việt 🌏 \"\\\n", 200)})
		}
		review.Rollouts = append(review.Rollouts, r)
	}
	for i := 0; i < 32; i++ {
		review.Memories = append(review.Memories, agentcore.MemoryEntry{ID: fmt.Sprint(i), ScopeID: "scope", Content: strings.Repeat("remember ", 800), Tags: []string{"fixture"}})
	}
	before, _ := json.Marshal(review)
	calls := 0
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"fixture"}`), Stream: func(ctx context.Context, model json.RawMessage, transcript ai.TranscriptContext, opts map[string]any) (*ai.AssistantMessageEventStream, error) {
		calls++
		messages := transcript.Messages()
		raw := []byte(*messages[len(messages)-1].Content.Text)
		var got Consolidation
		if len(raw) > 60000 || json.Unmarshal(raw, &got) != nil {
			t.Errorf("large evidence is not bounded complete JSON: %d bytes, valid=%v", len(raw), json.Valid(raw))
		}
		if len(got.Rollouts) != 4 || len(got.Memories) != 32 {
			t.Errorf("evidence or memory IDs disappeared: %d/%d", len(got.Rollouts), len(got.Memories))
		} else {
			for i, r := range got.Rollouts {
				if r.ID != review.Rollouts[i].ID || len(r.Messages) != 32 || r.Messages[0].Content == "" {
					t.Error("rollout identity or all message evidence lost")
				}
			}
			for i, m := range got.Memories {
				if m.ID != review.Memories[i].ID || m.ScopeID != "scope" {
					t.Error("memory identity changed")
				}
			}
		}
		return ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.TextContent(`{"changes":[]}`)})(ctx, model, transcript, opts)
	}}}}
	if _, err := nativeConsolidator(provider, agentcore.RunInfo{})(context.Background(), review); err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(review)
	if string(before) != string(after) || calls != 1 {
		t.Fatal("source evidence mutated or calls amplified")
	}
	// An unbounded identity cannot be abbreviated safely. Refuse it before a
	// provider call, retaining the original pending evidence for the host.
	review.Rollouts[0].ID = strings.Repeat("id", 40000)
	if _, err := nativeConsolidator(provider, agentcore.RunInfo{})(context.Background(), review); err == nil || calls != 1 {
		t.Fatal("oversized immutable metadata reached provider")
	}
}
