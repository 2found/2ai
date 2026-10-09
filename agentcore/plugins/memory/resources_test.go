package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
)

func largeRolloutFixture() agentcore.RunResult {
	result := agentcore.RunResult{Turns: 1, StopReason: "stop", Final: "done"}
	for range 8 {
		m := agentcore.Message{Role: agentcore.RoleAssistant, Content: strings.Repeat("evidence ", 32768)}
		for range 8 {
			m.ToolCalls = append(m.ToolCalls, agentcore.ToolCall{Name: "inspect", Arguments: `{"query":"bounded"}`})
		}
		result.Messages = append(result.Messages, m)
	}
	return result
}

func BenchmarkLargeRolloutStaging(b *testing.B) {
	store := &consolidationStore{}
	c := &curation{store: store, consolidation: store, scopeID: "scope", worker: NewConsolidationWorker(1), consolidator: func(context.Context, Consolidation) ([]Change, error) { return nil, nil }}
	result := largeRolloutFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		store.pending = nil
		if err := c.FinalizeRun(context.Background(), result, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func TestRolloutDigestAndEvidenceCompatibility(t *testing.T) {
	store := &consolidationStore{}
	c := &curation{store: store, consolidation: store, scopeID: "scope", worker: NewConsolidationWorker(1), consolidator: func(context.Context, Consolidation) ([]Change, error) { return nil, nil }}
	for _, messages := range [][]agentcore.Message{nil, {}, {{Role: agentcore.RoleSystem, Content: "<policy>"}, {Role: agentcore.RoleUser, Content: "Tiếng Việt 🌏\n"}, {Role: agentcore.RoleAssistant, Content: "answer"}}, largeRolloutFixture().Messages} {
		store.pending = nil
		result := agentcore.RunResult{Messages: messages, Turns: 1, StopReason: "stop", Final: "done"}
		if err := c.FinalizeRun(context.Background(), result, nil); err != nil {
			t.Fatal(err)
		}
		if len(store.pending) != 1 {
			t.Fatal("evidence lost")
		}
		raw, _ := json.Marshal(messages)
		sum := sha256.Sum256(raw)
		got := store.pending[0]
		if got.ID != hex.EncodeToString(sum[:]) {
			t.Fatal("durable digest changed")
		}
		bytes := 0
		for _, m := range got.Messages {
			bytes += len(m.Content)
			if m.Role == agentcore.RoleSystem {
				t.Fatal("system prompt in evidence")
			}
		}
		if bytes > 32000 || len(got.Messages) > 32 {
			t.Fatal("unbounded evidence")
		}
		if len(messages) == 3 && (got.Messages[0].Content != messages[1].Content || got.Messages[1].Content != "answer") {
			t.Fatal("evidence order changed")
		}
	}
}
