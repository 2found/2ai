package ai

import (
	"testing"

	"github.com/2found/2ai/core/ai/protocol"
)

// With caching on, the final message carries the one moving breakpoint and no
// earlier message is marked; without a cache key nothing is marked.
func TestAnthropicEncodeMovingBreakpoint(t *testing.T) {
	p := NewAnthropicProvider("k", "")
	messages := []protocol.Message{
		{Role: protocol.RoleUser, Content: "u1"},
		{Role: protocol.RoleAssistant, Content: "a1"},
	}
	out := p.encode(protocol.ChatRequest{Model: "claude-x", CacheKey: "s", Messages: messages})
	last := out.Messages[len(out.Messages)-1]
	if last.Content[len(last.Content)-1].CacheControl == nil {
		t.Fatal("moving breakpoint missing on final message")
	}
	first := out.Messages[0]
	if first.Content[len(first.Content)-1].CacheControl != nil {
		t.Fatal("an earlier message was marked")
	}
	for i, m := range p.encode(protocol.ChatRequest{Model: "claude-x", Messages: messages}).Messages {
		if m.Content[len(m.Content)-1].CacheControl != nil {
			t.Fatalf("message %d marked without a cache key", i)
		}
	}
}
