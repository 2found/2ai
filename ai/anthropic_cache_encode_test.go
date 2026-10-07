package ai

import (
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// TestAnthropicEncode_CacheBreakpoints pins the two prompt-cache breakpoints the
// encoder stamps when a CacheKey is set: one on the system block (covers
// tools+system) and one on the final block of the final message (the moving
// breakpoint that turns the whole transcript-so-far into a cached prefix).
func TestAnthropicEncode_CacheBreakpoints(t *testing.T) {
	p := NewAnthropicProvider("k", "")
	req := protocol.ChatRequest{
		Model:    "claude-opus-4-8",
		CacheKey: "run-1",
		Messages: []protocol.Message{
			{Role: protocol.RoleSystem, Content: "you are a reviewer"},
			{Role: protocol.RoleUser, Content: "big diff payload"},
			{Role: protocol.RoleAssistant, Content: "", ToolCalls: []protocol.ToolCall{{ID: "t1", Name: "read_file", Arguments: `{"path":"a.go"}`}}},
			{Role: protocol.RoleTool, ToolCallID: "t1", Content: "file body"},
		},
	}
	out := p.encode(req)

	sys, ok := out.System.([]antSystemBlock)
	if !ok || len(sys) != 1 || sys[0].CacheControl == nil {
		t.Fatalf("system must be one block carrying cache_control, got %#v", out.System)
	}
	last := out.Messages[len(out.Messages)-1]
	lastBlock := last.Content[len(last.Content)-1]
	if lastBlock.CacheControl == nil {
		t.Fatalf("final block of final message must carry the moving breakpoint, got %#v", lastBlock)
	}
	if lastBlock.CacheControl.Type != "ephemeral" {
		t.Fatalf("breakpoint type = %q, want ephemeral", lastBlock.CacheControl.Type)
	}
	// Only the final message carries a breakpoint — earlier messages must not.
	for i, m := range out.Messages[:len(out.Messages)-1] {
		for j, b := range m.Content {
			if b.CacheControl != nil {
				t.Fatalf("unexpected cache_control on message %d block %d", i, j)
			}
		}
	}
}

// TestAnthropicEncode_NoCacheKeyNoBreakpoints keeps the uncached path byte-stable:
// a bare-string system and no cache_control anywhere.
func TestAnthropicEncode_NoCacheKeyNoBreakpoints(t *testing.T) {
	p := NewAnthropicProvider("k", "")
	out := p.encode(protocol.ChatRequest{
		Model: "claude-opus-4-8",
		Messages: []protocol.Message{
			{Role: protocol.RoleSystem, Content: "sys"},
			{Role: protocol.RoleUser, Content: "hi"},
		},
	})
	if _, ok := out.System.(string); !ok {
		t.Fatalf("uncached system must stay a bare string, got %#v", out.System)
	}
	for i, m := range out.Messages {
		for j, b := range m.Content {
			if b.CacheControl != nil {
				t.Fatalf("unexpected cache_control on message %d block %d", i, j)
			}
		}
	}
}
