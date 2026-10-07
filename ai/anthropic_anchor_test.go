package ai

import (
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// Anchor placement is the LOOP's (agentcore/cacheanchor.go); translating a
// placed anchor into Anthropic's cache_control blocks is this package's. These
// pin the second half.

// TestAnthropicEncodeHonorsAnchors: the provider translates loop-placed
// anchors instead of deciding placement itself; system-role anchors (hoisted
// into the system block) are skipped, and only the last 3 anchors survive so
// the request stays within Anthropic's 4-breakpoint cap.
func TestAnthropicEncodeHonorsAnchors(t *testing.T) {
	p := NewAnthropicProvider("k", "")
	req := protocol.ChatRequest{
		Model:    "claude-x",
		CacheKey: "s",
		Messages: []protocol.Message{
			{Role: protocol.RoleSystem, Content: "sys", CacheAnchor: true}, // hoisted — no attachment point
			{Role: protocol.RoleUser, Content: "u1", CacheAnchor: true},
			{Role: protocol.RoleAssistant, Content: "a1", CacheAnchor: true},
			{Role: protocol.RoleUser, Content: "u2", CacheAnchor: true},
			{Role: protocol.RoleAssistant, Content: "a2", CacheAnchor: true},
			{Role: protocol.RoleUser, Content: "u3"},
		},
	}
	out := p.encode(req)
	var marked []int
	for i, m := range out.Messages {
		if m.Content[len(m.Content)-1].CacheControl != nil {
			marked = append(marked, i)
		}
	}
	// Four non-system anchors → capped to the newest three (indices 1,2,3 of
	// out.Messages: a1, u2, a2). u3 carries no anchor and must stay unmarked.
	if len(marked) != 3 || marked[0] != 1 || marked[1] != 2 || marked[2] != 3 {
		t.Fatalf("marked = %v, want [1 2 3]", marked)
	}
}

// TestAnthropicEncodeAnchorFallback: with caching on but no anchors (provider
// used standalone), the classic moving breakpoint on the final message must
// still apply.
func TestAnthropicEncodeAnchorFallback(t *testing.T) {
	p := NewAnthropicProvider("k", "")
	out := p.encode(protocol.ChatRequest{
		Model:    "claude-x",
		CacheKey: "s",
		Messages: []protocol.Message{
			{Role: protocol.RoleUser, Content: "u1"},
			{Role: protocol.RoleAssistant, Content: "a1"},
		},
	})
	last := out.Messages[len(out.Messages)-1]
	if last.Content[len(last.Content)-1].CacheControl == nil {
		t.Fatal("fallback moving breakpoint missing on final message")
	}
	first := out.Messages[0]
	if first.Content[len(first.Content)-1].CacheControl != nil {
		t.Fatal("fallback must not mark earlier messages")
	}
}

// TestAnthropicEncodeDoesNotFallBackOverADeliberateAnchor covers the case where
// the loop's placement and this encoder's fallback disagree.
//
// The loop moves the breakpoint OFF the final message on purpose: a ContextHook
// re-renders a trailer there every turn, so a prefix ending in it can never be
// re-read. If the only anchor the loop placed happens to land on a system
// message — hoisted into the system block here, with no attachment point — the
// fallback used to fire and re-anchor the final message, quietly restoring the
// behavior the placement exists to avoid. No anchors REQUESTED is standalone
// use; anchors requested and unmappable is not.
func TestAnthropicEncodeDoesNotFallBackOverADeliberateAnchor(t *testing.T) {
	p := NewAnthropicProvider("k", "")
	out := p.encode(protocol.ChatRequest{
		Model:    "claude-x",
		CacheKey: "s",
		Messages: []protocol.Message{
			{Role: protocol.RoleUser, Content: "u1"},
			{Role: protocol.RoleAssistant, Content: "a1"},
			// The loop's chosen breakpoint: a system message (a compaction
			// checkpoint or the pinned requirement), hoisted out of Messages.
			{Role: protocol.RoleSystem, Content: "checkpoint", CacheAnchor: true},
			// The hook's regenerated trailer, which must NOT be anchored.
			{Role: protocol.RoleUser, Content: "[run plan]\n[~] step one"},
		},
	})
	for i, m := range out.Messages {
		if m.Content[len(m.Content)-1].CacheControl != nil {
			t.Fatalf("message %d was anchored by the fallback although the loop had already placed "+
				"a breakpoint; the entry it writes ends in a trailer that is re-rendered every turn "+
				"and can never be read back", i)
		}
	}
}
