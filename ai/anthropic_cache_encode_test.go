package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
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

// nativeBreakpoints returns the JSON paths carrying cache_control in a request.
func nativeBreakpoints(t *testing.T, params json.RawMessage) (count int, messages []int, system int) {
	t.Helper()
	var body struct {
		System   []map[string]any
		Tools    []map[string]any
		Messages []struct{ Content any }
	}
	if err := json.Unmarshal(params, &body); err != nil {
		t.Fatal(err)
	}
	for _, block := range body.System {
		if block["cache_control"] != nil {
			count++
			system++
		}
	}
	for _, tool := range body.Tools {
		if tool["cache_control"] != nil {
			count++
		}
	}
	for i, message := range body.Messages {
		blocks, _ := message.Content.([]any)
		for _, block := range blocks {
			if fields, _ := block.(map[string]any); fields["cache_control"] != nil {
				count++
				messages = append(messages, i)
			}
		}
	}
	return count, messages, system
}

// TestAnthropicNativeLongToolLoopKeepsReadingCache: with one moving breakpoint
// a turn that adds more than ~20 blocks (wide parallel tool use) leaves the
// previous write outside Anthropic's lookback window, so every request writes
// the whole transcript again. The request must also mark the message that
// ended the previous request, byte-identical to how that request sent it.
// uncached renders a native message without its breakpoint. Breakpoints move
// between requests and a string is the same prompt as a single text block;
// neither changes the cached prefix.
func uncached(raw json.RawMessage) string {
	var message map[string]any
	_ = json.Unmarshal(raw, &message)
	if text, ok := message["content"].(string); ok {
		message["content"] = []any{map[string]any{"type": "text", "text": text}}
	}
	for _, block := range message["content"].([]any) {
		delete(block.(map[string]any), "cache_control")
	}
	encoded, _ := json.Marshal(message)
	return string(encoded)
}

func TestAnthropicNativeLongToolLoopKeepsReadingCache(t *testing.T) {
	model := json.RawMessage(`{"id":"claude-test","api":"anthropic-messages","provider":"anthropic","baseUrl":"https://api.anthropic.com","maxTokens":1000,"input":["text"],"cost":{}}`)
	tools := []Tool{{Name: "read", Description: "read", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}
	messages := []Message{{Role: "user", Content: TextContent("start"), Timestamp: 1}}
	turn := func(n int) {
		blocks := []ContentBlock{}
		for i := range 25 {
			blocks = append(blocks, ContentBlock{Type: "toolCall", ID: fmt.Sprintf("t%d_%d", n, i), Name: "read", Arguments: json.RawMessage(`{}`)})
		}
		messages = append(messages, Message{Role: "assistant", API: "anthropic-messages", Provider: "anthropic", Model: "claude-test", StopReason: "toolUse", Content: BlockContent(blocks...), Timestamp: int64(n*100 + 2)})
		for i := range 25 {
			messages = append(messages, Message{Role: "toolResult", ToolCallID: fmt.Sprintf("t%d_%d", n, i), ToolName: "read", Content: BlockContent(ContentBlock{Type: "text", Text: "ok"}), Timestamp: int64(n*100 + 3 + i)})
		}
	}
	build := func(oauth bool, options string) json.RawMessage {
		params, err := BuildAnthropicParams(model, ResolveTranscript(NormalizeContext(Context{SystemPrompt: "stable system", Tools: tools, Messages: messages}), false), oauth, json.RawMessage(options))
		if err != nil {
			t.Fatal(err)
		}
		return params
	}
	turn(1)
	previous := build(true, `{}`)
	var before struct{ Messages []json.RawMessage }
	_ = json.Unmarshal(previous, &before)
	turn(2)
	current := build(true, `{}`)
	if !bytes.Equal(current, build(true, `{}`)) {
		t.Fatal("request bytes are not deterministic")
	}
	count, marked, system := nativeBreakpoints(t, current)
	if count > 4 || system != 1 {
		t.Fatalf("breakpoints=%d system=%d, want at most Anthropic's 4 and one closing the stable prefix", count, system)
	}
	var after struct{ Messages []json.RawMessage }
	_ = json.Unmarshal(current, &after)
	last := len(before.Messages) - 1
	if len(marked) != 2 || marked[0] != last || marked[1] != len(after.Messages)-1 {
		t.Fatalf("message breakpoints at %v, want the previous request's end (%d) and the newest message (%d)", marked, last, len(after.Messages)-1)
	}
	if !strings.Contains(string(after.Messages[last]), "cache_control") {
		t.Fatal("previous write point lost its breakpoint")
	}
	for i := 0; i <= last; i++ {
		if uncached(before.Messages[i]) != uncached(after.Messages[i]) {
			t.Fatalf("message %d changed between requests, so the previous write cannot be read:\n%s\n%s", i, before.Messages[i], after.Messages[i])
		}
	}
	// Opting out of caching places no breakpoint at all.
	if count, _, _ := nativeBreakpoints(t, build(false, `{"cacheRetention":"none"}`)); count != 0 {
		t.Fatalf("cacheRetention none still placed %d breakpoints", count)
	}
	// A single-message request has no previous turn to read.
	messages = messages[:1]
	if _, marked, _ := nativeBreakpoints(t, build(false, `{}`)); len(marked) != 1 || marked[0] != 0 {
		t.Fatalf("first request breakpoints %v", marked)
	}
}

// A provider without mid-conversation system messages used to fold late
// system text into the head of the system prompt, so every late update
// rewrote the cached prefix. Late unnamed text is now a reminder at its
// place in the conversation; the system prompt stays byte-identical.
func TestCollapsedLateSystemTextKeepsCachedPrefix(t *testing.T) {
	model := json.RawMessage(`{"id":"claude-test","api":"anthropic-messages","provider":"anthropic","baseUrl":"https://api.anthropic.com","maxTokens":1000,"input":["text"],"cost":{}}`)
	tools := []Tool{{Name: "read", Description: "read", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}
	messages := []Message{
		{Role: "user", Content: TextContent("start"), Timestamp: 1},
		{Role: "assistant", API: "anthropic-messages", Provider: "anthropic", Model: "claude-test", StopReason: "toolUse", Content: BlockContent(ContentBlock{Type: "toolCall", ID: "t1", Name: "read", Arguments: json.RawMessage(`{}`)}), Timestamp: 2},
		{Role: "system", Content: TextContent("todo: A"), Timestamp: 3},
		{Role: "toolResult", ToolCallID: "t1", ToolName: "read", Content: BlockContent(ContentBlock{Type: "text", Text: "ok"}), Timestamp: 4},
		{Role: "system", Content: TextContent(""), Timestamp: 5},
	}
	build := func() (body struct {
		System   []json.RawMessage
		Messages []json.RawMessage
	}) {
		params, err := BuildAnthropicParams(model, ResolveTranscript(NormalizeContext(Context{SystemPrompt: "stable system", Tools: tools, Messages: messages}), false), false, json.RawMessage(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		_ = json.Unmarshal(params, &body)
		return body
	}
	before := build()
	// The reminder waits for the tool results, so tool_use stays answered by
	// the next message; empty late system text adds nothing.
	if len(before.Messages) != 4 || !strings.Contains(string(before.Messages[2]), "tool_result") || !strings.Contains(string(before.Messages[3]), `\u003csystem-reminder\u003e\ntodo: A\n\u003c/system-reminder\u003e`) {
		t.Fatalf("collapsed messages:\n%s", before.Messages)
	}
	// A reminder is volatile, so the breakpoint stays on the tool result.
	if strings.Contains(string(before.Messages[3]), "cache_control") || !strings.Contains(string(before.Messages[2]), "cache_control") {
		t.Fatalf("reminder anchored the cache:\n%s", before.Messages)
	}
	messages = append(messages,
		Message{Role: "assistant", API: "anthropic-messages", Provider: "anthropic", Model: "claude-test", StopReason: "stop", Content: BlockContent(ContentBlock{Type: "text", Text: "done"}), Timestamp: 6},
		Message{Role: "user", Content: TextContent("next"), Timestamp: 7},
		Message{Role: "system", Content: TextContent("todo: B"), Timestamp: 8},
	)
	after := build()
	if len(before.System) != 1 || len(after.System) != 1 || string(before.System[0]) != string(after.System[0]) || strings.Contains(string(after.System[0]), "todo") {
		t.Fatalf("system prompt changed:\n%s\n%s", before.System, after.System)
	}
	for i := range before.Messages {
		if uncached(before.Messages[i]) != uncached(after.Messages[i]) {
			t.Fatalf("message %d changed between requests:\n%s\n%s", i, before.Messages[i], after.Messages[i])
		}
	}
	// Previous write point (the tool result) and the newest stable message.
	marked := []int{}
	for i, message := range after.Messages {
		if strings.Contains(string(message), "cache_control") {
			marked = append(marked, i)
		}
	}
	if len(after.Messages) != 7 || fmt.Sprint(marked) != "[2 5]" {
		t.Fatalf("breakpoints %v over:\n%s", marked, after.Messages)
	}
}
