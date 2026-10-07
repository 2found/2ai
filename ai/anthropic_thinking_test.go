package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func TestAnthropicEncodeReplaysOnlyValidSameScopeReasoning(t *testing.T) {
	p := NewAnthropicProvider("key", "https://gateway.example")
	model := "claude-sonnet-test"
	scope := p.replayScope(model)
	body := p.encode(protocol.ChatRequest{
		Model: model,
		Messages: []protocol.Message{{
			Role: protocol.RoleAssistant,
			ReasoningBlocks: []protocol.ReasoningBlock{
				{Type: protocol.ReasoningBlockThinking, Text: "signed thought", Signature: "sig", ReplayScope: scope},
				{Type: protocol.ReasoningBlockThinking, Text: "unsigned", ReplayScope: scope},
				{Type: protocol.ReasoningBlockThinking, Text: "wrong endpoint", Signature: "sig-2", ReplayScope: "anthropic:foreign"},
				{Type: protocol.ReasoningBlockRedacted, Data: "encrypted", ReplayScope: scope},
			},
			Content:   "answer",
			ToolCalls: []protocol.ToolCall{{ID: "call-1", Name: "lookup", Arguments: `{"q":"x"}`}},
		}},
	})

	if len(body.Messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(body.Messages))
	}
	blocks := body.Messages[0].Content
	wantTypes := []string{"thinking", "redacted_thinking", "text", "tool_use"}
	if len(blocks) != len(wantTypes) {
		t.Fatalf("blocks = %+v, want types %v", blocks, wantTypes)
	}
	for i, want := range wantTypes {
		if blocks[i].Type != want {
			t.Fatalf("block %d type = %q, want %q", i, blocks[i].Type, want)
		}
	}
	if blocks[0].Thinking != "signed thought" || blocks[0].Signature != "sig" {
		t.Fatalf("signed thinking changed: %+v", blocks[0])
	}
	if blocks[1].Data != "encrypted" {
		t.Fatalf("redacted payload changed: %+v", blocks[1])
	}
}

func TestAnthropicChatCapturesSignedAndRedactedReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"content":[
				{"type":"thinking","thinking":"consider this","signature":"signature-1"},
				{"type":"thinking","thinking":"must not persist without a signature"},
				{"type":"redacted_thinking","data":"encrypted-1"},
				{"type":"text","text":"done"},
				{"type":"tool_use","id":"call-1","name":"lookup","input":{"q":"x"}}
			],
			"stop_reason":"tool_use",
			"usage":{"input_tokens":7,"output_tokens":5}
		}`)
	}))
	defer srv.Close()

	p := NewAnthropicProvider("key", srv.URL)
	p.HTTP = srv.Client()
	resp, err := p.Chat(context.Background(), protocol.ChatRequest{Model: "claude-test"})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Message.Content != "done" || len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("visible response changed: %+v", resp.Message)
	}
	want := []protocol.ReasoningBlock{
		{Type: protocol.ReasoningBlockThinking, Text: "consider this", Signature: "signature-1", ReplayScope: p.replayScope("claude-test")},
		{Type: protocol.ReasoningBlockRedacted, Data: "encrypted-1", ReplayScope: p.replayScope("claude-test")},
	}
	if !reasoningBlocksEqual(resp.Message.ReasoningBlocks, want) {
		t.Fatalf("reasoning blocks = %+v, want %+v", resp.Message.ReasoningBlocks, want)
	}
}

func TestAnthropicStreamCapturesCompleteReasoningBlocks(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		frames := []string{
			`{"type":"message_start","message":{"usage":{"input_tokens":9,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"plan "}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"carefully"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-stream"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"encrypted-stream"}}`,
			`{"type":"content_block_stop","index":1}`,
			`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"answer"}}`,
			`{"type":"content_block_stop","index":2}`,
			`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"call-1","name":"lookup"}}`,
			`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}}`,
			`{"type":"content_block_stop","index":3}`,
			`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":6}}`,
			`{"type":"message_stop"}`,
		}
		for _, frame := range frames {
			_, _ = io.WriteString(w, "data: "+frame+"\n\n")
		}
	}))
	defer srv.Close()

	p := NewAnthropicProvider("key", srv.URL)
	p.StreamHTTP = srv.Client()
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "claude-test"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var content string
	var reasoning []protocol.ReasoningBlock
	var calls []protocol.ToolCall
	var terminal protocol.ChatDelta
	for delta := range ch {
		content += delta.ContentDelta
		if delta.ReasoningBlock != nil {
			reasoning = append(reasoning, *delta.ReasoningBlock)
		}
		if delta.ToolCall != nil {
			calls = append(calls, *delta.ToolCall)
		}
		if delta.Done {
			terminal = delta
		}
	}
	wantReasoning := []protocol.ReasoningBlock{
		{Type: protocol.ReasoningBlockThinking, Text: "plan carefully", Signature: "sig-stream", ReplayScope: p.replayScope("claude-test")},
		{Type: protocol.ReasoningBlockRedacted, Data: "encrypted-stream", ReplayScope: p.replayScope("claude-test")},
	}
	if content != "answer" || !reasoningBlocksEqual(reasoning, wantReasoning) {
		t.Fatalf("stream content=%q reasoning=%+v", content, reasoning)
	}
	if len(calls) != 1 || calls[0].Arguments != `{"q":"x"}` {
		t.Fatalf("stream calls = %+v", calls)
	}
	if !terminal.Done || terminal.StopReason != "tool_use" || terminal.Usage.InputTokens != 9 || terminal.Usage.OutputTokens != 6 {
		t.Fatalf("terminal delta = %+v", terminal)
	}
}

func TestAnthropicReasoningBoundsDropWholeBlock(t *testing.T) {
	p := NewAnthropicProvider("key", "https://gateway.example")
	model := "claude-test"
	body := p.encode(protocol.ChatRequest{Model: model, Messages: []protocol.Message{{
		Role: protocol.RoleAssistant,
		ReasoningBlocks: []protocol.ReasoningBlock{{
			Type: protocol.ReasoningBlockThinking, Text: strings.Repeat("x", maxAnthropicReasoningBlockBytes),
			Signature: "too-large", ReplayScope: p.replayScope(model),
		}},
		Content: "safe answer",
	}}})
	if got := body.Messages[0].Content; len(got) != 1 || got[0].Type != "text" || got[0].Text != "safe answer" {
		t.Fatalf("oversized reasoning was truncated/replayed instead of dropped: %+v", got)
	}
}

func reasoningBlocksEqual(a, b []protocol.ReasoningBlock) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestAnthropicReplayScopeDoesNotExposeEndpoint(t *testing.T) {
	p := NewAnthropicProvider("key", "https://secret.internal.example")
	scope := p.replayScope("claude-test")
	if strings.Contains(scope, p.BaseURL) || !strings.HasPrefix(scope, "anthropic:") {
		t.Fatalf("unsafe replay scope %q", scope)
	}
	if _, err := json.Marshal(protocol.ReasoningBlock{ReplayScope: scope}); err != nil {
		t.Fatalf("scope is not persistable: %v", err)
	}
}
