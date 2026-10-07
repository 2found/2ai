package agentcore

import (
	"context"
	"strings"
	"testing"
)

func TestReplayProviderEmptyRecording(t *testing.T) {
	p := NewReplayProvider()
	_, err := p.Chat(context.Background(), ChatRequest{})
	if err == nil || !strings.Contains(err.Error(), "empty recording") {
		t.Fatalf("empty recording: %v", err)
	}
}

func TestReplayProviderExtraCall(t *testing.T) {
	p := NewReplayProvider(TurnRecord{
		Messages:   []Message{{Role: RoleUser, Content: "hi"}},
		Response:   "hello",
		StopReason: "stop",
	})
	if _, err := p.Chat(context.Background(), ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}); err != nil {
		t.Fatalf("first call: %v", err)
	}
	_, err := p.Chat(context.Background(), ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), "extra call") {
		t.Fatalf("extra call: %v", err)
	}
}

func TestReplayProviderReturnsRecordedError(t *testing.T) {
	p := NewReplayProvider(TurnRecord{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Response: "partial",
		Error:    "provider exploded",
	})
	resp, err := p.Chat(context.Background(), ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err == nil || err.Error() != "provider exploded" {
		t.Fatalf("error: %v", err)
	}
	if resp.Message.Content != "partial" {
		t.Fatalf("recorded response dropped: %q", resp.Message.Content)
	}
}

func TestReplayProviderReturnsReasoningBlocksInChatAndStream(t *testing.T) {
	block := ReasoningBlock{
		Type: ReasoningBlockThinking, Text: "opaque thought", Signature: "sig", ReplayScope: "anthropic:scope",
	}
	record := TurnRecord{
		Messages:        []Message{{Role: RoleUser, Content: "hi"}},
		Response:        "hello",
		ReasoningBlocks: []ReasoningBlock{block},
	}
	req := ChatRequest{Messages: record.Messages}
	chat := NewReplayProvider(record)
	resp, err := chat.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(resp.Message.ReasoningBlocks) != 1 || resp.Message.ReasoningBlocks[0] != block {
		t.Fatalf("chat reasoning blocks = %+v", resp.Message.ReasoningBlocks)
	}

	stream := NewReplayProvider(record)
	ch, err := stream.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var got []ReasoningBlock
	for delta := range ch {
		if delta.ReasoningBlock != nil {
			got = append(got, *delta.ReasoningBlock)
		}
	}
	if len(got) != 1 || got[0] != block {
		t.Fatalf("stream reasoning blocks = %+v", got)
	}
}

func TestReplayProviderRejectsDrift(t *testing.T) {
	p := NewReplayProvider(TurnRecord{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Response: "hello",
	})
	_, err := p.Chat(context.Background(), ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hello"}}})
	if err == nil || !strings.Contains(err.Error(), "drifted") {
		t.Fatalf("drift: %v", err)
	}
}

func TestReplayProviderRejectsRichContentDrift(t *testing.T) {
	recorded := Message{Role: RoleTool, ToolCallID: "c1", Content: "plot", ContentParts: []ContentPart{{
		Type: ContentPartImage, MIMEType: "image/png", Data: "original",
	}}}
	p := NewReplayProvider(TurnRecord{Messages: []Message{recorded}, Response: "ok"})
	changed := cloneSessionMessage(recorded)
	changed.ContentParts[0].Data = "changed"
	_, err := p.Chat(context.Background(), ChatRequest{Messages: []Message{changed}})
	if err == nil || !strings.Contains(err.Error(), "drifted") {
		t.Fatalf("rich content drift was not detected: %v", err)
	}
}

