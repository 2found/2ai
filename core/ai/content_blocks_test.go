package ai

import (
	"encoding/json"
	"testing"
)

func TestStreamSnapshotPreservesContentListGraph(t *testing.T) {
	block := &ContentBlock{Type: "text", Text: "source"}
	partial := &Message{Role: "assistant", Content: BlockReferences(block, nil, NullContentBlock())}
	other := *partial
	source := AssistantMessageEvent{Partial: partial, Message: &other}
	snapshot, err := NewAssistantMessageEventStream().SnapshotEvent(source)
	if err != nil {
		t.Fatal(err)
	}
	list := snapshot.Partial.Content.Blocks
	if list == partial.Content.Blocks || list != snapshot.Message.Content.Blocks {
		t.Fatal("snapshot lost content-list sharing or retained the source list")
	}
	if list.Get(0) == block || list.Get(1) != nil || !list.Get(2).IsNull() {
		t.Fatal("snapshot lost block isolation, holes, or explicit null")
	}
	list.SetLength(0)
	list.Append(&ContentBlock{Type: "text", Text: "snapshot"})
	if snapshot.Message.Content.Blocks.Len() != 1 || partial.Content.Blocks.Len() != 3 {
		t.Fatal("snapshot growth did not follow the detached list graph")
	}
	other.Content = BlockReferences(block, nil, NullContentBlock())
	snapshot, err = NewAssistantMessageEventStream().SnapshotEvent(source)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Partial.Content.Blocks == snapshot.Message.Content.Blocks {
		t.Fatal("snapshot merged equal but distinct content lists")
	}
}

func TestPiMessagesAdapterRetainsContentListIdentity(t *testing.T) {
	adapter := newPiMessagesTypedAdapter()
	text := func(value string) *Object {
		return NewObject(Property{Name: "type", Value: "text"}, Property{Name: "text", Value: value})
	}
	content := NewArray(text("a"))
	source := NewObject(Property{Name: "role", Value: "assistant"}, Property{Name: "content", Value: content}, Property{Name: "timestamp", Value: 1})
	message, err := adapter.message(source)
	if err != nil {
		t.Fatal(err)
	}
	retained := message.Content
	content.Set(3, text("b"))
	message, err = adapter.message(source)
	if err != nil {
		t.Fatal(err)
	}
	if retained.Blocks != message.Content.Blocks || retained.Blocks.Len() != 4 || retained.Blocks.Get(3).Text != "b" || retained.Blocks.Get(1) != nil {
		t.Fatal("typed adapter replaced a live content list or filled its holes")
	}
	source.Set("content", NewArray(text("replacement")))
	message, err = adapter.message(source)
	if err != nil {
		t.Fatal(err)
	}
	if retained.Blocks == message.Content.Blocks || retained.Blocks.Len() != 4 || message.Content.Blocks.Len() != 1 {
		t.Fatal("typed adapter mutated the old list when content was reassigned")
	}
}

func TestContentListNullAndHoleRoundTrip(t *testing.T) {
	live := BlockReferences(nil, NullContentBlock(), &ContentBlock{Type: "text", Text: "ok"})
	raw, err := json.Marshal(live)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `[null,null,{"text":"ok","type":"text"}]` {
		t.Fatalf("content: %s", raw)
	}
	var decoded MessageContent
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if live.Blocks.Get(0) != nil || !decoded.Blocks.Get(0).IsNull() || !decoded.Blocks.Get(1).IsNull() {
		t.Fatal("live hole and decoded null were conflated")
	}
}
