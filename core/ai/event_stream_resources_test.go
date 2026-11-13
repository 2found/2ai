package ai

import (
	"context"
	"testing"
)

func BenchmarkEventStreamToken(b *testing.B) {
	s := NewAssistantMessageEventStream()
	defer s.End()
	event := AssistantMessageEvent{Type: "text_delta", Delta: "token"}
	b.ReportAllocs()
	for b.Loop() {
		s.Push(event)
		if _, ok, err := s.Next(context.Background()); !ok || err != nil {
			b.Fatal(ok, err)
		}
	}
}

func TestEventStreamReusesSmallBuffersAndReleasesBursts(t *testing.T) {
	s := NewAssistantMessageEventStream()
	event := AssistantMessageEvent{Type: "text_delta", Partial: &Message{Role: "assistant"}}
	for range 100 {
		s.Push(event)
		got, ok, err := s.Next(context.Background())
		if err != nil || !ok || got.Partial != event.Partial {
			t.Fatal("event identity changed")
		}
		if cap(s.queue) != 1 {
			t.Fatalf("single-token buffer not reused: %d", cap(s.queue))
		}
		if s.queue[:cap(s.queue)][0].Partial != nil {
			t.Fatal("consumed payload retained")
		}
	}
	for range 4096 {
		s.Push(event)
	}
	for range 4096 {
		if _, ok, err := s.Next(context.Background()); err != nil || !ok {
			t.Fatal(ok, err)
		}
	}
	if cap(s.queue) != 0 {
		t.Fatal("large burst buffer retained")
	}
	s.Push(event)
	_, _, _ = s.Next(context.Background())
	s.End()
	if cap(s.queue) != 0 {
		t.Fatal("ended stream retained its scratch buffer")
	}
}
