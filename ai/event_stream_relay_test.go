package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

func TestAssistantStreamRelayPreservesLivePayloads(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	relay := NewAssistantMessageEventStream()
	shared := WithAssistantStreamSynchronization(ctx, relay)
	rejected := NewAssistantMessageEventStreamFor(shared)
	rejected.Push(AssistantMessageEvent{Type: "error", Error: &Message{Role: "assistant", StopReason: "error"}})
	rejected.End()
	// A rejected attempt must settle only its own queue and result.
	probe, stop := context.WithCancel(ctx)
	stop()
	if _, err := relay.Result(probe); err != context.Canceled {
		t.Fatal("attempt terminal leaked into relay result", err)
	}
	source := NewAssistantMessageEventStreamFor(shared)
	message := &Message{Role: "assistant", Content: BlockContent(ContentBlock{Type: "text", Text: ""}), StopReason: "stop"}
	go func() {
		defer source.End()
		source.Synchronize(func() { source.Push(AssistantMessageEvent{Type: "start", Partial: message}) })
		for i := 0; i < 1000; i++ {
			source.Synchronize(func() {
				message.Content.Blocks.Get(0).Text = fmt.Sprint(i)
				source.Push(AssistantMessageEvent{Type: "text_delta", Delta: "x", Partial: message})
			})
		}
		source.Synchronize(func() { source.Push(AssistantMessageEvent{Type: "done", Reason: "stop", Message: message}) })
	}()
	relayDone := make(chan error, 1)
	go func() {
		defer relay.End()
		for {
			event, ok, err := source.Next(ctx)
			if err != nil || !ok {
				relayDone <- err
				return
			}
			relay.Push(event)
		}
	}()
	var first *Message
	count := 0
	for {
		event, ok, err := relay.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		if event.Type == "start" {
			first = event.Partial
		}
		if _, err = relay.SnapshotEvent(event); err != nil {
			t.Fatal(err)
		}
		count++
	}
	if err := <-relayDone; err != nil {
		t.Fatal(err)
	}
	if count != 1002 || first != message {
		t.Fatalf("relay copied/lost events: %d", count)
	}
	result, err := relay.Result(ctx)
	if err != nil || result != message {
		t.Fatal("relay copied/lost terminal pointer", err)
	}
	snapshot, err := relay.SnapshotResult(ctx)
	if err != nil || snapshot.Content.Blocks.Get(0).Text != "999" {
		t.Fatal("relay snapshot lost final mutation", err)
	}
	independent := NewAssistantMessageEventStreamFor(context.Background())
	if independent.payloadMu == relay.payloadMu {
		t.Fatal("unrelated streams share a global lock")
	}
}

func TestNativeProvidersInheritRelaySynchronization(t *testing.T) {
	for _, tc := range []struct {
		name  string
		start func(context.Context, json.RawMessage, TranscriptContext, OpenAICompletionsStreamOptions) *AssistantMessageEventStream
	}{
		{"completions", StreamOpenAICompletions},
		{"responses", StreamOpenAIResponses},
		{"anthropic", StreamAnthropic},
		{"codex-sse", StreamCodexResponsesSSE},
		{"codex-combined", StreamCodexResponses},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			relay := NewAssistantMessageEventStream()
			// Malformed model fails preparation without making any network request.
			stream := tc.start(WithAssistantStreamSynchronization(ctx, relay), json.RawMessage(`!`), TranscriptContext{}, OpenAICompletionsStreamOptions{})
			if stream.payloadMu != relay.payloadMu {
				t.Fatal("provider did not inherit relay synchronization")
			}
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			result, err := stream.SnapshotResult(ctx)
			if err != nil || result == nil || result.StopReason != "error" {
				t.Fatal("provider failure did not settle independently", err)
			}
		})
	}
}
