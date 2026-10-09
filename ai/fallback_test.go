package ai

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

func TestFallbackStreamServerHintsAndBusyMessages(t *testing.T) {
	for _, tc := range []struct {
		name      string
		failure   *protocol.ProviderError
		wantCalls int
		wantWaits int
	}{
		{name: "over-limit hint", failure: &protocol.ProviderError{Status: 429, Message: "rate limit", RetryAfter: time.Minute}, wantCalls: 1},
		{name: "busy wrapper", failure: &protocol.ProviderError{Status: 200, Message: "server_busy"}, wantCalls: 2, wantWaits: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls, waits, fallbacks int
			final := &Message{Role: "assistant", StopReason: "stop", Content: TextContent("fallback")}
			provider := FallbackProvider{Retry: protocol.DefaultRetryPolicy(), Wait: func(context.Context, time.Duration) error { waits++; return nil }}
			provider.Candidates = []FallbackCandidate{
				{Stream: func(ctx context.Context, _ json.RawMessage, _ TranscriptContext, _ map[string]any) (*AssistantMessageEventStream, error) {
					calls++
					if calls == 1 || tc.wantCalls == 1 {
						return nil, tc.failure
					}
					return attemptFixture(AssistantMessageEvent{Type: "done", Message: final})(ctx)
				}},
				{Stream: func(ctx context.Context, _ json.RawMessage, _ TranscriptContext, _ map[string]any) (*AssistantMessageEventStream, error) {
					fallbacks++
					return attemptFixture(AssistantMessageEvent{Type: "done", Message: final})(ctx)
				}},
			}
			stream, err := provider.Stream(context.Background(), nil, TranscriptContext{}, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := stream.Result(context.Background())
			wantFallbacks := 1 - tc.wantWaits
			if err != nil || got != final || calls != tc.wantCalls || waits != tc.wantWaits || fallbacks != wantFallbacks {
				t.Fatal("retry/fallback policy mismatch", got, err, calls, waits, fallbacks)
			}
		})
	}
}

func TestFallbackStreamUsesIndependentCandidatesAndNativeResult(t *testing.T) {
	ctx := context.Background()
	transcript := NormalizeContext(Context{Messages: []Message{{Role: "user", Content: BlockContent(ContentBlock{Type: "text", Text: "original"})}}})
	options := map[string]any{"apiKey": "fixture"}
	failure := &protocol.ProviderError{Status: 503, Message: "busy"}
	final := &Message{Role: "assistant", StopReason: "stop", Content: BlockContent(ContentBlock{Type: "text", Text: "answer"})}
	var calls []string
	provider := FallbackProvider{Retry: protocol.RetryPolicy{MaxAttempts: 2}, Wait: func(context.Context, time.Duration) error { return nil }}
	open := func(ctx context.Context, model json.RawMessage, input TranscriptContext, controls map[string]any) (*AssistantMessageEventStream, error) {
		var selected struct{ ID string }
		_ = json.Unmarshal(model, &selected)
		calls = append(calls, selected.ID)
		if input.Messages()[0].Content.Blocks.Get(0).Text != "original" || controls["apiKey"] != "fixture" {
			t.Error("previous attempt mutated next request")
		}
		input.Messages()[0].Content.Blocks.Get(0).Text = "mutated"
		controls["apiKey"] = "mutated"
		model[0] = '!'
		if selected.ID == "primary" {
			return nil, failure
		}
		return attemptFixture(AssistantMessageEvent{Type: "start", Partial: final}, AssistantMessageEvent{Type: "text_delta", Delta: "answer", Partial: final}, AssistantMessageEvent{Type: "done", Reason: "stop", Message: final})(ctx)
	}
	provider.Candidates = []FallbackCandidate{{Model: json.RawMessage(`{"id":"primary"}`), Stream: open}, {Model: json.RawMessage(`{"id":"fallback"}`), Stream: open}}
	stream, err := provider.Stream(ctx, nil, transcript, options)
	if err != nil {
		t.Fatal(err)
	}
	for _, typ := range []string{"start", "text_delta", "done"} {
		event, ok, err := stream.Next(ctx)
		if err != nil || !ok || event.Type != typ {
			t.Fatalf("event=%s ok=%v err=%v", event.Type, ok, err)
		}
	}
	if _, ok, err := stream.Next(ctx); err != nil || ok {
		t.Fatal(ok, err)
	}
	got, err := stream.Result(ctx)
	if err != nil || got != final || !reflect.DeepEqual(calls, []string{"primary", "primary", "fallback"}) {
		t.Fatal(calls, got, err)
	}
	if options["apiKey"] != "fixture" || transcript.Messages()[0].Content.Blocks.Get(0).Text != "original" || !json.Valid(provider.Candidates[0].Model) {
		t.Fatal("caller request changed")
	}
}

func TestFallbackStreamDoesNotEscalateAfterOutputOrHostFailure(t *testing.T) {
	for _, mode := range []string{"visible", "aborted", "preparation", "protocol", "panic"} {
		t.Run(mode, func(t *testing.T) {
			ctx := context.Background()
			failure := &protocol.ProviderError{Status: 503, Message: "busy"}
			terminal := &Message{Role: "assistant", StopReason: "error"}
			calls := 0
			provider := FallbackProvider{Retry: protocol.RetryPolicy{MaxAttempts: 2}, Wait: func(context.Context, time.Duration) error { t.Error("unexpected retry"); return nil }}
			provider.Candidates = []FallbackCandidate{{Stream: func(ctx context.Context, _ json.RawMessage, _ TranscriptContext, _ map[string]any) (*AssistantMessageEventStream, error) {
				calls++
				switch mode {
				case "preparation":
					return nil, &PreparationError{Cause: failure}
				case "protocol":
					return attemptFixture()(ctx)
				case "panic":
					panic("provider panic")
				case "aborted":
					terminal.StopReason = "aborted"
					return attemptFixture(AssistantMessageEvent{Type: "error", Reason: "aborted", Error: terminal})(ctx)
				default:
					return attemptFixture(AssistantMessageEvent{Type: "text_delta", Delta: "visible", Partial: terminal}, AssistantMessageEvent{Type: "error", Error: terminal})(ctx)
				}
			}}, {Stream: func(context.Context, json.RawMessage, TranscriptContext, map[string]any) (*AssistantMessageEventStream, error) {
				t.Error("unexpected fallback")
				return nil, failure
			}}}
			stream, err := provider.Stream(ctx, nil, TranscriptContext{}, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := stream.Result(ctx)
			if mode == "visible" || mode == "aborted" {
				if err != nil || got != terminal {
					t.Fatal(got, err)
				}
			} else if err == nil {
				t.Fatal("host/protocol failure swallowed")
			}
			if mode == "preparation" && !errors.Is(err, failure) {
				t.Fatal("preparation cause lost", err)
			}
			if calls != 1 {
				t.Fatal(calls)
			}
		})
	}
}

func TestFallbackStreamSharesParentPayloadSynchronization(t *testing.T) {
	parent := NewAssistantMessageEventStream()
	ctx := WithAssistantStreamSynchronization(context.Background(), parent)
	provider := FallbackProvider{Candidates: []FallbackCandidate{{Stream: func(ctx context.Context, _ json.RawMessage, _ TranscriptContext, _ map[string]any) (*AssistantMessageEventStream, error) {
		source := NewAssistantMessageEventStreamFor(ctx)
		if source.payloadMu != parent.payloadMu {
			t.Error("nested provider detached live payload synchronization")
		}
		source.Push(AssistantMessageEvent{Type: "done", Message: &Message{Role: "assistant", StopReason: "stop"}})
		source.End()
		return source, nil
	}}}}
	stream, err := provider.Stream(ctx, nil, TranscriptContext{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if stream.payloadMu != parent.payloadMu {
		t.Fatal("fallback detached relay synchronization")
	}
	if _, err = stream.Result(ctx); err != nil {
		t.Fatal(err)
	}
}
