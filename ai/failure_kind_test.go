package ai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func TestNativeFailureKindUsesStructuralMetadata(t *testing.T) {
	if boundedFailureKind("SECRET_NATIVE_ERROR") != "unknown" {
		t.Fatal("private diagnostic accepted")
	}
	for _, tc := range []struct {
		cause    error
		callback bool
		want     string
	}{
		{nil, false, "none"},
		{&protocol.ProviderError{Status: 401, Message: "SECRET"}, false, "provider_auth"},
		{&codexHTTPError{Status: 429, Message: "SECRET"}, false, "provider_rate_limit"},
		{&AnthropicClientError{Status: 503, Message: "SECRET"}, false, "provider_unavailable"},
		{&codexProtocolError{Message: "SECRET", Payload: "SECRET"}, false, "provider_protocol"},
		{&json.SyntaxError{Offset: 100}, false, "provider_decode"},
		{&protocol.ProviderError{Status: 503, Message: "SECRET"}, true, "host_callback"},
		{&codexProviderCallbackError{cause: &protocol.ProviderError{Status: 503}}, false, "host_callback"},
		{&PreparationError{Cause: &protocol.ProviderError{Status: 503}}, false, "host_preparation"},
		{context.DeadlineExceeded, false, "request_timeout"},
		{context.Canceled, false, "request_cancelled"},
		{errors.New("401 unauthorized SECRET malformed JSON"), false, "unknown"},
	} {
		if got := nativeFailureKind(tc.cause, tc.callback); got != tc.want {
			t.Fatalf("kind %q want %q", got, tc.want)
		}
		ctx, capture := WithNativeProviderFailure(context.Background())
		recordNativeFailure(ctx, "test", tc.cause, tc.callback)
		if got := capture.Kind(); got != tc.want {
			t.Fatalf("captured kind %q want %q", got, tc.want)
		}
		recordNativeFailure(ctx, "test", nil, false)
		if capture.Kind() != "none" {
			t.Fatal("diagnostic leaked into next attempt")
		}
	}
}

func TestFailureKindSurvivesTerminalFormattingWithoutReplay(t *testing.T) {
	for _, tc := range []struct {
		cause error
		want  string
	}{
		{&codexProtocolError{Message: "SECRET", Payload: "SECRET"}, "provider_protocol"},
		{&json.SyntaxError{Offset: 100}, "provider_decode"},
		{errors.New("SECRET 429 malformed JSON"), "unknown"},
	} {
		out := NewAssistantMessageEventStream()
		calls := 0
		message := &Message{Role: "assistant", StopReason: "error"}
		runner := nativeRungAttempts{
			policy: protocol.RetryPolicy{MaxAttempts: 3},
			open: func(ctx context.Context, _ int) (*AssistantMessageEventStream, error) {
				calls++
				recordNativeFailure(ctx, "codex", tc.cause, false)
				return attemptFixture(AssistantMessageEvent{Type: "start", Partial: message}, AssistantMessageEvent{Type: "text_delta", Delta: "partial", Partial: message}, AssistantMessageEvent{Type: "error", Error: message})(ctx)
			},
			commit: func(context.Context) error { return nil },
			observe: func(_ context.Context, attempt FallbackAttempt) error {
				if attempt.FailureKind != tc.want || !attempt.Outcome.Committed {
					t.Fatalf("lost diagnostic/replay fence: %+v", attempt)
				}
				return nil
			},
		}
		if _, err := runner.run(context.Background(), out); err != nil || calls != 1 {
			t.Fatalf("visible failure replayed: calls=%d err=%v", calls, err)
		}
	}
}
