package ai

import (
	"context"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func TestContextRecoveryIsBoundedAndStopsAfterVisibleOrAmbiguousOutput(t *testing.T) {
	for _, mode := range []string{"rejected", "visible", "aborted", "preparation", "malformed", "cancelled"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls, recoveries := 0, 0
			failure := &protocol.ProviderError{Status: 400, Message: "context_length_exceeded"}
			message := &Message{Role: "assistant", StopReason: "error"}
			runner := nativeRungAttempts{policy: protocol.RetryPolicy{MaxAttempts: 1},
				commit: func(context.Context) error { return nil },
				open: func(ctx context.Context, _ int) (*AssistantMessageEventStream, error) {
					calls++
					if mode == "preparation" {
						return nil, &PreparationError{Cause: failure}
					}
					if mode == "malformed" {
						return attemptFixture()(ctx)
					}
					events := []AssistantMessageEvent{{Type: "start", Partial: message}}
					if mode == "visible" {
						events = append(events, AssistantMessageEvent{Type: "text_delta", Delta: "x", Partial: message})
					}
					if mode == "aborted" {
						message.StopReason = "aborted"
					}
					return attemptFixture(append(events, AssistantMessageEvent{Type: "error", Reason: message.StopReason, Error: message})...)(ctx)
				}, failure: func(int) error { return failure },
				observe: func(context.Context, FallbackAttempt) error {
					if mode == "cancelled" {
						cancel()
					}
					return nil
				},
				recover: func(context.Context, FallbackAttempt) (bool, error) { recoveries++; return true, nil },
			}
			_, _ = runner.run(ctx, NewAssistantMessageEventStream())
			wantCalls, wantRecovery := 1, 0
			if mode == "rejected" {
				wantCalls, wantRecovery = 2, 1
			}
			if calls != wantCalls || recoveries != wantRecovery {
				t.Fatalf("calls=%d recoveries=%d", calls, recoveries)
			}
		})
	}
}

func TestContextOverflowDoesNotClassifyTransportOrPayloadErrors(t *testing.T) {
	for _, status := range []int{400, 413, 429, 503} {
		got := IsContextOverflow(&protocol.ProviderError{Status: status, Message: "context_length_exceeded"})
		if got != (status == 400 || status == 413) {
			t.Fatal(status, got)
		}
	}
	if IsContextOverflow(&protocol.ProviderError{Status: 413, Message: "request body too large"}) {
		t.Fatal("generic payload overflow triggered compaction")
	}
}
