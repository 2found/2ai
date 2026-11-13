package ai

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func attemptFixture(events ...AssistantMessageEvent) func(context.Context) (*AssistantMessageEventStream, error) {
	return func(ctx context.Context) (*AssistantMessageEventStream, error) {
		source := NewAssistantMessageEventStreamFor(ctx)
		for _, event := range events {
			source.Push(event)
		}
		source.End()
		return source, nil
	}
}
func assertAttemptEmpty(t *testing.T, out *AssistantMessageEventStream) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, ok, err := out.Next(ctx); ok || !errors.Is(err, context.Canceled) {
		t.Fatal("uncommitted event escaped", ok, err)
	}
}

func TestNativeAttemptRelayRetryRetainsSuccessfulPointers(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out := NewAssistantMessageEventStream()
	rejected := &Message{Role: "assistant", StopReason: "error"}
	successful := &Message{Role: "assistant", StopReason: "stop", Content: BlockContent(ContentBlock{Type: "text", Text: "answer"})}
	commits := 0
	commit := func(context.Context) error { assertAttemptEmpty(t, out); commits++; return nil }
	first, err := relayNativeAttempt(ctx, out, attemptFixture(AssistantMessageEvent{Type: "start", Partial: rejected}, AssistantMessageEvent{Type: "error", Error: rejected}), commit)
	if err != nil || first.Committed || first.Terminal.Error != rejected || commits != 0 {
		t.Fatal("failed attempt lost identity or committed", err)
	}
	assertAttemptEmpty(t, out)
	second, err := relayNativeAttempt(ctx, out, attemptFixture(AssistantMessageEvent{Type: "start", Partial: successful}, AssistantMessageEvent{Type: "text_delta", Delta: "answer", Partial: successful}, AssistantMessageEvent{Type: "done", Reason: "stop", Message: successful}), commit)
	if err != nil || !second.Committed || commits != 1 {
		t.Fatal("successful attempt did not commit once", err)
	}
	second.publish(out)
	out.End()
	count := 0
	for {
		event, ok, err := out.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		count++
		if event.Partial != nil && event.Partial != successful {
			t.Fatal("rejected attempt metadata escaped or pointer copied")
		}
	}
	result, err := out.Result(ctx)
	if err != nil || result != successful || count != 3 {
		t.Fatal("relay lost terminal identity/events", count, err)
	}
}

func TestNativeAttemptRelayVisibleContentStopsReplay(t *testing.T) {
	for _, event := range []AssistantMessageEvent{
		{Type: "text_delta", Delta: "x"}, {Type: "thinking_delta", Delta: "x"}, {Type: "toolcall_delta", Delta: "{"},
		{Type: "toolcall_start"}, {Type: "toolcall_end"}, {Type: "text_end"}, {Type: "thinking_end"}, {Type: "future_non_replayable_event"},
	} {
		t.Run(event.Type, func(t *testing.T) {
			out := NewAssistantMessageEventStream()
			failed := &Message{Role: "assistant", StopReason: "error"}
			commits := 0
			result, err := relayNativeAttempt(context.Background(), out, attemptFixture(AssistantMessageEvent{Type: "start", Partial: failed}, event, AssistantMessageEvent{Type: "error", Error: failed}), func(context.Context) error { commits++; assertAttemptEmpty(t, out); return nil })
			if err != nil || !result.Committed || commits != 1 {
				t.Fatal("content remained replayable", err)
			}
			result.publish(out)
			out.End()
			terminal, err := out.Result(context.Background())
			if err != nil || terminal != failed {
				t.Fatal("failure pointer lost", err)
			}
		})
	}
}

func TestNativeAttemptRelayBuffersOnlyReplaySafeMetadata(t *testing.T) {
	out := NewAssistantMessageEventStream()
	failed := &Message{Role: "assistant", StopReason: "error"}
	events := []AssistantMessageEvent{{Type: "start", Partial: failed}, {Type: "text_start"}, {Type: "thinking_start"}, {Type: "text_delta"}, {Type: "thinking_delta"}, {Type: "toolcall_delta"}, {Type: "error", Error: failed}}
	result, err := relayNativeAttempt(context.Background(), out, attemptFixture(events...), func(context.Context) error { t.Error("metadata committed"); return nil })
	if err != nil || result.Committed || len(result.pending) != 6 {
		t.Fatal("metadata handling changed", err)
	}
	assertAttemptEmpty(t, out)
	// Exhaustion publishes the last failure and its original metadata in order.
	result.publish(out)
	out.End()
	for i := range events {
		event, ok, err := out.Next(context.Background())
		if err != nil || !ok || event.Type != events[i].Type {
			t.Fatal("final failure lost metadata order", i, err)
		}
	}
}

func TestNativeAttemptRelayCommitAndProtocolFailures(t *testing.T) {
	for _, mode := range []string{"commit", "empty", "nil", "open", "cancelled", "after-end"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			out := NewAssistantMessageEventStream()
			message := &Message{Role: "assistant", StopReason: "stop"}
			failure := errors.New(mode)
			open := attemptFixture(AssistantMessageEvent{Type: "start", Partial: message}, AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
			commit := func(context.Context) error { return nil }
			switch mode {
			case "commit":
				commit = func(context.Context) error { return failure }
			case "empty":
				open = attemptFixture()
			case "nil":
				open = func(context.Context) (*AssistantMessageEventStream, error) { return nil, nil }
			case "open":
				open = func(context.Context) (*AssistantMessageEventStream, error) { return nil, failure }
			case "cancelled":
				cancel()
				open = func(context.Context) (*AssistantMessageEventStream, error) {
					t.Error("cancelled attempt admitted")
					return nil, nil
				}
			case "after-end":
				open = func(context.Context) (*AssistantMessageEventStream, error) {
					s := NewAssistantMessageEventStreamWithHooks(AssistantStreamHooks{AfterEnd: func(context.Context) error { return failure }})
					s.Push(AssistantMessageEvent{Type: "done", Message: message})
					s.End()
					return s, nil
				}
			}
			result, err := relayNativeAttempt(ctx, out, open, commit)
			if mode == "open" {
				if err != nil || result.AdmissionError != failure || result.publish(out) != failure {
					t.Fatal("admission failure identity lost", err)
				}
				assertAttemptEmpty(t, out)
				return
			}
			if err == nil || result.Committed {
				t.Fatal("host/protocol failure accepted", err)
			}
			assertAttemptEmpty(t, out)
		})
	}
}

func TestNativeAttemptRelayProgressBeforeProducerSettlement(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out := NewAssistantMessageEventStream()
	release := make(chan struct{})
	var committed atomic.Bool
	done := make(chan error, 1)
	message := &Message{Role: "assistant", StopReason: "stop"}
	go func() {
		result, err := relayNativeAttempt(ctx, out, func(ctx context.Context) (*AssistantMessageEventStream, error) {
			source := NewAssistantMessageEventStreamFor(ctx)
			go func() {
				defer source.End()
				source.Push(AssistantMessageEvent{Type: "text_delta", Delta: "live", Partial: message})
				select {
				case <-release:
				case <-ctx.Done():
					return
				}
				source.Push(AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
			}()
			return source, nil
		}, func(context.Context) error { committed.Store(true); return nil })
		if err == nil {
			result.publish(out)
		}
		out.End()
		done <- err
	}()
	event, ok, err := out.Next(ctx)
	if err != nil || !ok || event.Delta != "live" || !committed.Load() {
		t.Fatal("progress was delayed or preceded commit", err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	result, err := out.Result(ctx)
	if err != nil || result != message {
		t.Fatal("progress/final pointer mismatch", err)
	}
}

func TestNativeAttemptRelayCancellationStopsInFlightProducer(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	out := NewAssistantMessageEventStream()
	opened := make(chan struct{})
	ended := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		_, err := relayNativeAttempt(ctx, out, func(ctx context.Context) (*AssistantMessageEventStream, error) {
			source := NewAssistantMessageEventStreamFor(ctx)
			go func() { <-ctx.Done(); source.End(); close(ended) }()
			close(opened)
			return source, nil
		}, func(context.Context) error { t.Error("cancelled attempt committed"); return nil })
		done <- err
	}()
	<-opened
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal("cancellation lost", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancelled attempt stuck")
	}
	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("producer context not cancelled")
	}
	assertAttemptEmpty(t, out)
}
