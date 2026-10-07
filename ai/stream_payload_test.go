package ai_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/2found/2ai/ai"
)

func TestPayloadYieldLetsProviderSettleWithLiveIdentity(t *testing.T) {
	stream := ai.NewAssistantMessageEventStream()
	message := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "before"})}
	stream.Push(ai.AssistantMessageEvent{Type: "start", Partial: message})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	start := make(chan struct{})
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		<-start
		stream.Synchronize(func() {
			message.Content.Blocks.Get(0).Text += "/provider"
			message.StopReason = "stop"
			stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
		})
		stream.End()
	}()
	var retained ai.PayloadYield
	err := stream.SynchronizeYielding(func(yield ai.PayloadYield) error {
		retained = yield
		message.Content.Blocks.Get(0).Text = "listener"
		close(start)
		var result *ai.Message
		if err := yield(func() (err error) {
			result, err = stream.Result(ctx)
			return err
		}); err != nil {
			return err
		}
		if result != message || result.Content.Blocks.Get(0).Text != "listener/provider" {
			t.Error("yield lost the provider/listener's shared message")
		}
		message.Timestamp = 77
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	<-finished
	if message.Timestamp != 77 {
		t.Fatal("post-yield mutation lost")
	}
	if err := retained(func() error { t.Error("expired yield ran work"); return nil }); err == nil {
		t.Fatal("retained yield was accepted")
	}
}

func TestPayloadYieldPreservesFailuresAndLock(t *testing.T) {
	for _, mode := range []string{"error", "panic"} {
		t.Run(mode, func(t *testing.T) {
			stream := ai.NewAssistantMessageEventStream()
			failure := errors.New("work failed")
			func() {
				defer func() {
					if value := recover(); value != nil {
						if mode != "panic" || value != failure {
							t.Errorf("panic changed: %v", value)
						}
					} else if mode == "panic" {
						t.Error("panic was swallowed")
					}
				}()
				err := stream.SynchronizeYielding(func(yield ai.PayloadYield) error {
					return yield(func() error {
						if err := yield(func() error { t.Error("nested yield ran work"); return nil }); err == nil {
							t.Error("nested yield accepted")
						}
						if mode == "panic" {
							panic(failure)
						}
						return failure
					})
				})
				if mode == "error" && err != failure {
					t.Errorf("error changed: %v", err)
				}
			}()
			// Both failure paths must release the lock for the next observer.
			done := make(chan struct{})
			go func() { stream.Synchronize(func() {}); close(done) }()
			select {
			case <-done:
			case <-time.After(3 * time.Second):
				t.Fatal("payload lock leaked")
			}
		})
	}
}

func TestPayloadYieldCallbackExitWaitsForOutstandingYield(t *testing.T) {
	stream := ai.NewAssistantMessageEventStream()
	started, release := make(chan struct{}), make(chan struct{})
	yielded, returned := make(chan error, 1), make(chan error, 1)
	go func() {
		returned <- stream.SynchronizeYielding(func(yield ai.PayloadYield) error {
			go func() {
				yielded <- yield(func() error {
					close(started)
					<-release
					return nil
				})
			}()
			<-started
			return nil
		})
	}()
	<-started
	// The provider can still acquire the payload lock while the callback's
	// cleanup waits for the outstanding yield to reacquire and return it.
	stream.Synchronize(func() {})
	select {
	case err := <-returned:
		t.Fatalf("callback returned before its yield: %v", err)
	default:
	}
	close(release)
	for _, completion := range []<-chan error{yielded, returned} {
		select {
		case err := <-completion:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("yield cleanup did not settle")
		}
	}
	stream.Synchronize(func() {})
}
