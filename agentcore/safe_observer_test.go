package agentcore_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry"
)

func TestSafeAbsentSlowErrorAndPanicConsumersKeepRunIdentical(t *testing.T) {
	const state = `{"revision":"eeac84ca92498ac18b6832754d01aef1d3c5f654","model":{"id":"test"},"messages":[{"role":"user","content":"private prompt","timestamp":1}]}`
	build := func() *agentcore.Agent {
		a, err := agentcore.New(agentcore.Config{Model: "test", NativeProvider: &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", Timestamp: 2, StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "same answer"}), Usage: &ai.Usage{Input: 3, Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}})}}}})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	baseline, err := build().RunNative(context.Background(), agentcore.NativeRun{State: json.RawMessage(state)})
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"absent", "slow", "error", "panic"} {
		t.Run(mode, func(t *testing.T) {
			o, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{Capacity: 1})
			if err != nil {
				t.Fatal(err)
			}
			ctx, _ := telemetry.WithSafeObserver(context.Background(), o, telemetry.SafeExecution{ExecutionID: "consumer-test"})
			started, release, consumerDone := make(chan struct{}), make(chan struct{}), make(chan error, 1)
			sentinel := errors.New("private sink error")
			if mode != "absent" {
				telemetry.NewSafeCall(ctx).StartAttempt(1, telemetry.SafeLabels{}).Finish("completed", "none", telemetry.SafeAccounting{})
				go func() {
					defer func() {
						if p := recover(); p != nil {
							if p != sentinel {
								consumerDone <- errors.New("panic identity changed")
							} else {
								consumerDone <- sentinel
							}
						}
					}()
					records := o.Drain(1)
					if len(records) != 1 {
						consumerDone <- errors.New("missing host drain seed")
						return
					}
					close(started)
					<-release
					if mode == "panic" {
						panic(sentinel)
					}
					if mode == "error" {
						consumerDone <- sentinel
					} else {
						consumerDone <- nil
					}
				}()
				<-started
			}
			// Completion cannot depend on any host consumer callback returning.
			result, err := build().RunNative(ctx, agentcore.NativeRun{State: json.RawMessage(state)})
			if mode != "absent" {
				close(release)
				got := <-consumerDone
				if (mode == "error" || mode == "panic") && got != sentinel || mode == "slow" && got != nil {
					t.Fatal("host consumer outcome lost", got)
				}
			}
			if err != nil || result.Final != baseline.Final || result.Usage != baseline.Usage || !reflect.DeepEqual(stableSafeCheckpoint(t, result.NativeState), stableSafeCheckpoint(t, baseline.NativeState)) {
				t.Fatalf("consumer %s changed native result/state: err=%v usage=%+v", mode, err, result.Usage)
			}
			h := o.Health()
			if h.Overflow != 3 || h.Queued != 1 || h.Complete {
				t.Fatalf("capacity-one exact drops: %+v", h)
			}
		})
	}
}

func stableSafeCheckpoint(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var checkpoint map[string]any
	if err := json.Unmarshal(raw, &checkpoint); err != nil {
		t.Fatal(err)
	}
	// Independent runs author fresh system timestamps in nativeSystemMessage.
	// Ignore only those wall-clock stamps; user/provider payloads stay exact.
	for _, item := range checkpoint["messages"].([]any) {
		message := item.(map[string]any)
		if message["role"] == "system" {
			delete(message, "timestamp")
		}
	}
	return checkpoint
}
