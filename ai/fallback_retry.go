package ai

import (
	"context"
	"errors"
	"time"

	"github.com/2found/2ai/ai/protocol"
	"github.com/2found/2ai/telemetry"
)

// FallbackAttempt is delivered after producer settlement, including failed
// attempts whose terminal event is withheld from the logical request. Observers
// can account for usage without inserting failed messages into the transcript.
type FallbackAttempt struct {
	Number      int
	Outcome     AttemptOutcome
	Failure     error
	FailureKind string
}

type nativeRungAttempts struct {
	safeCall   telemetry.SafeCall
	safeLabels telemetry.SafeLabels
	policy     protocol.RetryPolicy
	open       func(context.Context, int) (*AssistantMessageEventStream, error)
	commit     func(context.Context) error
	// failure returns transport metadata observed for this completed attempt.
	// It must not infer HTTP status/Retry-After from a formatted error message.
	failure func(int) error
	observe func(context.Context, FallbackAttempt) error
	recover func(context.Context, FallbackAttempt) (bool, error)
	wait    func(context.Context, time.Duration) error
}

// run drains attempts on one rung, leaving the final outcome unpublished so the
// ladder can decide escalation. Once content commits, errors never replay. Host
// callback/protocol failures abort immediately. Each rung owns its retry budget.
func (r nativeRungAttempts) run(ctx context.Context, out *AssistantMessageEventStream) (last AttemptOutcome, err error) {
	if r.open == nil || r.commit == nil {
		return last, errors.New("native retries require provider and commit callbacks")
	}
	wait := r.wait
	if wait == nil {
		wait = waitNativeRetry
	}
	recovered := false
	for number := 1; ; number++ {
		if err = ctx.Err(); err != nil {
			return last, err
		}
		attemptCtx, capture := WithNativeProviderFailure(ctx)
		var safeAttempt *telemetry.SafeAttempt
		physical := &safePhysicalObservation{call: r.safeCall, labels: r.safeLabels, scope: telemetry.CaptureSafeOrigin(ctx)}
		last, err = relayNativeAttempt(attemptCtx, out, func(ctx context.Context) (*AssistantMessageEventStream, error) {
			// Only nested composition in the same observation scope replaces
			// this producer. Auxiliary preparation has its own span/call.
			if parent, _ := ctx.Value(safePhysicalKey{}).(*safePhysicalObservation); parent != nil && parent.scope == physical.scope {
				parent.delegated.Store(true)
			}
			safeAttempt = r.safeCall.StartAttempt(number, r.safeLabels)
			return r.open(context.WithValue(ctx, safePhysicalKey{}, physical), number)
		}, r.commit)
		failure := last.AdmissionError
		var preparation *PreparationError
		if err == nil && errors.As(failure, &preparation) {
			err = failure
		}
		if err != nil {
			failure = err
		} else if last.Terminal.Type == "error" {
			failure = capture.Failure()
			if r.failure != nil {
				failure = r.failure(number)
			}
			if failure == nil {
				message := "native provider request failed"
				if last.Terminal.Error != nil && last.Terminal.Error.ErrorMessage != nil {
					message = *last.Terminal.Error.ErrorMessage
				}
				failure = errors.New(message)
			}
			if capture.HostFailure() {
				err = failure
			}
		}
		kind := capture.Kind()
		if last.failureKind != "" {
			kind = last.failureKind
		}
		if kind == "" || kind == "none" {
			kind = nativeFailureKind(failure, capture.HostFailure())
		}
		status := "completed"
		if failure != nil {
			status = "failed"
		}
		if last.Terminal.Reason == "aborted" || (last.Terminal.Error != nil && last.Terminal.Error.StopReason == "aborted") || ctx.Err() != nil {
			status = "cancelled"
			kind = nativeFailureKind(ctx.Err(), false)
			if kind == "none" {
				kind = "request_cancelled"
			}
		}
		if r.observe != nil {
			if observeErr := r.observe(context.WithoutCancel(ctx), FallbackAttempt{Number: number, Outcome: last, Failure: failure, FailureKind: kind}); observeErr != nil {
				if !physical.delegated.Load() {
					finishSafeNative(safeAttempt, "failed", "host_callback", last.Message(), last.producerAdmitted)
				}
				return last, observeErr
			}
		}
		if !physical.delegated.Load() {
			finishSafeNative(safeAttempt, status, kind, last.Message(), last.producerAdmitted)
		}
		if err != nil {
			return last, err
		}
		if last.Terminal.Reason == "aborted" || (last.Terminal.Error != nil && last.Terminal.Error.StopReason == "aborted") {
			return last, nil
		}
		if ctx.Err() != nil {
			return last, ctx.Err()
		}
		if last.Committed || last.Terminal.Type == "done" {
			return last, nil
		}
		if r.recover != nil && !recovered && failure != nil {
			ok, recoverErr := r.recover(ctx, FallbackAttempt{Number: number, Outcome: last, Failure: failure, FailureKind: kind})
			if recoverErr != nil {
				return last, recoverErr
			}
			if ok {
				recovered = true
				continue
			}
		}
		delay, retry := r.policy.NextDelay(number, failure)
		if !retry {
			return last, nil
		}
		if err = wait(ctx, delay); err != nil {
			return last, err
		}
	}
}

func waitNativeRetry(ctx context.Context, delay time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return ctx.Err()
	}
}
