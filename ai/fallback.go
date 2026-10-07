package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

// StreamFn is the native provider contract shared by providers and the engine.
type StreamFn func(context.Context, json.RawMessage, TranscriptContext, map[string]any) (*AssistantMessageEventStream, error)

// PreparationError marks host setup failures, which must never retry or fall
// back even when their cause is a typed provider error.
type PreparationError struct{ Cause error }

func (e *PreparationError) Error() string { return e.Cause.Error() }
func (e *PreparationError) Unwrap() error { return e.Cause }

// FallbackCandidate binds a model to its native provider transport.
type FallbackCandidate struct {
	Model  json.RawMessage
	Stream StreamFn
}

// FallbackProvider treats retry and ordered provider fallback as one native
// stream. Each candidate receives its own retry budget. Once content is
// visible, a request can never replay on another attempt or candidate.
// Fields are immutable while requests are running; request selection belongs
// to the host and is supplied separately to Run.
type FallbackProvider struct {
	Candidates []FallbackCandidate
	Retry      protocol.RetryPolicy
	// Wait optionally supplies a clock for backoff. Nil uses a cancellable timer.
	Wait func(context.Context, time.Duration) error
}

// FallbackRequest binds host policy to one logical request. Open prepares a
// fresh candidate on every attempt; Commit durably selects it before visible
// output. Observe sees all settled attempts, including discarded failures.
// Validate fences host state after settlement, before escalation/publication.
// BeforePublish runs once before the chosen terminal event is exposed.
type FallbackRequest struct {
	Start         int
	Candidates    int
	Open          func(context.Context, int, int) (*AssistantMessageEventStream, error)
	Commit        func(context.Context, int) error
	Observe       func(context.Context, int, FallbackAttempt) error
	Validate      func() error
	BeforePublish func(AttemptOutcome) error
	// Recover may repair one confirmed pre-output provider rejection per rung.
	// AI still owns replay safety and the hard one-recovery bound.
	Recover func(context.Context, int, FallbackAttempt) (bool, error)
}

// Run publishes to a caller-owned stream without ending it. Hosts with durable
// selection/request admission use this entry point; ordinary consumers use Stream.
func (p FallbackProvider) Run(ctx context.Context, out *AssistantMessageEventStream, request FallbackRequest) (AttemptOutcome, error) {
	if out == nil || request.Open == nil || request.Start < 0 || request.Start >= request.Candidates {
		return AttemptOutcome{}, errors.New("fallback requires a stream, provider and active candidate")
	}
	for index := request.Start; index < request.Candidates; index++ {
		if request.Validate != nil {
			if err := request.Validate(); err != nil {
				return AttemptOutcome{}, err
			}
		}
		attempts := nativeRungAttempts{
			policy: p.Retry,
			open: func(ctx context.Context, number int) (*AssistantMessageEventStream, error) {
				return request.Open(ctx, index, number)
			},
			commit: func(ctx context.Context) error {
				if request.Commit != nil {
					return request.Commit(ctx, index)
				}
				return nil
			},
			wait: p.Wait,
		}
		if request.Observe != nil {
			attempts.observe = func(ctx context.Context, attempt FallbackAttempt) error { return request.Observe(ctx, index, attempt) }
		}
		if request.Recover != nil {
			attempts.recover = func(ctx context.Context, attempt FallbackAttempt) (bool, error) {
				return request.Recover(ctx, index, attempt)
			}
		}
		result, err := attempts.run(ctx, out)
		if err != nil {
			return result, err
		}
		if request.Validate != nil {
			if err := request.Validate(); err != nil {
				return result, err
			}
		}
		aborted := result.Terminal.Reason == "aborted" || (result.Terminal.Error != nil && result.Terminal.Error.StopReason == "aborted")
		if result.Committed || result.Terminal.Type == "done" || aborted || index+1 == request.Candidates {
			if result.AdmissionError == nil && request.BeforePublish != nil {
				if err := request.BeforePublish(result); err != nil {
					return result, err
				}
			}
			return result, result.publish(out)
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
	}
	return AttemptOutcome{}, errors.New("fallback has no active candidate")
}

// Stream implements StreamFn. Candidate models override the supplied model;
// an omitted candidate model uses it. It keeps no mutable selection between calls.
func (p FallbackProvider) Stream(ctx context.Context, model json.RawMessage, transcript TranscriptContext, options map[string]any) (*AssistantMessageEventStream, error) {
	if len(p.Candidates) == 0 {
		return nil, errors.New("fallback requires candidates")
	}
	for _, candidate := range p.Candidates {
		if candidate.Stream == nil {
			return nil, errors.New("fallback candidate requires a provider")
		}
	}
	done := make(chan struct{})
	var final AttemptOutcome
	var failure error
	wait := func(ctx context.Context) error {
		select {
		case <-done:
			return failure
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	out := NewAssistantMessageEventStreamFor(ctx)
	out.hooks = AssistantStreamHooks{
		Result: func(ctx context.Context) (*Message, error) {
			if err := wait(ctx); err != nil {
				return nil, err
			}
			return final.Message(), nil
		},
		AfterEnd: wait,
	}
	go func() {
		defer func() {
			if value := recover(); value != nil {
				failure = fmt.Errorf("fallback provider panic: %v", value)
			}
			out.End()
			close(done)
		}()
		final, failure = p.Run(ctx, out, FallbackRequest{
			Candidates: len(p.Candidates),
			Open: func(ctx context.Context, index, _ int) (*AssistantMessageEventStream, error) {
				candidate := p.Candidates[index]
				selected := candidate.Model
				if len(selected) == 0 {
					selected = model
				}
				attemptOptions := make(map[string]any, len(options))
				for key, value := range options {
					attemptOptions[key] = value
				}
				snapshot := streamSnapshot{messages: map[*Message]*Message{}, blocks: map[*ContentBlock]*ContentBlock{}, usages: map[*Usage]*Usage{}}
				attemptTranscript := TranscriptContext{messages: make([]Message, len(transcript.messages))}
				for i := range transcript.messages {
					attemptTranscript.messages[i] = *snapshot.message(&transcript.messages[i])
				}
				return candidate.Stream(ctx, append(json.RawMessage(nil), selected...), attemptTranscript, attemptOptions)
			},
		})
	}()
	return out, nil
}

// Message returns the original settled message without copying native payloads.
func (r AttemptOutcome) Message() *Message {
	if r.Terminal.Type == "done" {
		return r.Terminal.Message
	}
	if r.Terminal.Type == "error" {
		return r.Terminal.Error
	}
	return nil
}
