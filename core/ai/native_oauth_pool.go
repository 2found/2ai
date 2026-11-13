package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// nativeOAuthPoolStream keeps credentials local to each provider attempt and
// relays live native events. A terminal observation is copied to the host only
// when that attempt is published; a discarded auth failure cannot poison a later
// successful request. The concrete start callback also serves project-bound tokens.
func nativeOAuthPoolStream(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions, source TokenSource, label string,
	start func(context.Context, json.RawMessage, TranscriptContext, OpenAICompletionsStreamOptions, OAuthToken) (*AssistantMessageEventStream, error),
) (*AssistantMessageEventStream, error) {
	if source == nil {
		return nil, fmt.Errorf("%s account pool is required", label)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	ctx = safePhysicalPoolContext(ctx)
	rawModel = append(json.RawMessage(nil), rawModel...)
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	controls := map[string]json.RawMessage{}
	if len(options.Options) > 0 {
		if err := json.Unmarshal(options.Options, &controls); err != nil {
			return nil, err
		}
	}
	if controls == nil {
		controls = map[string]json.RawMessage{}
	}
	encoded, err := json.Marshal(transcript)
	if err != nil {
		return nil, err
	}
	var frozen Context
	if err = json.Unmarshal(encoded, &frozen); err != nil {
		return nil, err
	}
	transcript = NormalizeContext(frozen)
	out := NewAssistantMessageEventStreamFor(ctx)
	now := time.Now().UnixMilli()
	if options.Now != nil {
		now = options.Now()
	}
	recordNativeFailure(ctx, model.Provider, nil, false)
	fail := func(err error) {
		recordNativeFailure(ctx, model.Provider, err, false)
		reason := "error"
		if ctx.Err() != nil {
			reason = "aborted"
		}
		message := err.Error()
		result := &Message{Role: "assistant", Content: BlockContent(), API: model.API, Provider: model.Provider, Model: model.ID, Usage: &Usage{}, StopReason: reason, ErrorMessage: &message, Timestamp: now}
		out.Push(AssistantMessageEvent{Type: "error", Reason: reason, Error: result})
	}
	publishFailure := func(capture *NativeProviderFailure) {
		parent, _ := ctx.Value(nativeFailureKey{}).(*NativeProviderFailure)
		if parent == nil || capture == nil {
			return
		}
		failure, host := capture.Failure(), capture.HostFailure()
		parent.mu.Lock()
		parent.failure, parent.hostFailure, parent.kind = failure, host, capture.Kind()
		parent.mu.Unlock()
	}
	go func() {
		defer out.End()
		defer func() {
			if recovered := recover(); recovered != nil {
				fail(fmt.Errorf("%v", recovered))
			}
		}()
		state := newOAuthAttemptState()
		var previous *AssistantMessageEvent
		var previousFailure *NativeProviderFailure
		for {
			if ctx.Err() != nil {
				fail(errors.New("Request was aborted"))
				return
			}
			if state.attempts >= maxOAuthAuthAttempts {
				if previous != nil {
					publishFailure(previousFailure)
					out.Push(*previous)
				} else {
					fail(errors.New("OAuth account attempts exhausted"))
				}
				return
			}
			token, err := source.Acquire(ctx)
			if err != nil || !state.accept(token) {
				if previous != nil {
					publishFailure(previousFailure)
					out.Push(*previous)
					return
				}
				if err == nil {
					err = errors.New("OAuth token source returned an empty or repeated credential")
				}
				fail(err)
				return
			}
			controls["apiKey"], _ = json.Marshal(token.AccessToken)
			attemptOptions := options
			attemptOptions.Options, _ = json.Marshal(controls)
			attemptCtx, cancel := context.WithCancel(WithAssistantStreamSynchronization(ctx, out))
			attemptCtx, capture := WithNativeProviderFailure(attemptCtx)
			safeAttempt := startSafePhysical(attemptCtx, state.attempts)
			inner, err := start(attemptCtx, rawModel, transcript, attemptOptions, token)
			if err != nil {
				cancel()
				finishSafePhysical(ctx, safeAttempt, AssistantMessageEvent{Type: "error"}, err, false, false)
				fail(err)
				return
			}
			if inner == nil {
				cancel()
				finishSafePhysical(ctx, safeAttempt, AssistantMessageEvent{Type: "error"}, errors.New("missing provider stream"), false, false)
				fail(errors.New(label + " account stream was not created"))
				return
			}
			buffered := []AssistantMessageEvent{}
			committed := false
			retry := false
			flush := func() {
				for _, event := range buffered {
					out.Push(event)
				}
				buffered = nil
			}
			for {
				event, ok, readErr := inner.Next(context.WithoutCancel(ctx))
				if readErr != nil || !ok {
					cancel()
					_ = inner.WaitForEnd(context.WithoutCancel(ctx))
					if readErr == nil {
						readErr = errors.New(label + " account stream ended without a terminal event")
					}
					finishSafePhysical(ctx, safeAttempt, AssistantMessageEvent{Type: "error"}, readErr, capture.HostFailure(), true)
					fail(readErr)
					return
				}
				if event.Type == "error" {
					cancel()
					_ = inner.WaitForEnd(context.WithoutCancel(ctx))
					report := capture.Failure()
					if report == nil {
						inner.Synchronize(func() {
							message := label + " request failed"
							if event.Error != nil && event.Error.ErrorMessage != nil {
								message = *event.Error.ErrorMessage
							}
							report = errors.New(message)
						})
					}
					finishSafePhysical(ctx, safeAttempt, event, report, capture.HostFailure(), true)
					if !isOAuthConcurrencyCap(report) {
						source.Report(ctx, token, report)
					}
					if !committed && ctx.Err() == nil && !capture.HostFailure() && isOAuthAuthFailure(report) {
						copy := event
						previous = &copy
						previousFailure = capture
						state.lastAuth = report
						retry = true
						break
					}
					publishFailure(capture)
					flush()
					out.Push(event)
					return
				}
				if event.Type == "done" {
					cancel()
					_ = inner.WaitForEnd(context.WithoutCancel(ctx))
					finishSafePhysical(ctx, safeAttempt, event, nil, false, true)
					if ctx.Err() == nil {
						source.Report(ctx, token, nil)
					}
					publishFailure(capture)
					flush()
					out.Push(event)
					return
				}
				visible := (event.Type == "text_delta" || event.Type == "thinking_delta" || event.Type == "toolcall_delta") && event.Delta != "" || event.Type == "toolcall_start" || event.Type == "text_end" || event.Type == "thinking_end"
				if !committed && !visible {
					buffered = append(buffered, event)
					continue
				}
				if !committed {
					committed = true
					flush()
				}
				out.Push(event)
			}
			if !retry {
				return
			}
		}
	}()
	return out, nil
}
