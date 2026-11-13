package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// StreamCodexResponsesPooled binds an existing host account source to the native
// simple stream. Credential rotation is limited to typed authentication failures
// before visible content; the selected token is never stored on a shared client.
func StreamCodexResponsesPooled(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions, source TokenSource) (*AssistantMessageEventStream, error) {
	if source == nil {
		return nil, errors.New("Codex account pool is required")
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
	fail := func(err error) {
		acc := newCodexResponsesAccumulator(model, nil, out, now)
		acc.fail(err.Error(), ctx.Err() != nil)
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
		for {
			if ctx.Err() != nil {
				fail(errors.New("Request was aborted"))
				return
			}
			if state.attempts >= maxOAuthAuthAttempts {
				if previous != nil {
					out.Push(*previous)
				} else {
					fail(errors.New("OAuth account attempts exhausted"))
				}
				return
			}
			token, err := source.Acquire(ctx)
			if err != nil || !state.accept(token) {
				if previous != nil {
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
			rawOptions, _ := json.Marshal(controls)
			attemptOptions := options
			attemptOptions.Options, err = BuildCodexResponsesSimpleOptions(rawModel, transcript, rawOptions)
			if err != nil {
				fail(err)
				return
			}
			attemptCtx, cancel := context.WithCancel(ctx)
			// Forward live Pi payload pointers under one lock across all attempts.
			inner := NewAssistantMessageEventStreamFor(WithAssistantStreamSynchronization(attemptCtx, out))
			var cause error
			safeAttempt := startSafePhysical(attemptCtx, state.attempts)
			streamCodexResponsesInto(attemptCtx, rawModel, transcript, attemptOptions, false, func(err error) { cause = err }, inner)
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
						readErr = errors.New("Codex account stream ended without a terminal event")
					}
					finishSafePhysical(ctx, safeAttempt, AssistantMessageEvent{Type: "error"}, readErr, false, true)
					fail(readErr)
					return
				}
				if event.Type == "error" {
					cancel()
					_ = inner.WaitForEnd(context.WithoutCancel(ctx))
					report := codexPoolFailure(cause)
					if report == nil {
						inner.Synchronize(func() {
							message := "Codex request failed"
							if event.Error != nil && event.Error.ErrorMessage != nil {
								message = *event.Error.ErrorMessage
							}
							report = errors.New(message)
						})
					}
					finishSafePhysical(ctx, safeAttempt, event, report, codexNonTransportError(cause), true)
					if !isOAuthConcurrencyCap(report) {
						source.Report(ctx, token, report)
					}
					if !committed && ctx.Err() == nil && isOAuthAuthFailure(report) && !codexNonTransportError(cause) {
						copy := event
						previous = &copy
						state.lastAuth = report
						retry = true
						break
					}
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
