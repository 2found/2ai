package ai

import (
	"context"
	"errors"
)

// AttemptOutcome keeps the original terminal event and any withheld
// metadata. The caller decides retry/escalation and publishes a final terminal;
// a failed attempt must not settle the logical request's stream.
type AttemptOutcome struct {
	AdmissionError   error
	Terminal         AssistantMessageEvent
	pending          []AssistantMessageEvent
	Committed        bool
	failureKind      string
	producerAdmitted bool
}

func (r AttemptOutcome) publish(out *AssistantMessageEventStream) error {
	if r.AdmissionError != nil {
		return r.AdmissionError
	}
	for _, event := range r.pending {
		out.Push(event)
	}
	out.Push(r.Terminal)
	return nil
}

// relayNativeAttempt streams one native attempt. open must pass its supplied
// context to the provider so live payloads share out's synchronization. commit
// persists/applies the selected model before any non-replayable event escapes.
//
// A returned Go error (cancellation, stream protocol or commit failure) aborts
// orchestration. An admissionError or terminal provider error may be considered
// for retry only with committed=false. Admission errors retain their concrete
// type/Retry-After metadata. Error text never decides whether content escaped.
func relayNativeAttempt(ctx context.Context, out *AssistantMessageEventStream, open func(context.Context) (*AssistantMessageEventStream, error), commit func(context.Context) error) (result AttemptOutcome, err error) {
	if out == nil || open == nil || commit == nil {
		return result, errors.New("native attempt requires a stream, provider and commit callback")
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	attemptCtx, cancel := context.WithCancel(WithAssistantStreamSynchronization(ctx, out))
	defer cancel()
	source, err := open(attemptCtx)
	if err != nil {
		if ctx.Err() != nil {
			return result, ctx.Err()
		}
		result.AdmissionError = err
		return result, nil
	}
	if source == nil {
		result.failureKind = "provider_protocol"
		return result, errors.New("native attempt returned no stream")
	}
	result.producerAdmitted = true
	// Early host/protocol exits still cancel and join the real producer. Keep
	// the original error/replay decision while retaining any aborted usage.
	defer func() {
		if err == nil {
			return
		}
		cancel()
		readCtx := context.WithoutCancel(ctx)
		_ = source.WaitForEnd(readCtx)
		for {
			// Cleanup drains producer events only; a failed authoritative
			// AfterEnd hook must not run a second time.
			event, ok, readErr := source.EventStream.Next(readCtx)
			if readErr != nil || !ok {
				break
			}
			if event.Type == "done" || event.Type == "error" {
				result.Terminal = event
			}
		}
	}()
	admit := func() error {
		if result.Committed {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := commit(ctx); err != nil {
			result.failureKind = "host_callback"
			return err
		}
		result.Committed = true
		for _, event := range result.pending {
			out.Push(event)
		}
		result.pending = nil
		return nil
	}
	// Providers own abort settlement, as in Pi's engine. Cancel their request
	// signal, but drain their terminal message so native content/usage survives.
	// A provider which closes without a terminal still returns cancellation.
	readCtx := context.WithoutCancel(ctx)
	for {
		event, ok, readErr := source.Next(readCtx)
		if readErr != nil {
			return result, readErr
		}
		if !ok {
			if err = ctx.Err(); err != nil {
				return result, err
			}
			result.failureKind = "provider_protocol"
			return result, errors.New("native attempt ended without a terminal event")
		}
		if event.Type == "done" || event.Type == "error" {
			result.Terminal = event
			// A terminal event can precede producer settlement (e.g. proxy drainage).
			// Observe settlement/AfterEnd before allowing another attempt to start.
			if err = source.WaitForEnd(readCtx); err != nil {
				return result, err
			}
			if _, _, err = source.Next(readCtx); err != nil {
				return result, err
			}
			if event.Type == "done" {
				if err = admit(); err != nil {
					return result, err
				}
			}
			return result, nil
		}
		if !result.Committed && nativeAttemptReplaySafe(event) {
			result.pending = append(result.pending, event)
			continue
		}
		if !result.Committed && ctx.Err() != nil {
			// Cancellation cannot select a speculative model or expose its live
			// effects. Its settled aborted message is still returned to Pi.
			continue
		}
		if err = admit(); err != nil {
			return result, err
		}
		out.Push(event)
	}
}

func nativeAttemptReplaySafe(event AssistantMessageEvent) bool {
	switch event.Type {
	case "start", "text_start", "thinking_start":
		return true
	case "text_delta", "thinking_delta", "toolcall_delta":
		return event.Delta == ""
	default:
		// Tool starts/ends, completed text/thinking, and unknown future events
		// cannot be replayed after downstream code has observed them.
		return false
	}
}
