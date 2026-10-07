package ai

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"time"
)

// Failure memory is scoped by the original session ID, independently of account
// and the clamped prompt-cache/header ID. Explicit websocket modes share it.
type codexTransportPolicy struct {
	mu       sync.Mutex
	fallback map[string]bool
	stats    map[string]*CodexWebSocketDebugStats
}

func newCodexTransportPolicy() *codexTransportPolicy {
	return &codexTransportPolicy{fallback: map[string]bool{}, stats: map[string]*CodexWebSocketDebugStats{}}
}
func (p *codexTransportPolicy) disabled(session string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return session != "" && p.fallback[session]
}
func (p *codexTransportPolicy) recordFailure(session string, err error) {
	if session == "" {
		return
	}
	p.mu.Lock()
	p.fallback[session] = true
	stats := p.statsLocked(session)
	stats.WebSocketFailures++
	message := err.Error()
	stats.LastWebSocketError = &message
	active := true
	stats.WebSocketFallbackActive = &active
	p.mu.Unlock()
}
func (p *codexTransportPolicy) reset(session string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if session == "" {
		clear(p.fallback)
		clear(p.stats)
	} else {
		delete(p.fallback, session)
		delete(p.stats, session)
	}
}
func codexNonTransportError(err error) bool {
	var api *codexAPIError
	var protocol *codexProtocolError
	var callback *codexProviderCallbackError
	return errors.As(err, &api) || errors.As(err, &protocol) || errors.As(err, &callback)
}

// run returns true only when SSE should execute. onFailure records the diagnostic
// before failure memory changes; errors after start never fall back mid-message.
func (p *codexTransportPolicy) run(ctx context.Context, session, transport string, attempt func(func()) error, onStart func(), onFailure func(error, bool), onFallback func()) (bool, error) {
	if transport == "sse" {
		return true, nil
	}
	if p.disabled(session) {
		p.recordFallback(session)
		if onFallback != nil {
			onFallback()
		}
		return true, nil
	}
	retriedLimit, retriedMissing, startEmitted := false, false, false
	for {
		started := false
		err := attempt(func() {
			started = true
			if !startEmitted {
				startEmitted = true
				if onStart != nil {
					onStart()
				}
			}
		})
		if err == nil {
			if ctx.Err() != nil {
				return false, errors.New("Request was aborted")
			}
			return false, nil
		}
		var api *codexAPIError
		errors.As(err, &api)
		limit := !started && api != nil && api.Code == "websocket_connection_limit_reached"
		missing := api != nil && api.Code == "previous_response_not_found"
		if ctx.Err() == nil && missing && !retriedMissing {
			retriedMissing = true
			continue
		}
		if ctx.Err() == nil && limit && !retriedLimit {
			retriedLimit = true
			continue
		}
		if ctx.Err() != nil || (codexNonTransportError(err) && !limit) {
			return false, err
		}
		if onFailure != nil {
			onFailure(err, started)
		}
		p.recordFailure(session, err)
		if started {
			return false, err
		}
		p.recordFallback(session)
		if onFallback != nil {
			onFallback()
		}
		return true, nil
	}
}

// runCodexWebSocketTransport combines policy with concrete request/cache ownership.
// A true result hands the already prepared logical request to SSE.
func runCodexWebSocketTransport(ctx context.Context, policy *codexTransportPolicy, cache *codexSocketCache, session, account string, connect func(context.Context) (*codexSocket, error), body, rawModel json.RawMessage, acc *codexResponsesAccumulator, options CodexResponsesStreamOptions, idle time.Duration, onFailure func(error, bool), onFallback func()) (bool, error) {
	controls, _ := samplingObject(options.Options)
	transport := samplingString(controls["transport"])
	return policy.run(ctx, session, transport, func(start func()) error {
		if err := processCodexWebSocketStarted(ctx, cache, session, account, connect, body, rawModel, acc, options, idle, start, policy); err != nil {
			return err
		}
		if ctx.Err() != nil {
			return errors.New("Request was aborted")
		}
		var failure string
		acc.stream.Synchronize(func() {
			switch acc.output.StopReason {
			case "pending":
				failure = "OpenAI Responses stream ended without a stop reason"
			case "error", "aborted":
				failure = "An unknown error occurred"
				if acc.output.ErrorMessage != nil && *acc.output.ErrorMessage != "" {
					failure = *acc.output.ErrorMessage
				}
			}
		})
		if failure != "" {
			return errors.New(failure)
		}
		return nil
	}, acc.start, onFailure, onFallback)
}
