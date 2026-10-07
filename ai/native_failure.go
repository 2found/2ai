package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"sync"

	"github.com/2found/2ai/ai/protocol"
)

type nativeFailureKey struct{}

// NativeProviderFailure is a request-scoped, passive host observation. It never
// changes Pi's event JSON or wraps the error passed to Pi's display formatter.
type NativeProviderFailure struct {
	mu          sync.Mutex
	failure     error
	hostFailure bool
}

func (c *NativeProviderFailure) HostFailure() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hostFailure
}

func WithNativeProviderFailure(ctx context.Context) (context.Context, *NativeProviderFailure) {
	capture := &NativeProviderFailure{}
	return context.WithValue(ctx, nativeFailureKey{}, capture), capture
}
func (c *NativeProviderFailure) Failure() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if failure, ok := c.failure.(*protocol.ProviderError); ok {
		copy := *failure
		return &copy
	}
	return c.failure
}
func recordNativeFailure(ctx context.Context, provider string, cause error, callbackFailure bool) {
	capture, _ := ctx.Value(nativeFailureKey{}).(*NativeProviderFailure)
	if capture == nil {
		return
	}
	var failure error
	if cause != nil {
		failure = errors.New(cause.Error())
		if !callbackFailure && !codexNonTransportError(cause) {
			var providerError *protocol.ProviderError
			var openAI *completionsRequestError
			var codex *codexHTTPError
			var anthropic *AnthropicClientError
			var piMessages *PiMessagesResponseError
			var transport net.Error
			switch {
			case errors.As(cause, &providerError) && providerError.Provider == provider:
				failure = providerError
			case errors.As(cause, &openAI):
				failure = protocol.NewProviderError(provider, &http.Response{StatusCode: openAI.status, Header: openAI.headers.Clone()}, openAI.message)
			case errors.As(cause, &codex):
				failure = protocol.NewProviderError(provider, &http.Response{StatusCode: codex.Status, Header: codex.Headers.Clone()}, codex.Message)
			case errors.As(cause, &anthropic):
				failure = protocol.NewProviderError(provider, &http.Response{StatusCode: anthropic.Status, Header: anthropic.Headers.Clone()}, anthropic.Message)
			case errors.As(cause, &piMessages):
				failure = protocol.NewProviderError(provider, &http.Response{StatusCode: piMessages.Status, Header: piMessages.Headers.Clone()}, piMessages.Message)
			case errors.As(cause, &transport):
				failure = &protocol.ProviderError{Provider: provider, Message: cause.Error()}
			}
		}
	}
	capture.mu.Lock()
	capture.failure = failure
	capture.hostFailure = cause != nil && callbackFailure
	capture.mu.Unlock()
}

// Track callback origin without changing the original error: a callback may
// return a typed transport/provider error, but that is not an HTTP failure and
// must not trigger host retries or account rotation. Callbacks are serial.
func nativeFailureCallbacks(options OpenAICompletionsStreamOptions) (OpenAICompletionsStreamOptions, *bool) {
	failed := new(bool)
	if original := options.OnPayload; original != nil {
		options.OnPayload = func(ctx context.Context, payload, model json.RawMessage) (json.RawMessage, error) {
			value, err := original(ctx, payload, model)
			if err != nil {
				*failed = true
			}
			return value, err
		}
	}
	if original := options.OnResponse; original != nil {
		options.OnResponse = func(ctx context.Context, response CompletionsResponse, model json.RawMessage) error {
			err := original(ctx, response, model)
			if err != nil {
				*failed = true
			}
			return err
		}
	}
	if original := options.OnProviderStreamEvent; original != nil {
		options.OnProviderStreamEvent = func(ctx context.Context, event *json.RawMessage, model json.RawMessage) error {
			err := original(ctx, event, model)
			if err != nil {
				*failed = true
			}
			return err
		}
	}
	return options, failed
}
