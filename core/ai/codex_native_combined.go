package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net/url"
	"time"
)

var codexNativeSockets = newCodexSocketCache()
var codexNativePolicy = newCodexTransportPolicy()

// StreamCodexResponses selects the native WebSocket/SSE path using Pi's
// transport option and session fallback memory. apiKey is an OAuth access token.
func StreamCodexResponses(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions) *AssistantMessageEventStream {
	return streamCodexResponses(ctx, rawModel, transcript, options, false)
}

func appendCodexTransportDiagnostic(acc *codexResponsesAccumulator, err error, started bool, prepared *codexPreparedRequest, options CodexResponsesStreamOptions) {
	name := "Error"
	info := map[string]any{"message": err.Error()}
	var api *codexAPIError
	var closed *codexWebSocketCloseError
	if errors.As(err, &api) {
		name = "CodexApiError"
		if api.Code != "" {
			info["code"] = api.Code
		}
	}
	if errors.As(err, &closed) {
		name = "WebSocketCloseError"
		if closed.Code != nil {
			info["code"] = *closed.Code
		}
	}
	info["name"] = name
	phase := "before_message_stream_start"
	if started {
		phase = "after_message_stream_start"
	}
	transport := samplingString(prepared.controls["transport"])
	if transport == "" {
		transport = "auto"
	}
	details := map[string]any{"configuredTransport": transport, "eventsEmitted": started, "phase": phase, "requestBytes": len(prepared.body)}
	if !started {
		details["fallbackTransport"] = "sse"
	}
	now := time.Now().UnixMilli()
	if options.Now != nil {
		now = options.Now()
	}
	diagnostic := map[string]any{"type": "provider_transport_failure", "timestamp": now, "error": info, "details": details}
	acc.stream.Synchronize(func() {
		var diagnostics []any
		_ = json.Unmarshal(acc.output.Diagnostics, &diagnostics)
		diagnostics = append(diagnostics, diagnostic)
		acc.output.Diagnostics, _ = json.Marshal(diagnostics)
	})
}
func runCodexCombined(ctx context.Context, acc *codexResponsesAccumulator, rawModel json.RawMessage, options CodexResponsesStreamOptions, prepared *codexPreparedRequest) error {
	target := codexNativeURL(acc.model.BaseURL)
	connect := func(ctx context.Context) (*codexSocket, error) {
		parsed, err := url.Parse(target)
		if err != nil {
			return nil, err
		}
		switch parsed.Scheme {
		case "https":
			parsed.Scheme = "wss"
		case "http":
			parsed.Scheme = "ws"
		}
		return connectCodexWebSocket(ctx, parsed.String(), prepared.websocketHeaders, time.Duration(prepared.connectTimeout)*time.Millisecond, nil)
	}
	fallback, err := runCodexWebSocketTransport(ctx, codexNativePolicy, codexNativeSockets, prepared.session, prepared.account, connect, prepared.body, rawModel, acc, options, time.Duration(prepared.timeout)*time.Millisecond, func(err error, started bool) { appendCodexTransportDiagnostic(acc, err, started, prepared, options) }, nil)
	if err != nil {
		return err
	}
	if fallback {
		return runCodexSSEPrepared(ctx, acc, rawModel, options, defaultCompletionsRetryTiming(), prepared)
	}
	acc.finish(ctx)
	return nil
}

// CloseCodexResponsesSessions releases cached sockets for one session, or all
// sessions when session is empty. As in Pi, closing does not reset fallback memory.
func CloseCodexResponsesSessions(session string) { codexNativeSockets.closeSessions(session) }
