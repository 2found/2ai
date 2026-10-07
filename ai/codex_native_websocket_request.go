package ai

import (
	"context"
	"encoding/json"
	"errors"
	"time"
)

type codexProviderCallbackError struct{ cause error }

func (e *codexProviderCallbackError) Error() string { return e.cause.Error() }
func (e *codexProviderCallbackError) Unwrap() error { return e.cause }

// processCodexWebSocket owns the lease through response conversion. Its caller
// decides retries/fallback and publishes the final done/error event.
func processCodexWebSocket(ctx context.Context, cache *codexSocketCache, session, account string, connect func(context.Context) (*codexSocket, error), body, rawModel json.RawMessage, acc *codexResponsesAccumulator, options CodexResponsesStreamOptions, idle time.Duration) (err error) {
	return processCodexWebSocketStarted(ctx, cache, session, account, connect, body, rawModel, acc, options, idle, acc.start, nil)
}
func processCodexWebSocketStarted(ctx context.Context, cache *codexSocketCache, session, account string, connect func(context.Context) (*codexSocket, error), body, rawModel json.RawMessage, acc *codexResponsesAccumulator, options CodexResponsesStreamOptions, idle time.Duration, onStart func(), stats *codexTransportPolicy) (err error) {
	lease, err := cache.acquire(ctx, session, account, connect)
	if err != nil {
		return err
	}
	keep := true
	defer func() {
		recovered := recover()
		if err != nil || recovered != nil {
			keep = false
			if lease.entry != nil {
				lease.entry.continuation = nil
			}
		}
		lease.release(keep)
		if recovered != nil {
			panic(recovered)
		}
	}()
	controls, _ := samplingObject(options.Options)
	transport := samplingString(controls["transport"])
	cached := transport == "websocket-cached" || transport == "auto"
	request := body
	if cached && lease.entry != nil {
		request, err = codexCachedWebSocketBody(body, &lease.entry.continuation)
		if err != nil {
			return err
		}
	}
	if stats != nil {
		stats.recordRequest(session, lease.reused, cached, request)
	}
	fields, _ := samplingObject(request)
	if fields == nil {
		fields = map[string]json.RawMessage{}
	}
	keys := []string{"type"}
	fields["type"] = json.RawMessage(`"response.create"`)
	// Object spread preserves a caller's explicit type and its original position
	// follows the leading type key in the final request.
	original, _ := samplingObject(request)
	if value, ok := original["type"]; ok {
		fields["type"] = value
	}
	for _, key := range samplingObjectKeys(request) {
		if key != "type" {
			keys = append(keys, key)
		}
	}
	payload, err := stringifyCompletionsJSON(marshalSamplingObject(fields, keys))
	if err != nil {
		return err
	}
	started := false
	start := func() {
		if !started {
			started = true
			onStart()
		}
	}
	err = runCodexWebSocketRequest(ctx, lease.socket, payload, idle, func(raw json.RawMessage) (bool, error) {
		if options.OnProviderStreamEvent != nil {
			if err := options.OnProviderStreamEvent(ctx, &raw, rawModel); err != nil {
				return false, &codexProviderCallbackError{err}
			}
		}
		return acc.chunkStarted(raw, start)
	})
	if err != nil {
		return err
	}
	var streamFailure string
	acc.stream.Synchronize(func() { streamFailure = acc.streamFailure() })
	if streamFailure != "" {
		return errors.New(streamFailure)
	}
	if ctx.Err() != nil {
		keep = false
		return nil
	}
	if cached && lease.entry != nil && acc.output.ResponseID != nil && *acc.output.ResponseID != "" {
		include := false
		items, err := ConvertResponsesMessages(rawModel, NormalizeContext(Context{Messages: []Message{*acc.output}}), []string{"openai", "openai-codex", "opencode"}, ResponsesMessagesOptions{IncludeSystemPrompt: &include, GrammarToolInputProperties: acc.grammar})
		if err != nil {
			return err
		}
		var decoded []json.RawMessage
		if err = json.Unmarshal(items, &decoded); err != nil {
			return err
		}
		filtered := []json.RawMessage{}
		for _, item := range decoded {
			fields, _ := samplingObject(item)
			kind := samplingString(fields["type"])
			if kind != "function_call_output" && kind != "custom_tool_call_output" {
				filtered = append(filtered, item)
			}
		}
		lease.entry.continuation = &codexWebSocketContinuation{LastRequestBody: append(json.RawMessage(nil), body...), LastResponseID: *acc.output.ResponseID, LastResponseItems: filtered}
	}
	return nil
}
