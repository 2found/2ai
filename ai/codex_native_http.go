package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/klauspost/compress/zstd"
)

type CodexResponsesStreamOptions = OpenAICompletionsStreamOptions

// StreamCodexResponsesSSE executes Pi's explicit SSE transport in Go. The
// combined transport selector is available through StreamCodexResponses.
// Credentials are a per-call OAuth access token in apiKey.
func StreamCodexResponsesSSE(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions) *AssistantMessageEventStream {
	return streamCodexResponses(ctx, rawModel, transcript, options, true)
}
func streamCodexResponses(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions, sseOnly bool) *AssistantMessageEventStream {
	return streamCodexResponsesObserved(ctx, rawModel, transcript, options, sseOnly, nil)
}
func streamCodexResponsesObserved(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions, sseOnly bool, onFailure func(error)) *AssistantMessageEventStream {
	return streamCodexResponsesInto(ctx, rawModel, transcript, options, sseOnly, onFailure, NewAssistantMessageEventStreamFor(ctx))
}
func streamCodexResponsesInto(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions, sseOnly bool, onFailure func(error), stream *AssistantMessageEventStream) *AssistantMessageEventStream {
	rawModel = append(json.RawMessage(nil), rawModel...)
	options.Options = append(json.RawMessage(nil), options.Options...)
	var model completionsModel
	prepareErr := json.Unmarshal(rawModel, &model)
	encoded, err := json.Marshal(transcript)
	if prepareErr == nil {
		prepareErr = err
	}
	var frozen Context
	if err = json.Unmarshal(encoded, &frozen); prepareErr == nil {
		prepareErr = err
	}
	transcript = ResolveTranscript(NormalizeContext(frozen), samplingTruthy(model.Compat["supportsMidConvoSystemMessages"]))
	now := time.Now().UnixMilli()
	if options.Now != nil {
		now = options.Now()
	}
	options, callbackFailure := nativeFailureCallbacks(options)
	recordNativeFailure(ctx, model.Provider, nil, false)
	acc := newCodexResponsesAccumulator(model, nil, stream, now)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				failure := fmt.Errorf("%v", recovered)
				recordNativeFailure(ctx, model.Provider, failure, true)
				if onFailure != nil {
					onFailure(failure)
				}
				acc.fail(failure.Error(), ctx.Err() != nil)
			}
		}()
		err := prepareErr
		if err == nil {
			prepared, prepareErr := prepareCodexRequest(ctx, acc, rawModel, transcript, options)
			err = prepareErr
			if err == nil {
				if sseOnly {
					err = runCodexSSEPrepared(ctx, acc, rawModel, options, defaultCompletionsRetryTiming(), prepared)
				} else {
					err = runCodexCombined(ctx, acc, rawModel, options, prepared)
				}
			}
		}
		if err != nil {
			if ctx.Err() != nil {
				err = errors.New("Request was aborted")
			}
			if onFailure != nil {
				onFailure(err)
			}
			recordNativeFailure(ctx, model.Provider, err, *callbackFailure)
			acc.fail(err.Error(), ctx.Err() != nil)
		}
	}()
	return stream
}

func runCodexSSEPrepared(ctx context.Context, acc *codexResponsesAccumulator, rawModel json.RawMessage, options CodexResponsesStreamOptions, timing completionsRetryTiming, prepared *codexPreparedRequest) error {
	controls, params, headers, timeout := prepared.controls, prepared.body, prepared.sseHeaders.Clone(), prepared.timeout
	// One compression per logical request: retries send identical bytes.
	encoder, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(3)), zstd.WithEncoderCRC(false), zstd.WithEncoderConcurrency(1))
	if err != nil {
		return err
	}
	body := encoder.EncodeAll(params, nil)
	_ = encoder.Close()
	headers.Set("Content-Encoding", "zstd")
	client := options.Client
	if client == nil {
		client = http.DefaultClient
	}
	retries := 0.0
	_ = json.Unmarshal(controls["maxRetries"], &retries)
	var maxDelay *float64
	_ = json.Unmarshal(controls["maxRetryDelayMs"], &maxDelay)
	if options.Now != nil {
		timing.now = func() time.Time { return time.UnixMilli(options.Now()) }
	}
	var response *http.Response
	var lastError error
	for attempt := 0.0; attempt <= retries; attempt++ {
		if ctx.Err() != nil {
			return errors.New("Request was aborted")
		}
		delay := 1000 * math.Pow(2, attempt)
		var retryStatus bool
		response, lastError = func() (*http.Response, error) {
			requestCtx, cancel := context.WithCancel(ctx)
			req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, codexNativeURL(acc.model.BaseURL), bytes.NewReader(body))
			if err != nil {
				cancel()
				return nil, err
			}
			req.Header = headers.Clone()
			var timer *time.Timer
			if timeout > 0 {
				timer = time.AfterFunc(time.Duration(timeout)*time.Millisecond, cancel)
			}
			response, err := client.Do(req)
			if timer != nil {
				timer.Stop()
			}
			if err != nil {
				timedOut := ctx.Err() == nil && requestCtx.Err() != nil
				cancel()
				if timedOut {
					return nil, fmt.Errorf("Codex SSE response headers timed out after %.0fms", timeout)
				}
				return nil, err
			}
			response.Body = &completionsHTTPBody{ReadCloser: response.Body, cancel: cancel, ctx: requestCtx}
			keep := false
			defer func() {
				if !keep {
					_ = response.Body.Close()
				}
			}()
			if options.OnResponse != nil {
				observed := CompletionsResponse{Status: response.StatusCode, Headers: map[string]string{}}
				for name, values := range response.Header {
					observed.Headers[strings.ToLower(name)] = strings.Join(values, ", ")
				}
				if err := options.OnResponse(ctx, observed, rawModel); err != nil {
					return nil, &codexProviderCallbackError{err}
				}
			}
			if response.StatusCode >= 200 && response.StatusCode < 300 {
				keep = true
				return response, nil
			}
			data, err := io.ReadAll(response.Body)
			if err != nil {
				return nil, err
			}
			if attempt < retries && codexRetryableError(response.StatusCode, string(data)) {
				if requested, ok := codexRetryAfter(response.Header, timing.now()); ok {
					if err := codexValidateRetryDelay(requested, maxDelay); err != nil {
						return nil, err
					}
					delay = requested
				}
				retryStatus = true
				return nil, nil
			}
			statusText := strings.TrimSpace(strings.TrimPrefix(response.Status, fmt.Sprint(response.StatusCode)))
			parsed := codexParseErrorResponse(string(data), statusText, response.StatusCode, timing.now())
			if parsed.FriendlyMessage != nil {
				return nil, &codexHTTPError{Status: response.StatusCode, Headers: response.Header.Clone(), Message: *parsed.FriendlyMessage}
			}
			return nil, &codexHTTPError{Status: response.StatusCode, Headers: response.Header.Clone(), Message: parsed.Message}
		}()
		if response != nil {
			break
		}
		if ctx.Err() != nil {
			return errors.New("Request was aborted")
		}
		var exceeded *codexRetryDelayExceededError
		if !retryStatus && (attempt >= retries || errors.As(lastError, &exceeded) || (lastError != nil && strings.Contains(lastError.Error(), "usage limit"))) {
			return lastError
		}
		// JS timers clamp delays outside their signed 32-bit range.
		if delay > 2147483647 || math.IsNaN(delay) {
			delay = 1
		}
		if err := timing.sleep(ctx, time.Duration(math.Max(0, delay))*time.Millisecond); err != nil {
			return err
		}
	}
	if response == nil {
		if lastError != nil {
			return lastError
		}
		return errors.New("Failed after retries")
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNoContent || response.StatusCode == http.StatusResetContent {
		return errors.New("No response body")
	}
	acc.start()
	if err := readCodexSSE(ctx, response.Body, func(raw json.RawMessage) (bool, error) {
		if options.OnProviderStreamEvent != nil {
			if err := options.OnProviderStreamEvent(ctx, &raw, rawModel); err != nil {
				return false, &codexProviderCallbackError{err}
			}
		}
		return acc.chunk(raw)
	}); err != nil {
		return err
	}
	acc.finish(ctx)
	return nil
}
