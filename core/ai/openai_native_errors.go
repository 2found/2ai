package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf16"
)

// Request errors retain the fields used by Pi's retry and display helpers.
// A zero status denotes a connection/stream error rather than an HTTP reply.
type completionsRequestError struct {
	status  int
	headers http.Header
	body    json.RawMessage
	message string
}

func (e *completionsRequestError) Error() string { return e.message }

func newCompletionsRequestError(status int, body json.RawMessage, fallback string, headers http.Header) *completionsRequestError {
	fields, _ := samplingObject(body)
	message := fallback
	if samplingTruthy(fields["message"]) {
		message = samplingString(fields["message"])
		if len(fields["message"]) > 0 && fields["message"][0] != '"' {
			encoded, _ := stringifyCompletionsJSON(fields["message"])
			message = string(encoded)
		}
	} else if samplingTruthy(body) {
		encoded, _ := stringifyCompletionsJSON(body)
		message = string(encoded)
	}
	if status != 0 {
		if message != "" {
			message = fmt.Sprintf("%d %s", status, message)
		} else {
			message = fmt.Sprintf("%d status code (no body)", status)
		}
	} else if message == "" {
		message = "(no status code or body)"
	}
	return &completionsRequestError{status: status, headers: headers, body: body, message: message}
}

func formatCompletionsError(err error) string {
	return formatOpenAIProviderError(err, "", true)
}
func formatResponsesError(err error, provider string) string {
	if provider == "openai" {
		provider = "OpenAI"
	}
	return formatOpenAIProviderError(err, provider+" API error", false)
}
func formatOpenAIProviderError(err error, prefix string, appendRawMetadata bool) string {
	message := err.Error()
	var provider *completionsRequestError
	if !errors.As(err, &provider) {
		return message
	}
	fields, ok := samplingObject(provider.body)
	if ok && len(fields) > 0 {
		encoded, _ := stringifyCompletionsJSON(provider.body)
		body := strings.TrimFunc(string(encoded), jsWhitespace)
		units := utf16.Encode([]rune(body))
		if len(units) > 4000 {
			body = string(utf16.Decode(units[:4000])) + fmt.Sprintf("... [truncated %d chars]", len(units)-4000)
		}
		if provider.status != 0 && !strings.Contains(message, body) {
			if prefix != "" {
				message = body
			} else {
				message = fmt.Sprintf("%d: %s", provider.status, body)
			}
		}
	}
	if prefix != "" && provider.status != 0 {
		message = fmt.Sprintf("%s (%d): %s", prefix, provider.status, message)
	}
	metadata, _ := samplingObject(fields["metadata"])
	if appendRawMetadata && samplingTruthy(metadata["raw"]) {
		raw := completionsErrorString(metadata["raw"])
		if !strings.Contains(message, raw) {
			message += "\n" + raw
		}
	}
	return message
}

func completionsErrorString(raw json.RawMessage) string {
	if _, ok := samplingObject(raw); ok {
		return "[object Object]"
	}
	var array []json.RawMessage
	if json.Unmarshal(raw, &array) == nil && array != nil {
		parts := make([]string, len(array))
		for i, item := range array {
			if samplingNonNull(item) {
				parts[i] = completionsErrorString(item)
			}
		}
		return strings.Join(parts, ",")
	}
	if len(raw) > 0 && raw[0] == '"' {
		return samplingString(raw)
	}
	encoded, _ := stringifyCompletionsJSON(raw)
	return string(encoded)
}

type completionsRetryTiming struct {
	now    func() time.Time
	random func() float64
	sleep  func(context.Context, time.Duration) error
}

func defaultCompletionsRetryTiming() completionsRetryTiming {
	return completionsRetryTiming{now: time.Now, random: rand.Float64, sleep: func(ctx context.Context, delay time.Duration) error {
		if ctx.Err() != nil {
			return errors.New("Request aborted")
		}
		timer := time.NewTimer(max(0, delay))
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return errors.New("Request aborted")
		case <-timer.C:
			return nil
		}
	}}
}

var completionsFloatPrefix = regexp.MustCompile(`^[+-]?(?:Infinity|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)`)

func completionsParseFloat(value string) float64 {
	prefix := completionsFloatPrefix.FindString(strings.TrimLeftFunc(value, jsWhitespace))
	if prefix == "" {
		return math.NaN()
	}
	number, _ := strconv.ParseFloat(prefix, 64)
	return number
}

func completionsRetryDelay(err *completionsRequestError, index float64, maxDelay *float64, timing completionsRetryTiming) (time.Duration, error) {
	validate := func(ms float64) (time.Duration, error) {
		limit := 60000.0
		if maxDelay != nil {
			limit = *maxDelay
		}
		if limit > 0 && ms > limit {
			return 0, fmt.Errorf("Server requested %.0fs retry delay (max: %.0fs). %s", math.Ceil(ms/1000), math.Ceil(limit/1000), err.Error())
		}
		// JS timers clamp negative values to zero; durations beyond the timer
		// integer range also fire promptly instead of wrapping a Go duration.
		if ms < 0 {
			ms = 0
		}
		if ms > 2147483647 {
			ms = 1
		}
		return time.Duration(math.Trunc(ms)) * time.Millisecond, nil
	}
	finite := func(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
	if raw := err.headers.Get("retry-after-ms"); raw != "" {
		if value := completionsParseFloat(raw); finite(value) {
			return validate(value)
		}
	}
	if raw := err.headers.Get("retry-after"); raw != "" {
		seconds := completionsParseFloat(raw)
		ms := seconds * 1000
		if math.IsNaN(seconds) {
			if date, parseErr := http.ParseTime(raw); parseErr == nil {
				ms = float64(date.UnixMilli() - timing.now().UnixMilli())
			}
		}
		if finite(ms) {
			return validate(ms)
		}
	}
	ms := math.Min(0.5*math.Pow(2, index), 8) * 1000 * (1 - timing.random()*0.25)
	return time.Duration(math.Trunc(ms)) * time.Millisecond, nil
}

func retryCompletionsRequest(ctx context.Context, request func() (*http.Response, error), retries float64, maxDelay *float64, timing completionsRetryTiming) (*http.Response, error) {
	remaining := retries
	for {
		response, err := request()
		if err == nil {
			return response, nil
		}
		if ctx.Err() != nil {
			return nil, errors.New("Request aborted")
		}
		var provider *completionsRequestError
		if remaining <= 0 || !errors.As(err, &provider) {
			return nil, err
		}
		directive := provider.headers.Get("x-should-retry")
		retry := directive == "true" || directive != "false" && (provider.status == 0 || provider.status == 408 || provider.status == 409 || provider.status == 429 || provider.status >= 500)
		if !retry {
			return nil, err
		}
		delay, err := completionsRetryDelay(provider, retries-remaining, maxDelay, timing)
		if err != nil {
			return nil, err
		}
		remaining--
		if err := timing.sleep(ctx, delay); err != nil {
			return nil, err
		}
	}
}
