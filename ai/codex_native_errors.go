package ai

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var codexTerminalRateLimit = regexp.MustCompile(`(?i)GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota|out of budget|quota exceeded|billing`)
var codexTransientError = regexp.MustCompile(`(?i)rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused`)
var codexUsageLimit = regexp.MustCompile(`(?i)usage_limit_reached|usage_not_included|rate_limit_exceeded`)

func codexRetryableError(status int, text string) bool {
	if status == 429 && codexTerminalRateLimit.MatchString(text) {
		return false
	}
	switch status {
	case 429, 500, 502, 503, 504:
		return true
	}
	return codexTransientError.MatchString(text)
}

func codexRetryAfter(headers http.Header, now time.Time) (float64, bool) {
	if values, exists := headers[http.CanonicalHeaderKey("retry-after-ms")]; exists {
		millis := ParseJSNumber(strings.Join(values, ", "))
		if !math.IsNaN(millis) && !math.IsInf(millis, 0) {
			return math.Max(0, millis), true
		}
	}
	raw := headers.Get("retry-after")
	if raw == "" {
		return 0, false
	}
	seconds := ParseJSNumber(raw)
	if !math.IsNaN(seconds) && !math.IsInf(seconds, 0) {
		return math.Max(0, seconds*1000), true
	}
	if date, err := http.ParseTime(raw); err == nil {
		return math.Max(0, float64(date.UnixMilli()-now.UnixMilli())), true
	}
	return 0, false
}

type codexRetryDelayExceededError struct{ Delay, Limit float64 }

func (e *codexRetryDelayExceededError) Error() string {
	return fmt.Sprintf("Server requested %ss retry delay (max: %ss)", strconv.FormatFloat(math.Ceil(e.Delay/1000), 'f', -1, 64), strconv.FormatFloat(math.Ceil(e.Limit/1000), 'f', -1, 64))
}
func codexValidateRetryDelay(delay float64, limit *float64) error {
	maximum := 60000.0
	if limit != nil {
		maximum = *limit
	}
	if maximum > 0 && delay > maximum {
		return &codexRetryDelayExceededError{delay, maximum}
	}
	return nil
}

// FriendlyMessage is separate because Codex prefers it over a provider's raw
// error.message when throwing, but retains both in the error-response parser.
type codexErrorResponse struct {
	Message         string  `json:"message"`
	FriendlyMessage *string `json:"friendlyMessage,omitempty"`
}

func codexParseErrorResponse(raw, statusText string, status int, now time.Time) codexErrorResponse {
	result := codexErrorResponse{Message: raw}
	if result.Message == "" {
		result.Message = statusText
	}
	if result.Message == "" {
		result.Message = "Request failed"
	}
	parsed, _ := samplingObject(json.RawMessage(raw))
	failure, _ := samplingObject(parsed["error"])
	if failure == nil {
		return result
	}
	code := failure["code"]
	if !samplingTruthy(code) {
		code = failure["type"]
	}
	if codexUsageLimit.MatchString(completionsErrorString(code)) || status == 429 {
		plan := ""
		if samplingTruthy(failure["plan_type"]) {
			var name string
			// A non-string plan makes the source's toLowerCase throw; its catch
			// leaves the original raw response untouched.
			if json.Unmarshal(failure["plan_type"], &name) != nil {
				return result
			}
			plan = " (" + strings.ToLower(name) + " plan)"
		}
		when := ""
		if samplingTruthy(failure["resets_at"]) {
			reset := ParseJSNumber(completionsErrorString(failure["resets_at"]))
			minutes := math.Max(0, math.Floor((reset*1000-float64(now.UnixMilli()))/60000+0.5))
			when = " Try again in ~" + strconv.FormatFloat(minutes, 'f', -1, 64) + " min."
		}
		friendly := "You have hit your ChatGPT usage limit" + plan + "." + when
		result.FriendlyMessage = &friendly
	}
	if samplingTruthy(failure["message"]) {
		result.Message = completionsErrorString(failure["message"])
	} else if result.FriendlyMessage != nil {
		result.Message = *result.FriendlyMessage
	}
	return result
}

// codexHTTPError preserves transport metadata for the host account pool while
// retaining Pi's exact user-visible error message.
type codexHTTPError struct {
	Status  int
	Headers http.Header
	Message string
}

func (e *codexHTTPError) Error() string { return e.Message }
