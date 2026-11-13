package protocol

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ProviderError is the typed error a provider returns for an HTTP-level failure,
// so the loop can classify it (retryable status? honor Retry-After?) rather than
// string-matching a bare error (pi's isRetryableError, made structural where a
// status exists). A nil/zero Status marks a transport-level failure with no HTTP
// response.
type ProviderError struct {
	Provider   string        // provider name ("openai", "anthropic")
	Status     int           // HTTP status; 0 for a transport failure
	RetryAfter time.Duration // longest parsed retry/reset header; 0 if absent
	Message    string        // server-supplied detail
}

func (e *ProviderError) Error() string {
	if e.Status > 0 {
		if e.Message != "" {
			return fmt.Sprintf("%s: unexpected response (status %d): %s", e.Provider, e.Status, e.Message)
		}
		return fmt.Sprintf("%s: unexpected response (status %d)", e.Provider, e.Status)
	}
	if e.Message != "" {
		return fmt.Sprintf("%s: %s", e.Provider, e.Message)
	}
	return e.Provider + ": provider error"
}

// NewProviderError builds a ProviderError from an HTTP response, parsing the
// standard and provider-specific retry/reset headers so the loop can pace a
// 429/503 backoff to the server's longest hint.
//
// It is exported because it is the contract between a provider and the loop's
// retry/escalation logic: IsRetryable classifies structurally first (status
// code, Retry-After), falling back to the message only where the status carries
// no signal. A provider that returns a plain error is silently un-retryable, so
// any provider — in this repo or out of it — builds its transport failures with
// this.
func NewProviderError(provider string, resp *http.Response, message string) *ProviderError {
	pe := &ProviderError{Provider: provider, Message: message}
	if resp != nil {
		pe.Status = resp.StatusCode
		pe.RetryAfter = parseRetryHeaders(resp.Header, time.Now())
	}
	return pe
}

// parseRetryAfterAt reads delay-seconds or an HTTP date relative to now.
// An unparseable or absent value yields 0.
func parseRetryAfterAt(v string, now time.Time) time.Duration {
	if v == "" {
		return 0
	}
	if secs, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil {
		if secs <= 0 {
			return 0
		}
		return durationFromFloat(secs, time.Second)
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := t.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// parseRetryHeaders accepts the retry hints used by the major provider and
// gateway families. When more than one is present the longest delay wins: a
// short generic Retry-After must not override a more precise rate-limit reset.
func parseRetryHeaders(h http.Header, now time.Time) time.Duration {
	if h == nil {
		return 0
	}
	candidates := []time.Duration{
		parseMilliseconds(headerValue(h, "Retry-After-Ms")),
		parseRetryAfterAt(headerValue(h, "Retry-After"), now),
		parseRateLimitReset(headerValue(h, "X-RateLimit-Reset-Ms"), true, now),
		parseRateLimitReset(headerValue(h, "X-RateLimit-Reset"), false, now),
	}
	var longest time.Duration
	for _, candidate := range candidates {
		if candidate > longest {
			longest = candidate
		}
	}
	return longest
}

func headerValue(h http.Header, name string) string {
	if value := h.Get(name); value != "" {
		return value
	}
	for key, values := range h {
		if strings.EqualFold(key, name) && len(values) > 0 {
			return values[0]
		}
	}
	return ""
}

func parseMilliseconds(v string) time.Duration {
	n, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil || n <= 0 {
		return 0
	}
	return durationFromFloat(n, time.Millisecond)
}

// parseRateLimitReset accepts either a relative delta or an epoch timestamp.
// Compat gateways are inconsistent about units, so values above 1e12 are epoch
// milliseconds and values above 1e9 are epoch seconds, matching OMP's proven
// interoperability rule; smaller values are relative in the header's unit.
func parseRateLimitReset(v string, milliseconds bool, now time.Time) time.Duration {
	n, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil || n <= 0 {
		return 0
	}
	if n > 1e12 {
		return positiveDuration(time.UnixMilli(int64(n)).Sub(now))
	}
	if n > 1e9 {
		return positiveDuration(time.Unix(int64(n), 0).Sub(now))
	}
	if milliseconds {
		return durationFromFloat(n, time.Millisecond)
	}
	return durationFromFloat(n, time.Second)
}

func durationFromFloat(value float64, unit time.Duration) time.Duration {
	// time.Duration conversion truncates; add one sub-unit so a server's decimal
	// hint is never honored for less time than requested.
	d := time.Duration(math.Ceil(value * float64(unit)))
	if d <= 0 {
		return 0
	}
	return d
}

func positiveDuration(d time.Duration) time.Duration {
	if d > 0 {
		return d
	}
	return 0
}

// exhaustedLimitPattern marks an account-level limit that no amount of retrying
// will clear: a spent quota, an empty balance, a billing problem. These arrive
// wearing the SAME 429 as an ordinary throttle, so status alone cannot tell them
// apart and a status-only classifier burns the whole ladder on a failure that is
// permanent until a human tops the account up. Checked before anything else, so
// it also overrides a retryable status. (pi's
// NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.)
var exhaustedLimitPattern = regexp.MustCompile(
	`(?i)insufficient_quota|quota exceeded|out of budget|billing|` +
		`monthly usage limit reached|available balance|usage limit`)

// retryableMessagePattern catches transient failures whose status code is
// useless — a gateway that wraps an upstream blip in a 200, or a transport error
// with no HTTP response at all. Deliberately narrower than pi's list: it covers
// the wordings actually observed against this stack rather than every provider
// pi supports, because a false positive here retries something permanent.
var retryableMessagePattern = regexp.MustCompile(
	`(?i)overloaded|rate.?limit|too many requests|service.?unavailable|` +
		`server_busy|servers are currently busy|` +
		`too many concurren|concurren(?:cy|t).{0,20}limit|` +
		`server.?error|internal.?error|provider.?returned.?error|` +
		`connection reset|connection refused|other side closed|fetch failed|` +
		`socket hang up|stream ended before|ended without|truncated`)

// retryableStatus reports whether an HTTP status is worth another attempt.
//
// Beyond the standard transient set this includes two ranges that a status-only
// switch misses. 520-524 are Cloudflare's origin-side codes (520 unknown error,
// 521 origin down, 522 timeout, 523 unreachable, 524 origin timeout): the edge
// is healthy and answering, so the failure is upstream of it and typically
// brief. 529 is Anthropic's "overloaded". Omitting these is not academic — the
// idle-game benchmark took a 521 and gave up without a single retry.
func retryableStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout,
		529: // Anthropic overloaded_error
		return true
	}
	// Cloudflare origin-side range.
	return status >= 520 && status <= 524
}

// IsRetryable reports whether err is a transient failure worth retrying the same
// model: a rate limit (429), a server-side 5xx (500/502/503/504), a Cloudflare
// origin error (520-524), an Anthropic overload (529), a request timeout (408),
// or a transport-level network blip. A cancellation is never retryable — the
// caller guards on ctx.Err() — and a client error (4xx other than 408/429) won't
// be fixed by retrying, so it falls through to escalation.
//
// An exhausted quota or a billing failure is never retryable even when it wears
// a retryable status, because the account, not the network, is what is broken.
//
// Exported alongside NewProviderError as the other half of the provider↔loop
// retry contract: a provider builds the error, the loop classifies it, and an
// out-of-tree provider can assert its own transport failures land on the right
// side of that line.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	var pe *ProviderError
	if errors.As(err, &pe) {
		// An account limit is permanent until a human acts on it; it outranks
		// both the status and the transient-wording check below.
		if exhaustedLimitPattern.MatchString(pe.Message) {
			return false
		}
		if pe.Status == 0 {
			return true // transport failure captured as a ProviderError
		}
		if retryableStatus(pe.Status) {
			return true
		}
		// The status was unhelpful (a gateway 200 wrapping an upstream failure,
		// say); fall back to what the provider actually said.
		return retryableMessagePattern.MatchString(pe.Message)
	}
	// Transport-level failures (connection reset, DNS, timeouts) arrive as
	// *url.Error / net.Error from the HTTP client; these are worth a retry.
	var ne net.Error
	if errors.As(err, &ne) {
		return true
	}
	var ue *url.Error
	if errors.As(err, &ue) {
		return true
	}
	return false
}

// retryAfterOf extracts a server-supplied Retry-After from a provider error, if
// any, so the backoff can honor it instead of the exponential schedule.
func retryAfterOf(err error) time.Duration {
	var pe *ProviderError
	if errors.As(err, &pe) {
		return pe.RetryAfter
	}
	return 0
}

// RetryPolicy bounds the same-model retry of a transient provider failure before
// the loop escalates down the model ladder. It is per-rung: each rung gets its
// own attempt budget, so a flaky rung is retried in place (cheap) before paying
// to escalate to a pricier one.
type RetryPolicy struct {
	MaxAttempts int           // total attempts per rung, including the first (>=1)
	BaseDelay   time.Duration // first backoff; doubles each attempt
	MaxDelay    time.Duration // cap on any single backoff
}

// DefaultRetryPolicy is a conservative same-rung backoff: three attempts with
// exponential delay capped at a few seconds, enough to ride out a brief 429/503
// without stalling a run.
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{MaxAttempts: 3, BaseDelay: 500 * time.Millisecond, MaxDelay: 8 * time.Second}
}

// NextDelay applies the host's same-rung retry contract to a completed attempt
// (numbered from one). Native stream orchestration uses the same normalization,
// error classification, Retry-After limit and jitter as callRung. A server hint
// above MaxDelay stops same-model retry rather than retrying before it permits.
func (rp RetryPolicy) NextDelay(failedAttempt int, failure error) (time.Duration, bool) {
	rp = rp.Normalized()
	if failedAttempt < 1 || failedAttempt >= rp.MaxAttempts || !IsRetryable(failure) {
		return 0, false
	}
	retryAfter := retryAfterOf(failure)
	if retryAfter > rp.MaxDelay {
		return 0, false
	}
	return rp.delay(failedAttempt-1, retryAfter), true
}

// Normalized fills any zero field from the default so a partial override is safe.
func (rp RetryPolicy) Normalized() RetryPolicy {
	d := DefaultRetryPolicy()
	if rp.MaxAttempts <= 0 {
		rp.MaxAttempts = d.MaxAttempts
	}
	if rp.BaseDelay <= 0 {
		rp.BaseDelay = d.BaseDelay
	}
	if rp.MaxDelay <= 0 {
		rp.MaxDelay = d.MaxDelay
	}
	return rp
}

// delay computes the backoff before attempt n (0-based: the wait before the
// first retry uses n=0). A server Retry-After wins when present; otherwise it is
// exponential with equal jitter, capped at MaxDelay.
func (rp RetryPolicy) delay(n int, retryAfter time.Duration) time.Duration {
	if retryAfter > 0 {
		return retryAfter
	}
	d := rp.BaseDelay << n
	if d <= 0 || d > rp.MaxDelay {
		d = rp.MaxDelay
	}
	// Equal jitter: half the window fixed, half random, to de-correlate retries.
	half := d / 2
	return half + time.Duration(rand.Int63n(int64(half)+1))
}
