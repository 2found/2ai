package protocol

import (
	"net/http"
	"strconv"
	"testing"
	"time"
)

func TestRetryPolicyHonorsServerWaitOrStops(t *testing.T) {
	policy := DefaultRetryPolicy()
	for _, hint := range []time.Duration{time.Millisecond, policy.MaxDelay, policy.MaxDelay + time.Millisecond, time.Minute} {
		delay, retry := policy.NextDelay(1, &ProviderError{Status: 429, Message: "rate limit", RetryAfter: hint})
		if hint > policy.MaxDelay {
			if retry || delay != 0 {
				t.Fatalf("hint %v retried early: %v/%v", hint, delay, retry)
			}
		} else if !retry || delay != hint {
			t.Fatalf("hint %v not honored: %v/%v", hint, delay, retry)
		}
	}
}

func TestBusyMessagesRetryWithoutOverridingAccountLimits(t *testing.T) {
	for _, message := range []string{"server_busy", "servers are currently busy", "SERVERS ARE CURRENTLY BUSY"} {
		if !IsRetryable(&ProviderError{Status: 200, Message: message}) {
			t.Fatalf("wrapped busy failure not retryable: %q", message)
		}
		if IsRetryable(&ProviderError{Status: 429, Message: "insufficient_quota: " + message}) {
			t.Fatalf("busy wording overrode exhausted quota: %q", message)
		}
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Unix(2_000_000_000, 0)
	if got := parseRetryAfterAt("5", now); got != 5*time.Second {
		t.Fatalf("parseRetryAfter(5) = %v, want 5s", got)
	}
	if got := parseRetryAfterAt("", now); got != 0 {
		t.Fatalf("parseRetryAfter(empty) = %v, want 0", got)
	}
	if got := parseRetryAfterAt("garbage", now); got != 0 {
		t.Fatalf("parseRetryAfter(garbage) = %v, want 0", got)
	}
}

func TestNewProviderErrorUsesLongestRetryHint(t *testing.T) {
	now := time.Unix(2_000_000_000, 0)
	h := http.Header{
		"Retry-After":          []string{"1"},
		"Retry-After-Ms":       []string{"2500"},
		"X-RateLimit-Reset-Ms": []string{"4200"},
		"X-RateLimit-Reset":    []string{"3"},
	}
	if got := parseRetryHeaders(h, now); got != 4200*time.Millisecond {
		t.Fatalf("parseRetryHeaders = %v, want 4.2s", got)
	}

	epoch := now.Add(7 * time.Second).Unix()
	h = http.Header{"X-RateLimit-Reset": []string{strconv.FormatInt(epoch, 10)}}
	if got := parseRetryHeaders(h, now); got != 7*time.Second {
		t.Fatalf("epoch reset = %v, want 7s", got)
	}
}
