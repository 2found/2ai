package protocol

import (
	"net/http"
	"strconv"
	"testing"
	"time"
)

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
