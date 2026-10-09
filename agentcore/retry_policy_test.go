package agentcore

import (
	"errors"
	"testing"
	"time"
)

func TestRetryPolicyNextDelaySharesLegacyContract(t *testing.T) {
	failure := &ProviderError{Status: 429, Message: "busy", RetryAfter: time.Hour}
	policy := RetryPolicy{MaxAttempts: 3, BaseDelay: 4 * time.Millisecond, MaxDelay: 10 * time.Millisecond}
	for _, n := range []int{-1, 0, 3, 4} {
		if _, ok := policy.NextDelay(n, failure); ok {
			t.Fatalf("attempt %d retried", n)
		}
	}
	for _, n := range []int{1, 2} {
		if delay, ok := policy.NextDelay(n, failure); ok || delay != 0 {
			t.Fatal("over-limit Retry-After retried early", delay, ok)
		}
	}
	if delay, ok := (RetryPolicy{}).NextDelay(1, failure); ok || delay != 0 {
		t.Fatal("default policy retried over-limit hint", delay, ok)
	}
	failure.RetryAfter = policy.MaxDelay
	if delay, ok := policy.NextDelay(1, failure); !ok || delay != policy.MaxDelay {
		t.Fatal("allowed Retry-After not honored", delay, ok)
	}
	if _, ok := (RetryPolicy{}).NextDelay(3, failure); ok {
		t.Fatal("default attempt budget changed")
	}
	failure.RetryAfter = 0
	for n := 1; n <= 2; n++ {
		delay, ok := policy.NextDelay(n, failure)
		window := policy.BaseDelay << uint(n-1)
		if !ok || delay < window/2 || delay > window {
			t.Fatal("legacy equal jitter changed", n, delay)
		}
	}
	for _, failure := range []error{nil, errors.New("503 busy"), &ProviderError{Status: 403, Message: "denied"}, &ProviderError{Status: 429, Message: "insufficient_quota"}} {
		if _, ok := policy.NextDelay(1, failure); ok {
			t.Fatal("permanent/untyped failure retried", failure)
		}
	}
}
