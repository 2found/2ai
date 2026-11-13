package ai

import (
	"context"
	"math"
	"time"
)

func oauthCancelCause(ctx context.Context) error {
	cause := context.Cause(ctx)
	if cause == context.Canceled {
		name := "AbortError"
		return &OAuthDiagnosticError{Name: &name, Message: "The operation was aborted.", Code: 20}
	}
	if cause == context.DeadlineExceeded {
		name := "TimeoutError"
		return &OAuthDiagnosticError{Name: &name, Message: "The operation timed out.", Code: 23}
	}
	return cause
}
func oauthSleep(ms float64, ctx context.Context) error {
	if ctx.Err() != nil {
		return oauthCancelCause(ctx)
	}
	if math.IsNaN(ms) || ms < 1 || ms > math.MaxInt32 {
		ms = 1
	}
	timer := time.NewTimer(time.Duration(math.Floor(ms)) * time.Millisecond)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return oauthCancelCause(ctx)
	case <-timer.C:
		return nil
	}
}
