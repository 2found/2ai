package ai

import (
	"context"
	"sync/atomic"

	"github.com/2found/2ai/telemetry"
)

type safePhysicalKey struct{}
type safePhysicalObservation struct {
	call      telemetry.SafeCall
	labels    telemetry.SafeLabels
	scope     telemetry.SafeOrigin
	delegated atomic.Bool
}

func safePhysicalPoolContext(ctx context.Context) context.Context {
	scope := telemetry.CaptureSafeOrigin(ctx)
	if parent, _ := ctx.Value(safePhysicalKey{}).(*safePhysicalObservation); parent != nil && parent.scope.SameScope(scope) {
		return ctx
	}
	return context.WithValue(ctx, safePhysicalKey{}, &safePhysicalObservation{call: telemetry.NewSafeCall(ctx), scope: scope})
}

func startSafePhysical(ctx context.Context, ordinal int) *telemetry.SafeAttempt {
	if parent, _ := ctx.Value(safePhysicalKey{}).(*safePhysicalObservation); parent != nil && parent.scope.SameScope(telemetry.CaptureSafeOrigin(ctx)) {
		parent.delegated.Store(true)
		return parent.call.StartAttempt(ordinal, parent.labels)
	}
	return telemetry.NewSafeCall(ctx).StartAttempt(ordinal, telemetry.SafeLabels{})
}
func finishSafePhysical(ctx context.Context, attempt *telemetry.SafeAttempt, event AssistantMessageEvent, cause error, host, admitted bool) {
	status, kind := "completed", nativeFailureKind(cause, host)
	message := event.Message
	if event.Type == "error" {
		status, message = "failed", event.Error
		if cause == nil {
			kind = "unknown"
		}
	}
	if ctx.Err() != nil || event.Reason == "aborted" || (message != nil && message.StopReason == "aborted") {
		status, kind = "cancelled", nativeFailureKind(ctx.Err(), false)
		if kind == "none" {
			kind = "request_cancelled"
		}
	}
	finishSafeNative(attempt, status, kind, message, admitted)
}
