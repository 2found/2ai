package ai

import (
	"encoding/json"
	"time"

	"github.com/2found/2ai/telemetry"
)

// AttemptTrace records detached JSON after producer settlement. It never
// relays events or feeds a display projection back into the native engine.
type AttemptTrace struct {
	span    *telemetry.Span
	started time.Time
	model   json.RawMessage
	view    json.RawMessage
}

func NewAttemptTrace(span *telemetry.Span) *AttemptTrace { return &AttemptTrace{span: span} }

// Start snapshots the request immediately before opening an attempt.
func (t *AttemptTrace) Start(model json.RawMessage, view TranscriptContext) {
	t.started = time.Now()
	t.model = append(json.RawMessage(nil), model...)
	t.view, _ = json.Marshal(view)
}

// Finish records one settled attempt, including admission and terminal errors.
func (t *AttemptTrace) Finish(attempt FallbackAttempt) {
	if t.started.IsZero() {
		return // host preparation failed before provider admission
	}
	durationMS := time.Since(t.started).Milliseconds()
	trace := map[string]any{
		"model": t.model, "context": t.view, "response": attempt.Outcome.Message(),
		"startedAtMs": t.started.UnixMilli(), "durationMs": durationMS,
		"attempt": map[string]any{"number": attempt.Number},
	}
	if attempt.Failure != nil {
		trace["error"] = map[string]any{"name": "Error", "message": attempt.Failure.Error()}
	}
	properties := []telemetry.Property{
		{Name: "ai.attempt", Value: attempt.Number},
		{Name: "ai.duration_ms", Value: durationMS},
		{Name: "ai.failure_kind", Value: boundedFailureKind(attempt.FailureKind)},
		{Name: "ai.output_committed", Value: attempt.Outcome.Committed},
	}
	// Detached native payloads may be malformed. Their serialization must never
	// suppress bounded operational attempt/failure metadata.
	if raw, err := json.Marshal(trace); err == nil {
		properties = append(properties, telemetry.Property{Name: "llm.trace", Value: string(raw)})
	}
	t.span.AddEvent("agentray.ai.attempt", telemetry.NewAttributes(properties...))
	t.started = time.Time{}
}
