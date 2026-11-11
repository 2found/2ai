package ai

import (
	"encoding/json"
	"errors"
	"time"

	"github.com/2found/2ai/telemetry"
)

// AttemptTrace records one model attempt after producer settlement. It never
// relays events or feeds a display projection back into the native engine.
//
// Every attempt event, and the request span it belongs to, carries queryable
// OpenTelemetry GenAI attributes (gen_ai.*) plus 2ai's ai.* attempt metadata.
// The full request and response ("llm.trace") is an opt-in payload: it is
// serialized only when the recorder captures payloads (telemetry.Limits).
type AttemptTrace struct {
	span    *telemetry.Span
	started time.Time
	model   json.RawMessage
	view    json.RawMessage
}

func NewAttemptTrace(span *telemetry.Span) *AttemptTrace { return &AttemptTrace{span: span} }

// Start marks the attempt's start immediately before it opens. The request
// is snapshotted only when the recorder would keep it.
func (t *AttemptTrace) Start(model json.RawMessage, view TranscriptContext) {
	t.started = time.Now()
	t.model = append(json.RawMessage(nil), model...)
	t.view = nil
	if t.span.CapturesPayloads() {
		t.view, _ = json.Marshal(view)
	}
}

// Finish records one settled attempt, including admission and terminal errors.
func (t *AttemptTrace) Finish(attempt FallbackAttempt) {
	if t.started.IsZero() {
		return // host preparation failed before provider admission
	}
	started, model, view := t.started, t.model, t.view
	t.started, t.view = time.Time{}, nil
	durationMS := time.Since(started).Milliseconds()
	message := attempt.Outcome.Message()
	properties := attemptAttributes(model, message, attempt)
	var ttft int64
	if !attempt.Outcome.FirstOutput.IsZero() {
		ttft = attempt.Outcome.FirstOutput.Sub(started).Milliseconds()
	}
	properties = append(properties,
		telemetry.Property{Name: "ai.duration_ms", Value: durationMS},
		telemetry.Property{Name: "ai.time_to_first_token_ms", Value: ttft},
	)
	// The request span keeps the latest attempt's values; every key is set
	// each time so a retry never leaves a stale value behind.
	t.span.SetAttributes(telemetry.NewAttributes(properties...))
	if attempt.Failure != nil {
		properties = append(properties, telemetry.Property{Name: "error.type", Value: boundedFailureKind(attempt.FailureKind)})
	}
	// Detached native payloads may be malformed. Their serialization must never
	// suppress bounded operational attempt/failure metadata.
	t.span.AddEventWithPayload("agentray.ai.attempt", telemetry.NewAttributes(properties...), "llm.trace", func() ([]byte, error) {
		if view == nil {
			return nil, errors.New("request was not captured")
		}
		trace := map[string]any{
			"model": model, "context": view, "response": message,
			"startedAtMs": started.UnixMilli(), "durationMs": durationMS,
			"attempt": map[string]any{"number": attempt.Number},
		}
		if attempt.Failure != nil {
			trace["error"] = map[string]any{"name": "Error", "message": attempt.Failure.Error()}
		}
		return json.Marshal(trace)
	})
}

// attemptAttributes are the bounded, queryable facts about one attempt.
// Token counts follow the GenAI conventions: input_tokens is the whole
// prompt, cached reads and writes included; reasoning is part of output.
func attemptAttributes(model json.RawMessage, message *Message, attempt FallbackAttempt) []telemetry.Property {
	var requested struct {
		ID, Provider  string
		ContextWindow int64 `json:"contextWindow"`
	}
	_ = json.Unmarshal(model, &requested)
	provider, responseModel, reasons := requested.Provider, "", []string{}
	var input, output, cacheRead, cacheWrite, reasoning int64
	var cost float64
	if message != nil {
		if provider == "" {
			provider = message.Provider
		}
		responseModel = message.Model
		if message.StopReason != "" {
			reasons = []string{message.StopReason}
		}
		if usage := message.Usage; usage != nil {
			cacheRead, cacheWrite = int64(usage.CacheRead), int64(usage.CacheWrite)
			input, output = int64(usage.Input)+cacheRead+cacheWrite, int64(usage.Output)
			if usage.Reasoning != nil {
				reasoning = int64(*usage.Reasoning)
			}
			cost = usage.Cost.Total
		}
	}
	return []telemetry.Property{
		{Name: "gen_ai.operation.name", Value: "chat"},
		{Name: "gen_ai.provider.name", Value: provider},
		{Name: "gen_ai.request.model", Value: requested.ID},
		{Name: "gen_ai.response.model", Value: responseModel},
		{Name: "gen_ai.response.finish_reasons", Value: reasons},
		{Name: "gen_ai.usage.input_tokens", Value: input},
		{Name: "gen_ai.usage.output_tokens", Value: output},
		{Name: "gen_ai.usage.cache_read.input_tokens", Value: cacheRead},
		{Name: "gen_ai.usage.cache_creation.input_tokens", Value: cacheWrite},
		{Name: "gen_ai.usage.reasoning.output_tokens", Value: reasoning},
		{Name: "ai.cost_usd", Value: cost},
		{Name: "ai.context_window", Value: requested.ContextWindow},
		{Name: "ai.attempt", Value: attempt.Number},
		{Name: "ai.fallback.rung", Value: attempt.Rung},
		{Name: "ai.failure_kind", Value: boundedFailureKind(attempt.FailureKind)},
		{Name: "ai.output_committed", Value: attempt.Outcome.Committed},
	}
}
