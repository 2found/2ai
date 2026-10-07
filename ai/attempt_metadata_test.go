package ai

import (
	"encoding/json"
	"errors"
	"github.com/2found/2ai/telemetry"
	"github.com/2found/2ai/telemetry/export"
	"testing"
)

func TestAttemptMetadataSurvivesMalformedPrivateTrace(t *testing.T) {
	for _, malformed := range []bool{false, true} {
		var batch export.Batch
		backend := export.New(func(value export.Batch) { batch = value })
		err := backend.StartSpan(telemetry.SpanOptions{Name: "agentray.ai.request"}, func(span *telemetry.Span) error {
			trace := NewAttemptTrace(span)
			model := json.RawMessage(`{"id":"fixture"}`)
			if malformed {
				model = json.RawMessage(`{"private":"incomplete`)
			}
			trace.Start(model, TranscriptContext{})
			trace.Finish(FallbackAttempt{Number: 1, Failure: errors.New("PRIVATE_PROVIDER_MESSAGE"), FailureKind: "provider_protocol", Outcome: AttemptOutcome{Committed: true}})
			return errors.New("fixture terminal failure")
		})
		if err == nil || len(batch.Spans) != 1 || len(batch.Spans[0].Events) != 1 {
			t.Fatal(err, batch.Spans)
		}
		attributes := batch.Spans[0].Events[0].Attributes
		if attributes.Get("ai.failure_kind") != "provider_protocol" || attributes.Get("ai.output_committed") != true || attributes.Get("ai.attempt") != 1 {
			t.Fatal("safe metadata lost")
		}
		if malformed && attributes.Get("llm.trace") != nil {
			t.Fatal("malformed private trace emitted")
		}
		if !malformed && attributes.Get("llm.trace") == nil {
			t.Fatal("valid detached trace lost")
		}
	}
}
