package ai

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/telemetry"
	"github.com/2found/2ai/telemetry/export"
)

func TestAttemptMetadataSurvivesMalformedPrivateTrace(t *testing.T) {
	for _, malformed := range []bool{false, true} {
		var batch export.Batch
		backend := export.New(func(value export.Batch) { batch = value }, telemetry.Limits{Payloads: 4})
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
		if malformed && (attributes.Get("llm.trace") != nil || attributes.Get("llm.trace.dropped") != "error" || batch.Dropped.Payloads != 1) {
			t.Fatal("malformed private trace emitted or not counted", batch.Dropped)
		}
		if !malformed && attributes.Get("llm.trace") == nil {
			t.Fatal("valid detached trace lost")
		}
	}
}

// Request spans used to carry no queryable attributes, and every attempt
// serialized the full request and response even when nothing kept them.
func TestAttemptTraceRecordsGenAIAttributesAndOptInPayloads(t *testing.T) {
	reasoning := 3.0
	answer := &Message{Role: "assistant", Provider: "anthropic", Model: "claude-x-20260101", StopReason: "stop", Usage: &Usage{Input: 10, Output: 5, CacheRead: 100, CacheWrite: 20, Reasoning: &reasoning, Cost: UsageCost{Total: 0.5}}}
	model := json.RawMessage(`{"id":"claude-x","provider":"anthropic","contextWindow":200000}`)
	run := func(limits telemetry.Limits, attempts ...FallbackAttempt) (export.Batch, []*AttemptTrace) {
		var batch export.Batch
		var traces []*AttemptTrace
		_ = export.New(func(value export.Batch) { batch = value }, limits).StartSpan(telemetry.SpanOptions{Name: "agentray.ai.request"}, func(span *telemetry.Span) error {
			for _, attempt := range attempts {
				trace := NewAttemptTrace(span)
				trace.Start(model, NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent("hi")}}}))
				traces = append(traces, &AttemptTrace{view: trace.view})
				if attempt.Outcome.Committed {
					attempt.Outcome.FirstOutput = time.Now()
				}
				trace.Finish(attempt)
			}
			return nil
		})
		return batch, traces
	}
	failed := FallbackAttempt{Number: 1, Failure: errors.New("busy"), FailureKind: "provider_unavailable", Outcome: AttemptOutcome{Terminal: AssistantMessageEvent{Type: "error", Error: &Message{Role: "assistant", StopReason: "error"}}}}
	served := FallbackAttempt{Number: 1, Rung: 1, FailureKind: "none", Outcome: AttemptOutcome{Committed: true, Terminal: AssistantMessageEvent{Type: "done", Message: answer}}}
	batch, traces := run(telemetry.Limits{}, failed, served)
	if len(batch.Spans) != 1 || len(batch.Spans[0].Events) != 2 {
		t.Fatal(batch.Spans)
	}
	want := map[string]any{
		"gen_ai.operation.name": "chat", "gen_ai.provider.name": "anthropic", "gen_ai.request.model": "claude-x", "gen_ai.response.model": "claude-x-20260101",
		"gen_ai.usage.input_tokens": int64(130), "gen_ai.usage.output_tokens": int64(5), "gen_ai.usage.cache_read.input_tokens": int64(100),
		"gen_ai.usage.cache_creation.input_tokens": int64(20), "gen_ai.usage.reasoning.output_tokens": int64(3), "ai.cost_usd": 0.5,
		"ai.context_window": int64(200000), "ai.fallback.rung": 1, "ai.failure_kind": "none", "ai.output_committed": true,
	}
	span, first, second := batch.Spans[0].Attributes, batch.Spans[0].Events[0].Attributes, batch.Spans[0].Events[1].Attributes
	for name, value := range want {
		if span.Get(name) != value || second.Get(name) != value {
			t.Errorf("%s: span %v event %v, want %v", name, span.Get(name), second.Get(name), value)
		}
	}
	if reasons, _ := span.Get("gen_ai.response.finish_reasons").([]string); len(reasons) != 1 || reasons[0] != "stop" {
		t.Error("finish reasons", span.Get("gen_ai.response.finish_reasons"))
	}
	if ttft, ok := span.Get("ai.time_to_first_token_ms").(int64); !ok || ttft < 0 {
		t.Error("time to first token", span.Get("ai.time_to_first_token_ms"))
	}
	// The failed attempt says why on its event; the span keeps the final
	// attempt's values with nothing stale from the retry.
	if first.Get("error.type") != "provider_unavailable" || first.Get("gen_ai.usage.input_tokens") != int64(0) || span.Get("error.type") != nil || second.Get("error.type") != nil {
		t.Error("failure attributes", first.Get("error.type"), span.Get("error.type"))
	}
	// Payload capture is off by default: nothing was serialized or kept.
	if first.Get("llm.trace") != nil || first.Get("llm.trace.dropped") != nil || traces[0].view != nil || batch.Dropped != (telemetry.Drops{}) {
		t.Error("payload captured without opting in")
	}
	// Opted in, the count bound keeps the first payload and counts the rest.
	batch, traces = run(telemetry.Limits{Payloads: 1}, served, served)
	events := batch.Spans[0].Events
	if raw, _ := events[0].Attributes.Get("llm.trace").(string); !strings.Contains(raw, `"claude-x-20260101"`) || traces[0].view == nil {
		t.Error("opted-in payload missing", raw)
	}
	if events[1].Attributes.Get("llm.trace") != nil || events[1].Attributes.Get("llm.trace.dropped") != "count" || traces[1].view != nil || batch.Dropped.Payloads != 1 {
		t.Error("count bound", events[1].Attributes.Get("llm.trace.dropped"), batch.Dropped)
	}
	// And the size bound drops a payload too large to keep.
	batch, _ = run(telemetry.Limits{Payloads: 4, PayloadBytes: 16}, served)
	if attributes := batch.Spans[0].Events[0].Attributes; attributes.Get("llm.trace") != nil || attributes.Get("llm.trace.dropped") != "size" || batch.Dropped.Payloads != 1 {
		t.Error("size bound", attributes.Get("llm.trace.dropped"), batch.Dropped)
	}
}
