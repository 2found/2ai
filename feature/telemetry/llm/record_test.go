package llm_test

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/core/ai"
	"github.com/2found/2ai/core/ai/protocol"
	"github.com/2found/2ai/core/telemetry"
	"github.com/2found/2ai/core/telemetry/export"
	"github.com/2found/2ai/feature/telemetry/llm"
)

// Trace files hold prompts and responses: they used to be world-readable
// (0644) and write failures vanished silently.
func TestFileSinkIsOwnerOnlyAndReportsWriteFailures(t *testing.T) {
	dir := t.TempDir()
	fresh := filepath.Join(dir, "fresh.jsonl")
	sink, err := llm.NewFileSink(fresh)
	if err != nil {
		t.Fatal(err)
	}
	var errs []error
	sink.OnError = func(err error) { errs = append(errs, err) }
	sink.Record(llm.TraceRecord{Model: "m"})
	// An unencodable record is a loss, counted and reported.
	sink.Record(llm.TraceRecord{Model: "m", Usage: protocol.Usage{CostUSD: math.NaN()}})
	_ = sink.Close()
	// A closed file is a write failure too.
	sink.Record(llm.TraceRecord{Model: "m"})
	if sink.Dropped() != 2 || len(errs) != 2 {
		t.Fatalf("dropped=%d errs=%v", sink.Dropped(), errs)
	}
	raw, _ := os.ReadFile(fresh)
	if strings.Count(string(raw), "\n") != 1 {
		t.Fatalf("file %q", raw)
	}
	existing := filepath.Join(dir, "existing.jsonl")
	if err := os.WriteFile(existing, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	reopened, err := llm.NewFileSink(existing)
	if err != nil {
		t.Fatal(err)
	}
	_ = reopened.Close()
	for _, path := range []string{fresh, existing} {
		if info, err := os.Stat(path); err != nil || info.Mode().Perm() != 0o600 {
			t.Fatalf("%s mode %v %v", path, info.Mode().Perm(), err)
		}
	}
	if _, err := llm.NewFileSink(filepath.Join(dir, "missing", "x.jsonl")); err == nil {
		t.Fatal("open failure hidden")
	}
}

// Payload capture is opt-in, so RecordBatch must still produce cost and
// latency records from an attempt's operational attributes alone.
func TestRecordBatchWithAndWithoutPayloads(t *testing.T) {
	reasoning := 4.0
	answer := &ai.Message{Role: "assistant", Provider: "anthropic", Model: "claude-x", StopReason: "toolUse", Content: ai.TextContent("hi"),
		Usage: &ai.Usage{Input: 10, Output: 6, CacheRead: 100, CacheWrite: 20, Reasoning: &reasoning, Cost: ai.UsageCost{Total: 0.25}}}
	record := func(limits telemetry.Limits, meta llm.Metadata) []llm.TraceRecord {
		var batch export.Batch
		_ = export.New(func(value export.Batch) { batch = value }, limits).StartSpan(telemetry.SpanOptions{Name: "agentray.ai.request"}, func(span *telemetry.Span) error {
			trace := ai.NewAttemptTrace(span)
			trace.Start(json.RawMessage(`{"id":"claude-x","provider":"anthropic"}`), ai.NormalizeContext(ai.Context{Messages: []ai.Message{{Role: "user", Content: ai.TextContent("question")}}}))
			trace.Finish(ai.FallbackAttempt{Number: 1, FailureKind: "none", Outcome: ai.AttemptOutcome{Committed: true, Terminal: ai.AssistantMessageEvent{Type: "done", Message: answer}}})
			trace = ai.NewAttemptTrace(span)
			trace.Start(json.RawMessage(`{"id":"backup","provider":"openai"}`), ai.TranscriptContext{})
			trace.Finish(ai.FallbackAttempt{Number: 1, Rung: 1, Failure: errors.New("PRIVATE"), FailureKind: "provider_unavailable"})
			return nil
		})
		var records []llm.TraceRecord
		llm.RecordBatch(batch, llm.SinkFunc(func(r llm.TraceRecord) { records = append(records, r) }), meta)
		return records
	}
	before := time.Now()
	records := record(telemetry.Limits{}, llm.Metadata{TraceID: "run", PricingKnown: true})
	if len(records) != 2 {
		t.Fatal(records)
	}
	got := records[0]
	if got.Model != "claude-x" || got.Provider != "anthropic" || got.StopReason != "toolUse" || got.TraceID != "run" || got.SessionKey != "run" || len(got.Messages) != 0 || got.Timestamp.Before(before.Add(-time.Second)) {
		t.Fatalf("operational record %+v", got)
	}
	if u := got.Usage; u.InputTokens != 10 || u.OutputTokens != 6 || u.CacheReadTokens != 100 || u.CacheWriteTokens != 20 || u.ReasoningTokens != 4 || u.CostUSD != 0.25 || u.CostUnpriced {
		t.Fatalf("usage %+v", u)
	}
	// A failed attempt reports its bounded kind, never the private message.
	if failed := records[1]; failed.Model != "backup" || failed.Err != "provider_unavailable" || failed.Usage.CostUnpriced {
		t.Fatalf("failed record %+v", failed)
	}
	if unknown := record(telemetry.Limits{}, llm.Metadata{})[0]; !unknown.Usage.CostUnpriced {
		t.Fatal("unknown pricing presented as priced")
	}
	// Opted in, the payload gives the full record, usage subsets included.
	full := record(telemetry.Limits{Payloads: 4}, llm.Metadata{PricingKnown: true})
	if len(full) != 2 || len(full[0].Messages) != 1 || full[0].Messages[0].Content != "question" || full[0].Response != "hi" || full[0].Usage.ReasoningTokens != 4 || full[1].Err != "PRIVATE" {
		t.Fatalf("full record %+v", full)
	}
}
