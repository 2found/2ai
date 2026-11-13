package export_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/core/telemetry"
	"github.com/2found/2ai/core/telemetry/export"
)

type collector struct {
	mu      sync.Mutex
	batches [][]export.Record
}

func (c *collector) sink(_ context.Context, batch []export.Record) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.batches = append(c.batches, batch)
	return nil
}

func (c *collector) records() []export.Record {
	c.mu.Lock()
	defer c.mu.Unlock()
	var all []export.Record
	for _, batch := range c.batches {
		all = append(all, batch...)
	}
	return all
}

// Export used to happen once, synchronously, when the root span ended, so a
// crash mid-run lost the whole trace and a slow sink stalled the run.
func TestExporterStreamsSettledSpansBeforeTheRootEnds(t *testing.T) {
	ctx := context.Background()
	var c collector
	exporter := export.NewExporter(export.Options{Sink: c.sink, BatchSize: 2, Interval: time.Hour})
	defer exporter.Close(ctx)
	err := exporter.Context().StartSpan(telemetry.SpanOptions{Name: "run"}, func(root *telemetry.Span) error {
		for range 5 {
			_ = root.StartSpan(telemetry.SpanOptions{Name: "request"}, func(span *telemetry.Span) error {
				span.AddEvent("attempt", telemetry.NewAttributes())
				return nil
			})
		}
		if err := exporter.Flush(ctx); err != nil {
			return err
		}
		if got := c.records(); len(got) != 5 || got[0].Span.Name != "request" || got[0].Timing.StartedAt.IsZero() || len(got[0].Timing.EventTimes) != 1 {
			t.Errorf("children not exported while the root runs: %+v", got)
		}
		return errors.New("run failed")
	})
	if err == nil || exporter.Flush(ctx) != nil {
		t.Fatal(err)
	}
	_ = exporter.Context().StartSpan(telemetry.SpanOptions{Name: "second"}, func(*telemetry.Span) error { return nil })
	if err := exporter.Close(ctx); err != nil {
		t.Fatal(err)
	}
	records := c.records()
	if len(records) != 7 || records[5].Span.Name != "run" || records[5].Span.Status.Status != "error" || records[6].Span.Name != "second" {
		t.Fatalf("records %+v", records)
	}
	if len(records[0].TraceID) != 32 || records[0].TraceID != records[5].TraceID || records[6].TraceID == records[0].TraceID {
		t.Fatal("trace ids", records[0].TraceID, records[5].TraceID, records[6].TraceID)
	}
	for _, batch := range c.batches {
		if len(batch) > 2 {
			t.Fatal("batch over BatchSize", len(batch))
		}
	}
	if stats := exporter.Stats(); stats.Exported != 7 || stats.Dropped != 0 {
		t.Fatalf("stats %+v", stats)
	}
	// After Close, roots run unrecorded and nothing blocks.
	ran := false
	_ = exporter.Context().StartSpan(telemetry.SpanOptions{Name: "late"}, func(*telemetry.Span) error { ran = true; return nil })
	if !ran || len(c.records()) != 7 || exporter.Flush(ctx) != nil || exporter.Close(ctx) != nil {
		t.Fatal("late root")
	}
}

func TestExporterNeverBlocksRecordingOnASlowSink(t *testing.T) {
	ctx := context.Background()
	release := make(chan struct{})
	entered := make(chan struct{}, 1)
	exporter := export.NewExporter(export.Options{Queue: 1, BatchSize: 1, Interval: time.Hour, Sink: func(context.Context, []export.Record) error {
		select {
		case entered <- struct{}{}:
		default:
		}
		<-release
		return nil
	}})
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = exporter.Context().StartSpan(telemetry.SpanOptions{Name: "run"}, func(root *telemetry.Span) error {
			_ = root.StartSpan(telemetry.SpanOptions{Name: "first"}, func(*telemetry.Span) error { return nil })
			<-entered // the sink now holds the exporter goroutine
			for range 10 {
				_ = root.StartSpan(telemetry.SpanOptions{Name: "child"}, func(*telemetry.Span) error { return nil })
			}
			return nil
		})
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("recording blocked on the sink")
	}
	close(release)
	if err := exporter.Close(ctx); err != nil {
		t.Fatal(err)
	}
	stats := exporter.Stats()
	if stats.Dropped < 9 || stats.Exported+stats.Dropped != 12 {
		t.Fatalf("stats %+v", stats)
	}
}

func TestExporterSamplesHeartbeatsAndCountsFailures(t *testing.T) {
	ctx := context.Background()
	var c collector
	var errs []error
	var mu sync.Mutex
	failing := true
	exporter := export.NewExporter(export.Options{Heartbeat: 5 * time.Millisecond, Interval: time.Hour, Limits: telemetry.Limits{Events: 1},
		OnError: func(err error) { mu.Lock(); errs = append(errs, err); mu.Unlock() },
		Sink: func(ctx context.Context, batch []export.Record) error {
			mu.Lock()
			fail := failing
			mu.Unlock()
			if fail {
				panic("sink down")
			}
			return c.sink(ctx, batch)
		}})
	_ = exporter.Context().StartSpan(telemetry.SpanOptions{Name: "lost"}, func(*telemetry.Span) error { return nil })
	if exporter.Flush(ctx) != nil || exporter.Stats().Failed != 1 || len(errs) != 1 || !strings.Contains(errs[0].Error(), "sink down") {
		t.Fatalf("sink failure not counted: %+v %v", exporter.Stats(), errs)
	}
	mu.Lock()
	failing = false
	mu.Unlock()
	// A long root is visible through heartbeats before it ends.
	_ = exporter.Context().StartSpan(telemetry.SpanOptions{Name: "long"}, func(root *telemetry.Span) error {
		root.AddEvent("one", telemetry.NewAttributes())
		root.AddEvent("two", telemetry.NewAttributes())
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			_ = exporter.Flush(ctx)
			for _, record := range c.records() {
				if record.Active && record.Span.Name == "long" && record.Timing.Duration > 0 && !record.Span.Settled {
					return nil
				}
			}
			time.Sleep(5 * time.Millisecond)
		}
		t.Error("no heartbeat for a running root")
		return nil
	})
	// Head sampling decides per root; skipped roots still run.
	sampled := export.NewExporter(export.Options{Sink: c.sink, Sample: export.SampleRatio(0)})
	ran := false
	_ = sampled.Context().StartSpan(telemetry.SpanOptions{Name: "skipped"}, func(span *telemetry.Span) error { ran = true; return nil })
	_ = sampled.Close(ctx)
	if !ran || sampled.Stats().Unsampled != 1 || sampled.Stats().Exported != 0 {
		t.Fatalf("sampling %+v", sampled.Stats())
	}
	if err := exporter.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if stats := exporter.Stats(); stats.Recorder.Events != 1 {
		t.Fatalf("recorder drops not reported: %+v", stats)
	}
}

func TestOTLPSinkPostsTraceJSON(t *testing.T) {
	ctx := context.Background()
	var body []byte
	var header http.Header
	status := http.StatusOK
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		header = r.Header
		w.WriteHeader(status)
	}))
	defer server.Close()
	sink := export.OTLP(server.URL+"/v1/traces", "agent", map[string]string{"Authorization": "Bearer t"}, nil)
	var records []export.Record
	exporter := export.NewExporter(export.Options{Interval: time.Hour, Sink: func(_ context.Context, batch []export.Record) error {
		records = append(records, batch...)
		return nil
	}})
	_ = exporter.Context().StartSpan(telemetry.SpanOptions{Name: "run"}, func(root *telemetry.Span) error {
		return root.StartSpan(telemetry.SpanOptions{Name: "request"}, func(span *telemetry.Span) error {
			span.SetAttributes(telemetry.NewAttributes(
				telemetry.Property{Name: "gen_ai.usage.input_tokens", Value: int64(130)},
				telemetry.Property{Name: "gen_ai.response.finish_reasons", Value: []string{"stop"}},
				telemetry.Property{Name: "ai.cost_usd", Value: 0.5},
				telemetry.Property{Name: "ai.output_committed", Value: true},
			))
			span.AddEvent("attempt", telemetry.NewAttributes(telemetry.Property{Name: "ai.attempt", Value: 1}))
			return errors.New("provider failed")
		})
	})
	_ = exporter.Close(ctx)
	heartbeat := export.Record{TraceID: records[0].TraceID, Span: records[1].Span, Active: true}
	if err := sink(ctx, append(records, heartbeat)); err != nil {
		t.Fatal(err)
	}
	var sent struct {
		ResourceSpans []struct {
			Resource   struct{ Attributes []map[string]any }
			ScopeSpans []struct {
				Spans []struct {
					TraceID, SpanID, ParentSpanID, Name, StartTimeUnixNano, EndTimeUnixNano string
					Attributes                                                              []struct {
						Key   string
						Value map[string]any
					}
					Events []struct{ TimeUnixNano, Name string }
					Status struct {
						Code    int
						Message string
					}
				}
			}
		}
	}
	if err := json.Unmarshal(body, &sent); err != nil || header.Get("Authorization") != "Bearer t" || header.Get("Content-Type") != "application/json" {
		t.Fatal(err, header)
	}
	spans := sent.ResourceSpans[0].ScopeSpans[0].Spans
	if len(spans) != 2 || !strings.Contains(string(body), `"service.name","value":{"stringValue":"agent"}`) {
		t.Fatalf("heartbeat exported or resource missing: %s", body)
	}
	request, run := spans[0], spans[1]
	if request.TraceID != records[0].TraceID || len(request.SpanID) != 16 || request.ParentSpanID != run.SpanID || run.ParentSpanID != "" ||
		request.Status.Code != 2 || request.Status.Message != "provider failed" || run.Status.Code != 2 || request.StartTimeUnixNano > request.EndTimeUnixNano ||
		len(request.Events) != 1 || request.Events[0].TimeUnixNano == "" {
		t.Fatalf("span shape: %s", body)
	}
	values := map[string]string{}
	for _, attribute := range request.Attributes {
		raw, _ := json.Marshal(attribute.Value)
		values[attribute.Key] = string(raw)
	}
	if values["gen_ai.usage.input_tokens"] != `{"intValue":"130"}` || values["gen_ai.response.finish_reasons"] != `{"arrayValue":{"values":[{"stringValue":"stop"}]}}` ||
		values["ai.cost_usd"] != `{"doubleValue":0.5}` || values["ai.output_committed"] != `{"boolValue":true}` {
		t.Fatalf("attribute values %v", values)
	}
	status = http.StatusServiceUnavailable
	if err := sink(ctx, records); err == nil || !strings.Contains(err.Error(), "503") {
		t.Fatal("collector rejection not reported", err)
	}
	if err := sink(ctx, []export.Record{heartbeat}); err != nil {
		t.Fatal("a heartbeat-only batch should send nothing", err)
	}
}
