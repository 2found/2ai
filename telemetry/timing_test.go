package telemetry_test

import (
	"errors"
	"testing"
	"time"

	"github.com/2found/2ai/telemetry"
)

func TestSpanTimingExcludesActiveAndPreservesDetachedSnapshots(t *testing.T) {
	recorder := telemetry.NewInMemory()
	failure := errors.New("original failure")
	var snapshot []telemetry.SpanTiming
	err := recorder.StartSpan(telemetry.SpanOptions{Name: "root"}, func(root *telemetry.Span) error {
		if len(recorder.GetSpanTimings()) != 0 {
			t.Fatal("active span reported as completed")
		}
		if err := root.StartSpan(telemetry.SpanOptions{Name: "child"}, func(*telemetry.Span) error { return failure }); err != failure {
			t.Fatal("failure replaced")
		}
		snapshot = recorder.GetSpanTimings()
		if len(snapshot) != 1 || snapshot[0].SpanID != 2 {
			t.Fatalf("active root was timed: %+v", snapshot)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	final := recorder.GetSpanTimings()
	if len(final) != 2 || len(snapshot) != 1 || final[0].SpanID != 1 || final[1].SpanID != 2 || final[0].Duration < final[1].Duration || final[0].StartedAt.After(final[1].StartedAt) {
		t.Fatalf("invalid per-span timings: %+v / %+v", final, snapshot)
	}
	snapshot[0].Duration = -time.Hour
	if recorder.GetSpanTimings()[1].Duration < 0 {
		t.Fatal("snapshot edit changed recorder")
	}
	var absent *telemetry.InMemory
	if len(absent.GetSpanTimings()) != 0 {
		t.Fatal("nil recorder produced timing")
	}
}
