package export_test

import (
	"context"
	"errors"
	"testing"

	"github.com/2found/2ai/telemetry"
	"github.com/2found/2ai/telemetry/export"
)

func TestBatchKeepsParentageAndCallbackFailure(t *testing.T) {
	failure := errors.New("work failed")
	var batches []export.Batch
	parent := export.New(func(batch export.Batch) { batches = append(batches, batch); panic("broken sink") })
	ctx := telemetry.WithContext(context.Background(), parent)
	err := telemetry.FromContext(ctx).StartSpan(telemetry.SpanOptions{Name: "run"}, func(root *telemetry.Span) error {
		return root.StartSpan(telemetry.SpanOptions{Name: "request"}, func(child *telemetry.Span) error {
			child.AddEvent("attempt", telemetry.NewAttributes(telemetry.Property{Name: "input", Value: "opaque"}))
			return failure
		})
	})
	if err != failure || len(batches) != 1 || len(batches[0].Spans) != 2 {
		t.Fatalf("callback/delivery changed: %v %+v", err, batches)
	}
	root, child := batches[0].Spans[0], batches[0].Spans[1]
	if !root.Settled || !child.Settled || child.ParentID == nil || *child.ParentID != root.ID || root.Status.Status != "error" || child.Status.Status != "error" {
		t.Fatalf("invalid settled hierarchy: %+v", batches[0])
	}
	if len(child.Events) != 1 || batches[0].Duration < 0 {
		t.Fatal("missing attempt/timing")
	}
}

func TestCallbackPanicSurvivesSinkPanic(t *testing.T) {
	workPanic := &struct{ why string }{"original panic"}
	delivered := false
	parent := export.New(func(batch export.Batch) {
		delivered = true
		if !batch.Spans[0].Settled {
			t.Error("exported before settlement")
		}
		panic("sink panic")
	})
	defer func() {
		if got := recover(); got != workPanic || !delivered {
			t.Errorf("panic replaced/delivery lost: %v %v", got, delivered)
		}
	}()
	_ = parent.StartSpan(telemetry.SpanOptions{Name: "run"}, func(*telemetry.Span) error { panic(workPanic) })
}
