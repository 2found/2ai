package telemetry

import (
	"errors"
	"testing"
)

// The in-memory recorder used to grow without bound for the life of a run.
func TestInMemoryLimitsCountWhatTheyDrop(t *testing.T) {
	recorder := NewInMemory(Limits{Spans: 2, Events: 2})
	ran := 0
	_ = recorder.StartSpan(SpanOptions{Name: "root"}, func(root *Span) error {
		for range 3 {
			root.AddEvent("tick", NewAttributes())
		}
		for range 2 {
			_ = root.StartSpan(SpanOptions{Name: "child"}, func(*Span) error { ran++; return nil })
		}
		return nil
	})
	spans, timings := recorder.GetSpans(), recorder.GetSpanTimings()
	if ran != 2 || len(spans) != 2 || len(spans[0].Events) != 2 || len(timings[0].EventTimes) != 2 || timings[0].EventTimes[0].IsZero() {
		t.Fatalf("ran=%d spans=%+v timings=%+v", ran, spans, timings)
	}
	if got := recorder.Dropped(); got != (Drops{Spans: 1, Events: 1}) {
		t.Fatalf("drops %+v", got)
	}
}

func TestPayloadsAreOptInLazyAndBounded(t *testing.T) {
	calls := 0
	payload := func(raw string, err error) func() ([]byte, error) {
		return func() ([]byte, error) { calls++; return []byte(raw), err }
	}
	record := func(limits Limits, add func(*Span)) (RecordedSpan, Drops) {
		recorder := NewInMemory(limits)
		_ = recorder.StartSpan(SpanOptions{Name: "request"}, func(span *Span) error { add(span); return nil })
		return recorder.GetSpans()[0], recorder.Dropped()
	}
	// Off by default: the payload is never produced and nothing is counted.
	span, drops := record(Limits{}, func(span *Span) {
		if span.CapturesPayloads() {
			t.Error("captures without opting in")
		}
		span.AddEventWithPayload("attempt", NewAttributes(Property{Name: "n", Value: 1}), "body", payload("x", nil))
	})
	if calls != 0 || span.Events[0].Attributes.Get("body") != nil || span.Events[0].Attributes.Get("n") != 1 || drops != (Drops{}) {
		t.Fatalf("disabled payload: calls=%d %+v %+v", calls, span.Events, drops)
	}
	// Opted in: kept within bounds; a failing or panicking producer releases
	// its slot, an oversized one is dropped, and the count bound stops the rest.
	span, drops = record(Limits{Payloads: 2, PayloadBytes: 4}, func(span *Span) {
		span.AddEventWithPayload("a", NewAttributes(), "body", payload("", errors.New("bad")))
		span.AddEventWithPayload("b", NewAttributes(), "body", func() ([]byte, error) { panic("boom") })
		span.AddEventWithPayload("c", NewAttributes(), "body", payload("too long", nil))
		span.AddEventWithPayload("d", NewAttributes(), "body", payload("ok", nil))
		span.AddEventWithPayload("e", NewAttributes(), "body", payload("ok", nil))
		if span.CapturesPayloads() {
			t.Error("count bound reached but still capturing")
		}
		span.AddEventWithPayload("f", NewAttributes(), "body", payload("ok", nil))
	})
	want := []struct{ body, dropped any }{{nil, "error"}, {nil, "error"}, {nil, "size"}, {"ok", nil}, {"ok", nil}, {nil, "count"}}
	for i, event := range span.Events {
		if event.Attributes.Get("body") != want[i].body || event.Attributes.Get("body.dropped") != want[i].dropped {
			t.Errorf("event %s: %v %v", event.Name, event.Attributes.Get("body"), event.Attributes.Get("body.dropped"))
		}
	}
	if calls != 4 || drops.Payloads != 4 {
		t.Fatalf("calls=%d drops=%+v", calls, drops)
	}
	// Callback backends and settled spans never produce the payload.
	var delivered []Attributes
	callbacks := NewSpan(Context{}, SpanCallbacks{AddEvent: func(_ string, attributes Attributes) { delivered = append(delivered, attributes) }})
	callbacks.AddEventWithPayload("a", NewAttributes(Property{Name: "n", Value: 1}), "body", payload("x", nil))
	var settled *Span
	_ = NewInMemory(Limits{Payloads: 1}).StartSpan(SpanOptions{Name: "s"}, func(span *Span) error { settled = span; return nil })
	settled.AddEventWithPayload("late", NewAttributes(), "body", payload("x", nil))
	if calls != 4 || len(delivered) != 1 || delivered[0].Get("body") != nil || settled.CapturesPayloads() {
		t.Fatal("payload produced for a backend or settled span", calls, delivered)
	}
}

func TestStreamHandsOffSettledSpansAndForgetsThem(t *testing.T) {
	var got []RecordedSpan
	var stream *InMemory
	stream = NewStream(func(span RecordedSpan, timing SpanTiming) {
		if timing.SpanID != span.ID || timing.StartedAt.IsZero() || !span.Settled {
			t.Error("settled span without its timing", span, timing)
		}
		got = append(got, span)
		panic("sink panic is ignored")
	}, Limits{Spans: 2})
	_ = stream.StartSpan(SpanOptions{Name: "root"}, func(root *Span) error {
		// Only live spans count against the bound, so a long run of
		// sequential children is never refused.
		for range 5 {
			_ = root.StartSpan(SpanOptions{Name: "child"}, func(child *Span) error {
				child.AddEvent("work", NewAttributes())
				return nil
			})
		}
		if live := stream.GetSpans(); len(live) != 1 || live[0].Name != "root" {
			t.Error("settled spans retained", live)
		}
		return errors.New("failed")
	})
	if len(got) != 6 || got[5].Name != "root" || got[5].Status.Status != "error" || stream.Dropped() != (Drops{}) {
		t.Fatalf("%+v %+v", got, stream.Dropped())
	}
	for i, span := range got[:5] {
		if span.ID != i+2 || *span.ParentID != 1 || len(span.Events) != 1 {
			t.Fatalf("child %d: %+v", i, span)
		}
	}
	if len(stream.GetSpans()) != 0 {
		t.Fatal("root retained after settling")
	}
}
