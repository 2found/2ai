// Package telemetrytest provides runner-independent conformance cases for Go
// telemetry adapters, ported from Pi's telemetry/testing entry point.
package telemetrytest

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"github.com/2found/2ai/telemetry"
)

// AdapterFixture owns one isolated recorder and its cleanup. GetSpans returns
// normalized snapshots in admission order. Close may be nil for in-memory use.
type AdapterFixture struct {
	Context  telemetry.Context
	GetSpans func() ([]telemetry.RecordedSpan, error)
	Close    func() error
}

type ConformanceCase struct {
	Group string
	Name  string
	Run   func() error
}

// CreateAdapterConformance creates a fresh fixture for each invocation of Run.
// No testing framework or global state is required. The case names correspond
// to Pi's original suite. Error returns/gated goroutines replace Promise
// rejection/settlement; unsupported Go attributes replace unreadable JS proxies.
// SetStatusFrom readers model throwing status property access in Go.
// A supplemental source-verified case covers readable status normalization.
func CreateAdapterConformance(factory func() (*AdapterFixture, error)) []ConformanceCase {
	makeCase := func(group, name string, test func(*AdapterFixture)) ConformanceCase {
		return ConformanceCase{Group: group, Name: name, Run: func() (err error) {
			fixture, err := factory()
			if err != nil {
				return err
			}
			if fixture == nil {
				return errors.New("telemetry conformance: nil fixture")
			}
			defer func() {
				if failure := recover(); failure != nil {
					if assertion, ok := failure.(assertionFailure); ok {
						err = errors.New(string(assertion))
					} else {
						err = fmt.Errorf("telemetry conformance: unexpected panic of type %T", failure)
					}
				}
				if fixture.Close != nil {
					err = errors.Join(err, fixture.Close())
				}
			}()
			test(fixture)
			return nil
		}}
	}
	return []ConformanceCase{
		makeCase("callback lifecycle", "admits once synchronously and preserves the result", func(f *AdapterFixture) {
			expected := &struct{ Value int }{42}
			calls := 0
			result, err := telemetry.StartSpan(f.Context, telemetry.SpanOptions{Name: "success"}, func(*telemetry.Span) (*struct{ Value int }, error) { calls++; return expected, nil })
			check(err == nil && calls == 1 && result == expected, "callback result/identity/admission changed")
			span := find(f, "success")
			equal(span.Status, telemetry.SpanStatus{Status: "ok"})
			check(span.Settled, "success span did not settle")
		}),
		makeCase("callback lifecycle", "preserves synchronous and asynchronous rejection values", func(f *AdapterFixture) {
			syncError := errors.New("sync")
			check(capturePanic(func() {
				_ = f.Context.StartSpan(telemetry.SpanOptions{Name: "sync-error"}, func(*telemetry.Span) error { panic(syncError) })
			}) == syncError, "synchronous panic identity changed")
			asyncError := errors.New("async")
			returned := f.Context.StartSpan(telemetry.SpanOptions{Name: "async-error"}, func(*telemetry.Span) error { return asyncError })
			check(returned == asyncError, "returned error identity changed")
			plain := &struct{ Kind string }{"undefined-equivalent"}
			check(capturePanic(func() {
				_ = f.Context.StartSpan(telemetry.SpanOptions{Name: "undefined-error"}, func(*telemetry.Span) error { panic(plain) })
			}) == plain, "non-error panic identity changed")
			unreadable := &unreadableError{}
			check(capturePanic(func() {
				_ = f.Context.StartSpan(telemetry.SpanOptions{Name: "unreadable-error"}, func(*telemetry.Span) error { panic(unreadable) })
			}) == unreadable, "unreadable panic identity changed")
			returned = f.Context.StartSpan(telemetry.SpanOptions{Name: "async-unreadable-error"}, func(*telemetry.Span) error { return unreadable })
			check(returned == unreadable, "unreadable error identity changed")
			for _, name := range []string{"sync-error", "async-error", "undefined-error", "unreadable-error", "async-unreadable-error"} {
				check(find(f, name).Status.Status == "error", "automatic error status missing")
			}
		}),
		makeCase("status", "uses last explicit status without automatic overwrite", func(f *AdapterFixture) {
			check(f.Context.StartSpan(telemetry.SpanOptions{Name: "last-status"}, func(s *telemetry.Span) error {
				s.SetStatus(telemetry.SpanStatus{Status: "error", Error: &telemetry.ErrorDetails{Name: "Expected", Message: "first"}})
				s.SetStatus(telemetry.SpanStatus{Status: "ok"})
				return nil
			}) == nil, "successful span failed")
			thrown := errors.New("after explicit status")
			check(capturePanic(func() {
				_ = f.Context.StartSpan(telemetry.SpanOptions{Name: "explicit-before-throw"}, func(s *telemetry.Span) error { s.SetStatus(telemetry.SpanStatus{Status: "ok"}); panic(thrown) })
			}) == thrown, "explicit status changed panic")
			rejected := errors.New("after async explicit status")
			explicit := telemetry.SpanStatus{Status: "error", Error: &telemetry.ErrorDetails{Name: "Expected", Message: "async failure"}}
			check(f.Context.StartSpan(telemetry.SpanOptions{Name: "explicit-before-rejection"}, func(s *telemetry.Span) error { s.SetStatus(explicit); return rejected }) == rejected, "explicit status changed returned error")
			expected := telemetry.SpanStatus{Status: "error", Error: &telemetry.ErrorDetails{Name: "Expected", Message: "returned failure"}}
			check(f.Context.StartSpan(telemetry.SpanOptions{Name: "expected-failure"}, func(s *telemetry.Span) error { s.SetStatus(expected); return nil }) == nil, "explicit error status changed success")
			equal(find(f, "last-status").Status, telemetry.SpanStatus{Status: "ok"})
			equal(find(f, "explicit-before-throw").Status, telemetry.SpanStatus{Status: "ok"})
			equal(find(f, "explicit-before-rejection").Status, explicit)
			equal(find(f, "expected-failure").Status, expected)
		}),
		makeCase("recording", "merges attributes and records ordered events", func(f *AdapterFixture) {
			check(f.Context.StartSpan(telemetry.SpanOptions{Name: "recording", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "start", Value: "value"}, telemetry.Property{Name: "overwrite", Value: "start"}, telemetry.Property{Name: "ignored", Value: telemetry.Undefined})}, func(s *telemetry.Span) error {
				s.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "count", Value: 1}, telemetry.Property{Name: "overwrite", Value: "middle"}))
				s.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "count", Value: telemetry.Undefined}, telemetry.Property{Name: "overwrite", Value: "end"}))
				s.AddEvent("first", telemetry.NewAttributes(telemetry.Property{Name: "index", Value: 1}, telemetry.Property{Name: "ignored", Value: telemetry.Undefined}))
				s.AddEvent("second", telemetry.NewAttributes(telemetry.Property{Name: "index", Value: 2}))
				return nil
			}) == nil, "recording changed callback success")
			span := find(f, "recording")
			equal(span.Attributes, telemetry.NewAttributes(telemetry.Property{Name: "start", Value: "value"}, telemetry.Property{Name: "overwrite", Value: "end"}, telemetry.Property{Name: "count", Value: 1}))
			equal(span.Events, []telemetry.RecordedEvent{{Name: "first", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "index", Value: 1})}, {Name: "second", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "index", Value: 2})}})
		}),
		makeCase("recording", "ignores failed attribute calls atomically", func(f *AdapterFixture) {
			check(f.Context.StartSpan(telemetry.SpanOptions{Name: "atomic-attributes", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "retained", Value: "value"})}, func(s *telemetry.Span) error {
				s.SetAttributesFrom(func() telemetry.Attributes { panic("unreadable attribute") })
				return nil
			}) == nil, "invalid attributes changed callback success")
			equal(find(f, "atomic-attributes").Attributes, telemetry.NewAttributes(telemetry.Property{Name: "retained", Value: "value"}))
		}),
		makeCase("recording", "makes calls after settlement inert", func(f *AdapterFixture) {
			var captured *telemetry.Span
			check(f.Context.StartSpan(telemetry.SpanOptions{Name: "settled", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "value", Value: "initial"})}, func(s *telemetry.Span) error { captured = s; return nil }) == nil, "span failed")
			captured.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "value", Value: "late"}))
			captured.AddEvent("late", telemetry.NewAttributes(telemetry.Property{Name: "value", Value: true}))
			captured.SetStatus(telemetry.SpanStatus{Status: "error"})
			reads := 0
			late := func() telemetry.Attributes {
				reads++
				return telemetry.NewAttributes(telemetry.Property{Name: "late", Value: true})
			}
			captured.SetAttributesFrom(late)
			captured.AddEventFrom("late-reader", late)
			check(reads == 0, "settled span evaluated an attribute reader")
			admitted := false
			result, err := telemetry.StartSpanFrom(captured.Context(), func() telemetry.SpanOptions { reads++; return telemetry.SpanOptions{Name: "late-child"} }, func(*telemetry.Span) (int, error) { admitted = true; return 7, nil })
			check(reads == 0 && admitted && result == 7 && err == nil, "late child read options or changed callback result")
			spans := snapshots(f)
			check(len(spans) == 1, "late child created a recorded span")
			equal(spans[0].Attributes, telemetry.NewAttributes(telemetry.Property{Name: "value", Value: "initial"}))
			equal(spans[0].Events, []telemetry.RecordedEvent{})
			equal(spans[0].Status, telemetry.SpanStatus{Status: "ok"})
		}),
		makeCase("parentage", "records nested and concurrent child relationships", func(f *AdapterFixture) {
			check(f.Context.StartSpan(telemetry.SpanOptions{Name: "parent"}, func(parent *telemetry.Span) error {
				entered, release := make(chan struct{}), make(chan struct{})
				completed := make(chan error, 1)
				go func() {
					completed <- parent.StartSpan(telemetry.SpanOptions{Name: "first-child"}, func(*telemetry.Span) error { close(entered); <-release; return nil })
				}()
				<-entered
				result, err := telemetry.StartSpan(parent.Context(), telemetry.SpanOptions{Name: "second-child"}, func(*telemetry.Span) (string, error) { return "done", nil })
				close(release)
				firstErr := <-completed
				check(result == "done" && err == nil && firstErr == nil, "child callback result changed")
				return nil
			}) == nil, "parent failed")
			parent, first, second := find(f, "parent"), find(f, "first-child"), find(f, "second-child")
			check(parent.ParentID == nil && first.ParentID != nil && second.ParentID != nil, "explicit parent IDs missing")
			check(*first.ParentID == parent.ID && *second.ParentID == parent.ID, "child has wrong parent")
			check(first.EndSequence != nil && second.EndSequence != nil && parent.EndSequence != nil, "settlement sequence missing")
			check(*second.EndSequence < *first.EndSequence && *first.EndSequence < *parent.EndSequence, "settlement order changed")
		}),
		makeCase("passivity", "suppresses unreadable telemetry payload failures", func(f *AdapterFixture) {
			calls := 0
			reads := 0
			result, err := telemetry.StartSpanFrom(f.Context, func() telemetry.SpanOptions { reads++; panic("unreadable options") }, func(*telemetry.Span) (int, error) { calls++; return 9, nil })
			check(reads == 1 && calls == 1 && result == 9 && err == nil, "invalid options changed callback result or were not read")
			equal(snapshots(f), []telemetry.RecordedSpan{})
			check(f.Context.StartSpan(telemetry.SpanOptions{Name: "unreadable-recording"}, func(s *telemetry.Span) error {
				reads := 0
				bad := func() telemetry.Attributes { reads++; panic("unreadable attributes") }
				s.SetAttributesFrom(bad)
				s.AddEventFrom("unreadable-event", bad)
				check(reads == 2, "attribute/event readers were not admitted")
				s.SetStatusFrom(func() telemetry.SpanStatus { panic("unreadable status") })
				return nil
			}) == nil, "invalid mutations changed callback result")
			recorded := snapshots(f)
			check(len(recorded) == 1, "invalid mutations created/dropped a span")
			equal(recorded[0].Attributes, telemetry.NewAttributes())
			equal(recorded[0].Events, []telemetry.RecordedEvent{})
			equal(recorded[0].Status, telemetry.SpanStatus{Status: "ok"})
		}),
		makeCase("passivity", "ignores failed status calls atomically", func(f *AdapterFixture) {
			failure := errors.New("rejected after unreadable status")
			reads := 0
			err := f.Context.StartSpan(telemetry.SpanOptions{Name: "unreadable-status"}, func(s *telemetry.Span) error {
				s.SetStatusFrom(func() telemetry.SpanStatus { reads++; panic("unreadable status") })
				return failure
			})
			check(err == failure && reads == 1, "status read changed callback failure or was not admitted")
			check(find(f, "unreadable-status").Status.Status == "error", "failed status read suppressed automatic failure")
		}),
		makeCase("status", "normalizes readable status names and preserves explicit precedence", func(f *AdapterFixture) {
			for _, status := range []string{"", "invalid", "OK"} {
				name := "normalized-" + status
				failure := &unreadableError{}
				details := &telemetry.ErrorDetails{Name: "Explicit", Message: "kept"}
				err := f.Context.StartSpan(telemetry.SpanOptions{Name: name}, func(s *telemetry.Span) error {
					s.SetStatus(telemetry.SpanStatus{Status: status, Error: details})
					details.Message = "changed caller value"
					return failure
				})
				check(err == failure, "normalization changed callback error")
				equal(find(f, name).Status, telemetry.SpanStatus{Status: "error", Error: &telemetry.ErrorDetails{Name: "Explicit", Message: "kept"}})
			}
		}),
	}
}

type assertionFailure string

func check(condition bool, message string) {
	if !condition {
		panic(assertionFailure(message))
	}
}
func equal(actual, expected any) {
	a, err := json.Marshal(actual)
	check(err == nil, "cannot serialize actual snapshot")
	b, err := json.Marshal(expected)
	check(err == nil, "cannot serialize expected snapshot")
	var av, bv any
	check(json.Unmarshal(a, &av) == nil && json.Unmarshal(b, &bv) == nil, "cannot decode snapshots")
	check(reflect.DeepEqual(av, bv), fmt.Sprintf("snapshot differs: got %s; want %s", a, b))
}
func snapshots(f *AdapterFixture) []telemetry.RecordedSpan {
	spans, err := f.GetSpans()
	check(err == nil, "snapshot reader failed")
	return spans
}
func find(f *AdapterFixture, name string) telemetry.RecordedSpan {
	for _, span := range snapshots(f) {
		if span.Name == name {
			return span
		}
	}
	panic(assertionFailure("expected recorded span " + name))
}
func capturePanic(fn func()) (value any) { defer func() { value = recover() }(); fn(); return nil }

type unreadableError struct{}

func (*unreadableError) Error() string { panic("unreadable error") }
