package telemetrytest_test

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/2found/2ai/telemetry"
	telemetrytest "github.com/2found/2ai/telemetry/testing"
)

// Exercise the public callback backend seam as well as the native recorder.
func forwardingContext(parent telemetry.Context) telemetry.Context {
	wrap := func(callback func(*telemetry.Span) error) func(*telemetry.Span) error {
		return func(span *telemetry.Span) error {
			return callback(telemetry.NewSpan(forwardingContext(span.Context()), telemetry.SpanCallbacks{
				AddEvent: span.AddEvent, SetAttributes: span.SetAttributes, SetStatus: span.SetStatus,
				AddEventFrom: span.AddEventFrom, SetAttributesFrom: span.SetAttributesFrom, SetStatusFrom: span.SetStatusFrom,
			}))
		}
	}
	return telemetry.NewContextWithCallbacks(telemetry.ContextCallbacks{
		StartSpan: func(options telemetry.SpanOptions, callback func(*telemetry.Span) error) error {
			return parent.StartSpan(options, wrap(callback))
		},
		StartSpanFrom: func(read func() telemetry.SpanOptions, callback func(*telemetry.Span) error) error {
			return parent.StartSpanFrom(read, wrap(callback))
		},
	})
}

func TestAdapterConformance(t *testing.T) {
	for _, forward := range []bool{false, true} {
		name := "memory"
		if forward {
			name = "callback-backend"
		}
		t.Run(name, func(t *testing.T) {
			created, closed := 0, 0
			suite := telemetrytest.CreateAdapterConformance(func() (*telemetrytest.AdapterFixture, error) {
				created++
				recorder := telemetry.NewInMemory()
				parent := recorder.Context
				if forward {
					parent = forwardingContext(parent)
				}
				return &telemetrytest.AdapterFixture{Context: parent, GetSpans: func() ([]telemetry.RecordedSpan, error) { return recorder.GetSpans(), nil }, Close: func() error { closed++; return nil }}, nil
			})
			for _, test := range suite {
				t.Run(test.Group+"/"+test.Name, func(t *testing.T) {
					if err := test.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
			if created != 10 || closed != created {
				t.Fatalf("fixture isolation/cleanup: created=%d closed=%d", created, closed)
			}
			// Preserve all nine source case names. Deferred status reads model
			// the original throwing property access, separately from readable
			// status-name normalization in the supplemental case.
			data, err := os.ReadFile("../testdata/pi-schema.json")
			if err != nil {
				t.Fatal(err)
			}
			type namePair struct{ Group, Name string }
			var fixture struct{ Conformance []namePair }
			if err := json.Unmarshal(data, &fixture); err != nil {
				t.Fatal(err)
			}
			got := []namePair{}
			for _, test := range suite {
				got = append(got, namePair{test.Group, test.Name})
			}
			if len(fixture.Conformance) != 9 {
				t.Fatal("unexpected upstream conformance coverage")
			}
			if len(got) != 10 || !reflect.DeepEqual(got[:9], fixture.Conformance) || got[9] != (namePair{"status", "normalizes readable status names and preserves explicit precedence"}) {
				t.Fatal("conformance coverage differs from upstream")
			}
		})
	}
}

func TestConformanceRejectsBadAdapterAndCleansUp(t *testing.T) {
	cleanupFailure := errors.New("cleanup failure")
	closed := 0
	suite := telemetrytest.CreateAdapterConformance(func() (*telemetrytest.AdapterFixture, error) {
		return &telemetrytest.AdapterFixture{GetSpans: func() ([]telemetry.RecordedSpan, error) { return []telemetry.RecordedSpan{}, nil }, Close: func() error { closed++; return cleanupFailure }}, nil
	})
	err := suite[0].Run()
	if err == nil || !strings.Contains(err.Error(), "expected recorded span success") || !errors.Is(err, cleanupFailure) || closed != 1 {
		t.Fatalf("failure/cleanup lost: %v closed=%d", err, closed)
	}
	factoryFailure := errors.New("factory failure")
	suite = telemetrytest.CreateAdapterConformance(func() (*telemetrytest.AdapterFixture, error) { return nil, factoryFailure })
	if err := suite[0].Run(); err != factoryFailure {
		t.Fatalf("factory error changed: %v", err)
	}
}
