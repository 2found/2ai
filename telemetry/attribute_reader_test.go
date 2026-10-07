package telemetry_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/telemetry"
)

func TestPiAttributeReaderOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-attribute-readers.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Method, Scope, Phase string
				Fail                 bool
			}
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 64 {
		t.Fatal("unexpected attribute reader coverage")
	}
	for _, tc := range fixture.Cases {
		for _, forwarded := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%s/%s/fail=%v/forwarded=%v", tc.Input.Method, tc.Input.Scope, tc.Input.Phase, tc.Input.Fail, forwarded), func(t *testing.T) {
				recorder := telemetry.NewInMemory()
				parent := recorder.Context
				if tc.Input.Scope == "noop" {
					parent = telemetry.Context{}
				}
				if forwarded {
					original := parent
					parent = telemetry.NewContext(func(options telemetry.SpanOptions, callback func(*telemetry.Span) error) error {
						return original.StartSpan(options, func(span *telemetry.Span) error {
							return callback(telemetry.NewSpan(span.Context(), telemetry.SpanCallbacks{SetStatus: span.SetStatus, AddEvent: span.AddEvent, SetAttributes: span.SetAttributes, SetAttributesFrom: span.SetAttributesFrom, AddEventFrom: span.AddEventFrom}))
						})
					})
				}
				var captured *telemetry.Span
				reads, panicked, callbackError := 0, false, false
				snapshots := [][]telemetry.RecordedSpan{}
				failure := errors.New("callback failed")
				read := func() telemetry.Attributes {
					reads++
					if strings.HasPrefix(tc.Input.Phase, "nested") {
						captured.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "inner", Value: true}, telemetry.Property{Name: "base", Value: "changed"}))
						captured.AddEvent("inner", telemetry.NewAttributes(telemetry.Property{Name: "value", Value: 1}))
						captured.SetStatus(telemetry.SpanStatus{Status: "error", Error: &telemetry.ErrorDetails{Name: "Nested", Message: "inner"}})
					}
					if tc.Input.Phase == "snapshot" || strings.HasPrefix(tc.Input.Phase, "nested") {
						snapshots = append(snapshots, recorder.GetSpans())
					}
					if tc.Input.Phase == "nil_panic" {
						panic(nil)
					}
					if tc.Input.Phase == "panic" || tc.Input.Phase == "nested_panic" {
						panic(errors.New("read failed"))
					}
					value := any("read")
					if tc.Input.Phase == "nested_undefined" {
						value = telemetry.Undefined
					}
					if tc.Input.Phase == "nested_null" {
						value = nil
					}
					return telemetry.NewAttributes(telemetry.Property{Name: "outer", Value: value})
				}
				apply := func() {
					if tc.Input.Method == "attributes" {
						captured.SetAttributesFrom(read)
					} else {
						captured.AddEventFrom("outer", read)
					}
				}
				func() {
					defer func() {
						if value := recover(); value != nil {
							panicked = true
						}
					}()
					err := parent.StartSpan(telemetry.SpanOptions{Name: "read", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "base", Value: "initial"}, telemetry.Property{Name: "kept", Value: []int{1}})}, func(span *telemetry.Span) error {
						captured = span
						if tc.Input.Scope != "late" {
							apply()
						}
						if tc.Input.Fail {
							return failure
						}
						return nil
					})
					callbackError = err == failure
				}()
				if tc.Input.Scope == "late" {
					func() {
						defer func() {
							if value := recover(); value != nil {
								panicked = true
							}
						}()
						apply()
					}()
				}
				actual, err := json.Marshal(map[string]any{"reads": reads, "panicked": panicked, "callbackError": callbackError, "snapshots": snapshots, "spans": recorder.GetSpans()})
				if err != nil {
					t.Fatal(err)
				}
				var got, want any
				if err = json.Unmarshal(actual, &got); err != nil {
					t.Fatal(err)
				}
				if err = json.Unmarshal(tc.Expected, &want); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
				}
			})
		}
	}
}

// Readers may wait for work in another goroutine. Settlement must stay available
// while they run and must prevent a completed read from changing a closed span.
func TestAttributeReaderDoesNotHoldRecorderLockOrOverwriteSettlement(t *testing.T) {
	for _, method := range []string{"attributes", "event"} {
		t.Run(method, func(t *testing.T) {
			recorder := telemetry.NewInMemory()
			entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
			done := make(chan error, 1)
			failure := errors.New("callback failed")
			go func() {
				done <- recorder.StartSpan(telemetry.SpanOptions{Name: "concurrent", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "kept", Value: true})}, func(span *telemetry.Span) error {
					go func() {
						defer close(finished)
						read := func() telemetry.Attributes {
							close(entered)
							<-release
							return telemetry.NewAttributes(telemetry.Property{Name: "late", Value: true})
						}
						if method == "attributes" {
							span.SetAttributesFrom(read)
						} else {
							span.AddEventFrom("late", read)
						}
					}()
					<-entered
					return failure
				})
			}()
			select {
			case err := <-done:
				if err != failure {
					t.Fatal("callback error changed", err)
				}
			case <-time.After(5 * time.Second):
				close(release)
				t.Fatal("attribute reader held recorder lock")
			}
			before := recorder.GetSpans()
			close(release)
			select {
			case <-finished:
			case <-time.After(5 * time.Second):
				t.Fatal("attribute reader did not finish")
			}
			after := recorder.GetSpans()
			if !reflect.DeepEqual(before, after) || !after[0].Settled {
				t.Fatal("late reader changed settled span")
			}
		})
	}
}

func TestInvalidAttributeReadRetainsReentrantChanges(t *testing.T) {
	recorder := telemetry.NewInMemory()
	if err := recorder.StartSpan(telemetry.SpanOptions{Name: "invalid-reader"}, func(span *telemetry.Span) error {
		read := func() telemetry.Attributes {
			span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "inner", Value: true}))
			return telemetry.NewAttributes(telemetry.Property{Name: "partial", Value: true}, telemetry.Property{Name: "unsupported", Value: make(chan int)})
		}
		span.SetAttributesFrom(read)
		span.AddEventFrom("invalid", read)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	span := recorder.GetSpans()[0]
	if !reflect.DeepEqual(span.Attributes, telemetry.NewAttributes(telemetry.Property{Name: "inner", Value: true})) || len(span.Events) != 0 {
		t.Fatalf("invalid payload changed or partially committed attributes: %+v", span)
	}
}
