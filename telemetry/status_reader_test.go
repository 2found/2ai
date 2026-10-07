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

func TestPiStatusReaderOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-status-readers.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Scope, Phase string
				Fail         bool
			}
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 48 {
		t.Fatal("unexpected status reader coverage")
	}
	for _, tc := range fixture.Cases {
		for _, forwarded := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%s/fail=%v/forwarded=%v", tc.Input.Scope, tc.Input.Phase, tc.Input.Fail, forwarded), func(t *testing.T) {
				recorder := telemetry.NewInMemory()
				parent := recorder.Context
				if tc.Input.Scope == "noop" {
					parent = telemetry.Context{}
				}
				if forwarded {
					original := parent
					parent = telemetry.NewContext(func(options telemetry.SpanOptions, callback func(*telemetry.Span) error) error {
						return original.StartSpan(options, func(span *telemetry.Span) error {
							return callback(telemetry.NewSpan(span.Context(), telemetry.SpanCallbacks{SetStatus: span.SetStatus, SetStatusFrom: span.SetStatusFrom}))
						})
					})
				}
				var captured *telemetry.Span
				reads, panicked, callbackError := 0, false, false
				snapshots := [][]telemetry.RecordedSpan{}
				failure := errors.New("callback failed")
				read := func() telemetry.SpanStatus {
					reads++
					if strings.HasPrefix(tc.Input.Phase, "nested_") {
						captured.SetStatus(telemetry.SpanStatus{Status: "error", Error: &telemetry.ErrorDetails{Name: "Nested", Message: "inner"}})
					}
					if tc.Input.Phase == "snapshot" {
						snapshots = append(snapshots, recorder.GetSpans())
					}
					if tc.Input.Phase == "nil_panic" {
						panic(nil)
					}
					if tc.Input.Phase == "panic" || tc.Input.Phase == "nested_panic" {
						panic(errors.New("read failed"))
					}
					status := "ok"
					if tc.Input.Phase == "error" || tc.Input.Phase == "nested_error" {
						status = "error"
					}
					return telemetry.SpanStatus{Status: status, Error: &telemetry.ErrorDetails{Name: "Read", Message: "outer"}}
				}
				apply := func() { captured.SetStatusFrom(read) }
				func() {
					defer func() {
						if value := recover(); value != nil {
							panicked = true
						}
					}()
					err := parent.StartSpan(telemetry.SpanOptions{Name: "read"}, func(span *telemetry.Span) error {
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

func TestStatusReaderDoesNotHoldRecorderLockOrOverwriteSettlement(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(fmt.Sprint(fail), func(t *testing.T) {
			recorder := telemetry.NewInMemory()
			entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
			parentDone := make(chan error, 1)
			failure := errors.New("callback failed")
			go func() {
				parentDone <- recorder.StartSpan(telemetry.SpanOptions{Name: "concurrent"}, func(span *telemetry.Span) error {
					go func() {
						defer close(finished)
						span.SetStatusFrom(func() telemetry.SpanStatus {
							close(entered)
							<-release
							return telemetry.SpanStatus{Status: "error", Error: &telemetry.ErrorDetails{Name: "Late", Message: "must not overwrite"}}
						})
					}()
					<-entered
					if fail {
						return failure
					}
					return nil
				})
			}()
			select {
			case err := <-parentDone:
				if (fail && err != failure) || (!fail && err != nil) {
					t.Fatal("callback result changed", err)
				}
			case <-time.After(5 * time.Second):
				close(release)
				t.Fatal("status reader held recorder lock across user code")
			}
			before := recorder.GetSpans()
			close(release)
			select {
			case <-finished:
			case <-time.After(5 * time.Second):
				t.Fatal("status reader failed to finish")
			}
			after := recorder.GetSpans()
			if !reflect.DeepEqual(before, after) || !after[0].Settled {
				t.Fatal("late read overwrote settled span")
			}
		})
	}
}
