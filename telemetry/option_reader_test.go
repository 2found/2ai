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

func optionReaderContext(parent telemetry.Context) telemetry.Context {
	wrap := func(callback func(*telemetry.Span) error) func(*telemetry.Span) error {
		return func(span *telemetry.Span) error {
			return callback(telemetry.NewSpan(optionReaderContext(span.Context()), telemetry.SpanCallbacks{
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

func TestPiOptionReaderOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-option-readers.json")
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
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 64 {
		t.Fatal("unexpected option reader oracle coverage")
	}
	for _, tc := range fixture.Cases {
		for _, forwarded := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%s/fail=%v/forwarded=%v", tc.Input.Scope, tc.Input.Phase, tc.Input.Fail, forwarded), func(t *testing.T) {
				recorder := telemetry.NewInMemory()
				root := recorder.Context
				parent := root
				if tc.Input.Scope == "noop" {
					parent = telemetry.Context{}
				}
				if forwarded {
					root = optionReaderContext(root)
					parent = optionReaderContext(parent)
				}
				reads := []string{}
				snapshots := [][]telemetry.RecordedSpan{}
				token := &struct{ Value string }{"callback result"}
				failure := errors.New("callback failed")
				calls, children := 0, 0
				valueSame, errorSame, unexpected := false, false, false
				name := "read"
				read := func() telemetry.SpanOptions {
					reads = append(reads, "name")
					snapshots = append(snapshots, recorder.GetSpans())
					if tc.Input.Phase == "name_nil_panic" {
						panic(nil)
					}
					if tc.Input.Phase == "name_panic" {
						panic(errors.New("name failed"))
					}
					selectedName := name
					reads = append(reads, "attributes")
					if strings.HasPrefix(tc.Input.Phase, "reader_child") {
						if err := parent.StartSpan(telemetry.SpanOptions{Name: "reader-child"}, func(*telemetry.Span) error { return nil }); err != nil {
							t.Fatal(err)
						}
					}
					if tc.Input.Phase == "attributes_change_name" {
						name = "changed"
					}
					if tc.Input.Phase == "attributes_panic" || tc.Input.Phase == "reader_child_panic" {
						panic(errors.New("attributes failed"))
					}
					reads = append(reads, "value")
					if tc.Input.Phase == "value_panic" {
						panic(errors.New("value failed"))
					}
					return telemetry.SpanOptions{Name: selectedName, Attributes: telemetry.NewAttributes(telemetry.Property{Name: "value", Value: "read"})}
				}
				run := func() {
					defer func() {
						if recover() != nil {
							unexpected = true
						}
					}()
					value, err := telemetry.StartSpanFrom(parent, read, func(span *telemetry.Span) (*struct{ Value string }, error) {
						calls++
						span.AddEvent("callback", telemetry.NewAttributes(telemetry.Property{Name: "value", Value: true}))
						readChild := func() telemetry.SpanOptions {
							reads = append(reads, "child-name")
							return telemetry.SpanOptions{Name: "callback-child"}
						}
						if err := span.StartSpanFrom(readChild, func(*telemetry.Span) error { children++; return nil }); err != nil {
							return nil, err
						}
						if tc.Input.Fail {
							return nil, failure
						}
						return token, nil
					})
					valueSame, errorSame = value == token, err == failure
					if err != nil && !errorSame {
						unexpected = true
					}
				}
				switch tc.Input.Scope {
				case "child":
					if err := root.StartSpan(telemetry.SpanOptions{Name: "parent"}, func(span *telemetry.Span) error { parent = span.Context(); run(); return nil }); err != nil {
						t.Fatal(err)
					}
				case "late":
					if err := root.StartSpan(telemetry.SpanOptions{Name: "parent"}, func(span *telemetry.Span) error { parent = span.Context(); return nil }); err != nil {
						t.Fatal(err)
					}
					run()
				default:
					run()
				}
				if err := root.StartSpan(telemetry.SpanOptions{Name: "probe"}, func(*telemetry.Span) error { return nil }); err != nil {
					t.Fatal(err)
				}
				actual, err := json.Marshal(map[string]any{"reads": reads, "snapshots": snapshots, "calls": calls, "children": children, "valueSame": valueSame, "errorSame": errorSame, "unexpected": unexpected, "spans": recorder.GetSpans()})
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

func TestOptionReaderDoesNotHoldRecorderLockOrOutliveParent(t *testing.T) {
	recorder := telemetry.NewInMemory()
	entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
	parentDone := make(chan error, 1)
	callbackFailure := errors.New("child callback failed")
	calls := 0
	var childError error
	go func() {
		parentDone <- recorder.StartSpan(telemetry.SpanOptions{Name: "parent"}, func(parent *telemetry.Span) error {
			go func() {
				defer close(finished)
				childError = parent.StartSpanFrom(func() telemetry.SpanOptions {
					close(entered)
					<-release
					return telemetry.SpanOptions{Name: "late"}
				}, func(span *telemetry.Span) error {
					calls++
					span.AddEvent("must not record", telemetry.Attributes{})
					return callbackFailure
				})
			}()
			<-entered
			return nil
		})
	}()
	select {
	case err := <-parentDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("option reader held recorder lock")
	}
	before := recorder.GetSpans()
	close(release)
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("option reader did not finish")
	}
	if calls != 1 || childError != callbackFailure || !reflect.DeepEqual(before, recorder.GetSpans()) {
		t.Fatal("late admission changed recording or callback failure")
	}
	_ = recorder.StartSpan(telemetry.SpanOptions{Name: "probe"}, func(*telemetry.Span) error { return nil })
	if spans := recorder.GetSpans(); len(spans) != 2 || spans[1].ID != 2 {
		t.Fatal("late admission consumed a span ID")
	}
}

func TestOptionReaderPreservesCallbackPanic(t *testing.T) {
	for _, phase := range []string{"success", "panic", "nil-reader"} {
		t.Run(phase, func(t *testing.T) {
			recorder := telemetry.NewInMemory()
			failure := errors.New("callback failed")
			read := func() telemetry.SpanOptions {
				if phase == "panic" {
					panic("read failed")
				}
				return telemetry.SpanOptions{Name: "read"}
			}
			if phase == "nil-reader" {
				read = nil
			}
			calls := 0
			var thrown any
			func() {
				defer func() { thrown = recover() }()
				_ = recorder.StartSpanFrom(read, func(*telemetry.Span) error { calls++; panic(failure) })
			}()
			if calls != 1 || thrown != failure {
				t.Fatal("read replaced or suppressed callback panic")
			}
			spans := recorder.GetSpans()
			if phase == "success" {
				if len(spans) != 1 || !spans[0].Settled || spans[0].Status.Error == nil || spans[0].Status.Error.Message != "callback failed" {
					t.Fatal("admitted panic did not settle")
				}
			} else if len(spans) != 0 {
				t.Fatal("failed options recorded a span")
			}
		})
	}
}

func TestOpaqueContextRequiresDeferredAdmissionCallback(t *testing.T) {
	eager, reads, calls := 0, 0, 0
	parent := telemetry.NewContext(func(telemetry.SpanOptions, func(*telemetry.Span) error) error { eager++; return nil })
	value, err := telemetry.StartSpanFrom(parent, func() telemetry.SpanOptions { reads++; return telemetry.SpanOptions{Name: "unadmitted"} }, func(span *telemetry.Span) (int, error) {
		calls++
		span.AddEvent("ignored", telemetry.Attributes{})
		return 7, nil
	})
	if eager != 0 || reads != 0 || calls != 1 || value != 7 || err != nil {
		t.Fatal("opaque eager backend guessed deferred admission")
	}
}
