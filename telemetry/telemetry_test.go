package telemetry_test

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/telemetry"
)

type action struct {
	Op                  string               `json:"op"`
	Name                string               `json:"name"`
	Attributes          telemetry.Attributes `json:"attributes"`
	UndefinedAttributes []string             `json:"undefinedAttributes"`
	Status              telemetry.SpanStatus `json:"status"`
	Actions             []action             `json:"actions"`
	Inspection          []action             `json:"inspection"`
	Target              string               `json:"target"`
	Message             string               `json:"message"`
}

type oracleInspectedError struct {
	message string
	inspect func()
}

func (e *oracleInspectedError) Error() string {
	e.inspect()
	return e.message
}

func TestPiTelemetryOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/pi-spans.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string `json:"upstreamCommit"`
		Cases          []struct {
			Name     string          `json:"name"`
			Actions  []action        `json:"actions"`
			Expected json.RawMessage `json:"expected"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 39 {
		t.Fatal("unexpected oracle revision or coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			recorder := telemetry.NewInMemory()
			retained := map[string]*telemetry.Span{}
			snapshots := [][]telemetry.RecordedSpan{}
			failures, admitted := []string{}, []string{}
			var run func([]action, telemetry.Context, *telemetry.Span) error
			run = func(actions []action, parent telemetry.Context, current *telemetry.Span) error {
				for _, a := range actions {
					if len(a.UndefinedAttributes) > 0 {
						if a.Attributes.IsZero() {
							a.Attributes = telemetry.NewAttributes()
						}
						for _, key := range a.UndefinedAttributes {
							a.Attributes.Set(key, telemetry.Undefined)
						}
					}
					target, targetSpan := parent, current
					if a.Target != "" {
						targetSpan = retained[a.Target]
						target = targetSpan.Context()
					}
					switch a.Op {
					case "span":
						err := target.StartSpan(telemetry.SpanOptions{Name: a.Name, Attributes: a.Attributes}, func(child *telemetry.Span) error {
							admitted = append(admitted, a.Name)
							retained[a.Name] = child
							return run(a.Actions, child.Context(), child)
						})
						if err != nil {
							if inspected, ok := err.(*oracleInspectedError); ok {
								// Observing the callback result must not trigger another
								// inspection or hide whether telemetry inspected it.
								failures = append(failures, inspected.message)
							} else {
								failures = append(failures, err.Error())
							}
						}
					case "attributes":
						targetSpan.SetAttributes(a.Attributes)
					case "event":
						targetSpan.AddEvent(a.Name, a.Attributes)
					case "status":
						targetSpan.SetStatus(a.Status)
					case "snapshot":
						snapshots = append(snapshots, recorder.GetSpans())
					case "failure":
						if a.Inspection != nil {
							return &oracleInspectedError{message: a.Message, inspect: func() {
								if err := run(a.Inspection, target, targetSpan); err != nil {
									panic(err)
								}
							}}
						}
						return errors.New(a.Message)
					default:
						t.Fatalf("unknown oracle operation %q", a.Op)
					}
				}
				return nil
			}
			if err := run(tc.Actions, recorder.Context, nil); err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"spans": recorder.GetSpans(), "snapshots": snapshots, "failures": failures, "admitted": admitted})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}

func TestCallbackIdentityAndNoop(t *testing.T) {
	expected := &struct{ Value int }{42}
	failure := errors.New("same error")
	for _, context := range []telemetry.Context{{}, telemetry.NewInMemory().Context} {
		calls := 0
		value, err := telemetry.StartSpan(context, telemetry.SpanOptions{Name: "result"}, func(*telemetry.Span) (*struct{ Value int }, error) {
			calls++
			return expected, failure
		})
		if calls != 1 || value != expected || err != failure {
			t.Fatal("callback identity changed")
		}
		panicValue := &struct{ Kind string }{"failure"}
		func() {
			defer func() {
				if recover() != panicValue {
					t.Error("panic identity changed")
				}
			}()
			_ = context.StartSpan(telemetry.SpanOptions{Name: "panic"}, func(*telemetry.Span) error { panic(panicValue) })
		}()
	}
	_ = (telemetry.Context{}).StartSpan(telemetry.SpanOptions{}, func(parent *telemetry.Span) error {
		return parent.StartSpan(telemetry.SpanOptions{}, func(child *telemetry.Span) error {
			if parent != child {
				t.Error("no-op span is not shared")
			}
			return nil
		})
	})
}

func TestPiTelemetryThrownValues(t *testing.T) {
	data, err := os.ReadFile("testdata/pi-spans.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		ThrownValues   []struct {
			Input struct {
				Kind, Name, Message string
				Value               any
				ExplicitStatus      *telemetry.SpanStatus
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.ThrownValues) != 24 {
		t.Fatal("unexpected thrown-value oracle revision or coverage")
	}
	for _, test := range fixture.ThrownValues {
		name, _ := json.Marshal(test.Input)
		t.Run(string(name), func(t *testing.T) {
			recorder := telemetry.NewInMemory()
			failure := test.Input.Value
			if test.Input.Kind == "error" {
				if test.Input.Name == "Error" {
					failure = errors.New(test.Input.Message)
				} else {
					failure = &telemetry.ErrorDetails{Name: test.Input.Name, Message: test.Input.Message}
				}
			}
			sameFailure := false
			func() {
				defer func() {
					recovered := recover()
					if failure == nil {
						// Modern Go represents panic(nil) using this runtime value.
						_, sameFailure = recovered.(*runtime.PanicNilError)
					} else {
						value := reflect.ValueOf(failure)
						switch value.Kind() {
						case reflect.Map, reflect.Slice, reflect.Pointer:
							sameFailure = reflect.TypeOf(recovered) == value.Type() && reflect.ValueOf(recovered).Pointer() == value.Pointer()
						default:
							sameFailure = recovered == failure
						}
					}
				}()
				_ = recorder.StartSpan(telemetry.SpanOptions{Name: "failure"}, func(span *telemetry.Span) error {
					if test.Input.ExplicitStatus != nil {
						span.SetStatus(*test.Input.ExplicitStatus)
					}
					panic(failure)
				})
			}()
			actual, err := json.Marshal(map[string]any{"spans": recorder.GetSpans(), "sameFailure": sameFailure})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(test.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, test.Expected)
			}
		})
	}
}

func TestDetachedSnapshotsAndInputs(t *testing.T) {
	recorder := telemetry.NewInMemory()
	strings := []string{"original"}
	numbers := []float64{1}
	bools := []bool{true}
	attrs := telemetry.NewAttributes(telemetry.Property{Name: "strings", Value: strings}, telemetry.Property{Name: "numbers", Value: numbers}, telemetry.Property{Name: "bools", Value: bools})
	details := &telemetry.ErrorDetails{Name: "Expected", Message: "original"}
	_ = recorder.StartSpan(telemetry.SpanOptions{Name: "parent", Attributes: attrs}, func(parent *telemetry.Span) error {
		return parent.StartSpan(telemetry.SpanOptions{Name: "child", Attributes: attrs}, func(child *telemetry.Span) error {
			child.AddEvent("event", attrs)
			child.SetStatus(telemetry.SpanStatus{Status: "error", Error: details})
			strings[0], numbers[0], bools[0] = "mutated", 2, false
			details.Message = "mutated"
			attrs.Set("new", true)
			return nil
		})
	})
	before := recorder.GetSpans()
	if before[1].Attributes.Get("strings").([]string)[0] != "original" || before[1].Status.Error.Message != "original" {
		t.Fatal("retained input references")
	}
	mutated := recorder.GetSpans()
	mutated[1].Attributes.Get("strings").([]string)[0] = "changed snapshot"
	mutated[1].Attributes.Get("numbers").([]float64)[0] = 100
	mutated[1].Attributes.Get("bools").([]bool)[0] = false
	mutated[1].Events[0].Attributes.Get("strings").([]string)[0] = "changed event"
	mutated[1].Status.Error.Message = "changed status"
	*mutated[1].ParentID, *mutated[1].EndSequence = 100, 100
	if !reflect.DeepEqual(before, recorder.GetSpans()) {
		t.Fatal("snapshot mutation reached recorder")
	}
}

type unreadableError struct{}

func (*unreadableError) Error() string { panic("unreadable error") }

type callbackError struct{ inspect func() string }

func (e *callbackError) Error() string { return e.inspect() }

func TestErrorInspectionIsPassiveAndConditional(t *testing.T) {
	recorder := telemetry.NewInMemory()
	inspected := false
	failure := &callbackError{inspect: func() string {
		inspected = true
		_ = recorder.GetSpans() // Must not run under the recorder's mutex.
		return "failure"
	}}
	_ = recorder.StartSpan(telemetry.SpanOptions{Name: "explicit"}, func(span *telemetry.Span) error {
		span.SetStatus(telemetry.SpanStatus{Status: "ok"})
		return failure
	})
	if inspected {
		t.Fatal("explicit status must suppress automatic error inspection")
	}
	_ = recorder.StartSpan(telemetry.SpanOptions{Name: "automatic"}, func(*telemetry.Span) error { return failure })
	if !inspected || recorder.GetSpans()[1].Status.Error.Message != "failure" {
		t.Fatal("missing automatic error")
	}
	func() {
		defer func() {
			if recover() != failure {
				t.Error("panic identity changed")
			}
		}()
		_ = recorder.StartSpan(telemetry.SpanOptions{Name: "panic"}, func(*telemetry.Span) error { panic(failure) })
	}()
	span := recorder.GetSpans()[2]
	if !span.Settled || span.Status.Status != "error" || span.Status.Error.Message != "failure" {
		t.Fatal("panic did not settle span")
	}
}

func TestErrorInspectionSettlementAllowsConcurrentRecorderAccess(t *testing.T) {
	recorder := telemetry.NewInMemory()
	inspecting, release := make(chan struct{}), make(chan struct{})
	defer close(release)
	spanReady := make(chan *telemetry.Span, 1)
	finished := make(chan error, 1)
	failure := &callbackError{inspect: func() string {
		close(inspecting)
		<-release
		return "original failure"
	}}
	go func() {
		finished <- recorder.StartSpan(telemetry.SpanOptions{Name: "concurrent-inspection"}, func(span *telemetry.Span) error {
			spanReady <- span
			return failure
		})
	}()
	select {
	case <-inspecting:
	case <-time.After(5 * time.Second):
		t.Fatal("error inspection did not start")
	}
	span := <-spanReady
	span.SetStatus(telemetry.SpanStatus{Status: "ok"})
	span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "duringInspection", Value: true}))
	span.AddEvent("duringInspection", telemetry.Attributes{})
	active := recorder.GetSpans()[0]
	if active.Settled || active.Status.Status != "ok" || active.EndSequence != nil {
		t.Fatalf("inspection prematurely settled its span: %+v", active)
	}
	release <- struct{}{}
	select {
	case err := <-finished:
		if err != failure {
			t.Fatal("error inspection replaced callback failure")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("error inspection did not finish")
	}
	span.SetStatus(telemetry.SpanStatus{Status: "ok"}) // Inert after settlement.
	settled := recorder.GetSpans()[0]
	if !settled.Settled || settled.EndSequence == nil || *settled.EndSequence != 1 || settled.Status.Status != "error" || settled.Status.Error == nil || settled.Status.Error.Message != "original failure" || settled.Attributes.Get("duringInspection") != true || len(settled.Events) != 1 {
		t.Fatalf("automatic assignment lost status or reentrant mutations: %+v", settled)
	}
}

func TestPassiveRecording(t *testing.T) {
	recorder := telemetry.NewInMemory()
	failure := &unreadableError{}
	err := recorder.StartSpan(telemetry.SpanOptions{Name: "passive", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "kept", Value: true})}, func(span *telemetry.Span) error {
		span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "partial", Value: true}, telemetry.Property{Name: "invalid", Value: make(chan int)}))
		span.AddEvent("invalid", telemetry.NewAttributes(telemetry.Property{Name: "invalid", Value: make(chan int)}))
		return failure
	})
	if err != failure {
		t.Fatal("recording replaced callback error")
	}
	span := recorder.GetSpans()[0]
	if !reflect.DeepEqual(span.Attributes, telemetry.NewAttributes(telemetry.Property{Name: "kept", Value: true})) || len(span.Events) != 0 || span.Status.Status != "error" || span.Status.Error != nil {
		t.Fatalf("passive recording contract: %+v", span)
	}
	calls := 0
	_ = recorder.StartSpan(telemetry.SpanOptions{Name: "invalid", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "invalid", Value: make(chan int)})}, func(*telemetry.Span) error { calls++; return nil })
	if calls != 1 || len(recorder.GetSpans()) != 1 {
		t.Fatal("invalid telemetry prevented callback or recorded partial span")
	}
}

func TestConcurrentChildrenAndSettlement(t *testing.T) {
	recorder := telemetry.NewInMemory()
	firstEntered, releaseFirst, firstDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	_ = recorder.StartSpan(telemetry.SpanOptions{Name: "parent"}, func(parent *telemetry.Span) error {
		go func() {
			defer close(firstDone)
			_ = parent.StartSpan(telemetry.SpanOptions{Name: "first"}, func(span *telemetry.Span) error {
				close(firstEntered)
				<-releaseFirst
				span.AddEvent("finished", telemetry.Attributes{})
				return nil
			})
		}()
		<-firstEntered
		_ = parent.StartSpan(telemetry.SpanOptions{Name: "second"}, func(*telemetry.Span) error { return nil })
		close(releaseFirst)
		<-firstDone
		return nil
	})
	spans := recorder.GetSpans()
	if *spans[1].ParentID != 1 || *spans[2].ParentID != 1 || *spans[2].EndSequence != 1 || *spans[1].EndSequence != 2 || *spans[0].EndSequence != 3 {
		t.Fatalf("concurrent parentage/order: %+v", spans)
	}
	var wg sync.WaitGroup
	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = recorder.StartSpan(telemetry.SpanOptions{Name: "concurrent"}, func(span *telemetry.Span) error {
				span.AddEvent("event", telemetry.Attributes{})
				span.SetAttributes(telemetry.NewAttributes(telemetry.Property{Name: "value", Value: 1}))
				_ = recorder.GetSpans()
				return nil
			})
		}()
	}
	wg.Wait()
	if len(recorder.GetSpans()) != 33 {
		t.Fatal("lost concurrent spans")
	}
}

func TestExplicitErrorDetailsRetainNameAndIdentity(t *testing.T) {
	recorder := telemetry.NewInMemory()
	failure := &telemetry.ErrorDetails{Name: "TypeError", Message: "invalid request"}
	err := recorder.StartSpan(telemetry.SpanOptions{Name: "named"}, func(*telemetry.Span) error { return failure })
	if err != failure || err.Error() != "invalid request" {
		t.Fatal("named failure identity/text changed")
	}
	failure.Name = "mutated"
	spans := recorder.GetSpans()
	if spans[0].Status.Error == nil || spans[0].Status.Error.Name != "TypeError" || spans[0].Status.Error.Message != "invalid request" {
		t.Fatal("named error details not snapshotted")
	}
}
