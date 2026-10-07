package telemetry_test

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/telemetry"
)

func TestPiBackendStatusDelegation(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-backend-status.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Mode, Kind string
				Throws     bool
				Status     telemetry.SpanStatus
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 32 {
		t.Fatal("unexpected backend-status coverage")
	}
	for _, tc := range fixture.Cases {
		name := tc.Input.Mode + "/" + tc.Input.Kind
		if tc.Input.Throws {
			name += "/throw"
		}
		t.Run(name, func(t *testing.T) {
			events := []any{}
			marker := &struct{ kind string }{"backend"}
			var backend func(any) telemetry.Context
			backend = func(parent any) telemetry.Context {
				return telemetry.NewContext(func(options telemetry.SpanOptions, callback func(*telemetry.Span) error) error {
					events = append(events, map[string]any{"op": "start", "name": options.Name, "parent": parent})
					defer func() { events = append(events, map[string]any{"op": "end", "name": options.Name}) }()
					span := telemetry.NewSpan(backend(options.Name), telemetry.SpanCallbacks{SetStatus: func(status telemetry.SpanStatus) {
						raw, err := json.Marshal(status)
						if err != nil {
							t.Fatal(err)
						}
						events = append(events, map[string]any{"op": "status", "name": options.Name, "value": json.RawMessage(raw)})
						if tc.Input.Throws {
							panic(marker)
						}
					}})
					return callback(span)
				})
			}
			ctx := backend(nil)
			start := telemetry.CreateTypedSpanStarter(ctx)
			action := func(span *telemetry.Span) (int, error) { span.SetStatus(tc.Input.Status); return 42, nil }
			typedAction := func(span *telemetry.Span, _ telemetry.SpanStarter) (int, error) { return action(span) }
			var result any
			failure := false
			func() {
				defer func() {
					if value := recover(); value != nil {
						if value != marker {
							t.Fatalf("backend panic changed: %v", value)
						}
						failure = true
					}
				}()
				var value int
				var err error
				switch tc.Input.Mode {
				case "direct":
					value, err = telemetry.StartSpan(ctx, telemetry.SpanOptions{Name: "root"}, action)
				case "typed":
					value, err = telemetry.StartTypedSpan(start, "root", telemetry.Attributes{}, typedAction)
				case "child":
					value, err = telemetry.StartTypedSpan(start, "root", telemetry.Attributes{}, func(_ *telemetry.Span, child telemetry.SpanStarter) (int, error) {
						return telemetry.StartTypedSpan(child, "child", telemetry.Attributes{}, typedAction)
					})
				case "late":
					var child telemetry.SpanStarter
					err = start.StartSpan("root", telemetry.Attributes{}, func(_ *telemetry.Span, next telemetry.SpanStarter) error { child = next; return nil })
					if err != nil {
						t.Fatal(err)
					}
					value, err = telemetry.StartTypedSpan(child, "child", telemetry.Attributes{}, typedAction)
				}
				if err != nil {
					t.Fatal(err)
				}
				result = value
			}()
			encoded, err := json.Marshal(map[string]any{"events": events, "result": result, "failure": failure})
			if err != nil {
				t.Fatal(err)
			}
			var actual, expected any
			if err := json.Unmarshal(encoded, &actual); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(tc.Expected, &expected); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(actual, expected) {
				t.Fatalf("Go: %s\nPi: %s", encoded, tc.Expected)
			}
		})
	}
}

func TestBackendSpanWithRootChildrenHasNoRecordingScope(t *testing.T) {
	for _, operation := range []string{"event", "attributes", "status", "event_reader", "attribute_reader", "status_reader"} {
		t.Run(operation, func(t *testing.T) {
			recorder := telemetry.NewInMemory()
			span := telemetry.NewSpan(recorder.Context, telemetry.SpanCallbacks{})
			read := func() telemetry.Attributes {
				t.Fatal("reader invoked without a recording scope")
				return telemetry.Attributes{}
			}
			func() {
				defer func() {
					if failure := recover(); failure != nil {
						t.Fatalf("missing backend operation panicked: %v", failure)
					}
				}()
				switch operation {
				case "event":
					span.AddEvent("ignored", telemetry.Attributes{})
				case "attributes":
					span.SetAttributes(telemetry.Attributes{})
				case "status":
					span.SetStatus(telemetry.SpanStatus{Status: "error"})
				case "event_reader":
					span.AddEventFrom("ignored", read)
				case "attribute_reader":
					span.SetAttributesFrom(read)
				case "status_reader":
					span.SetStatusFrom(func() telemetry.SpanStatus {
						t.Fatal("status read without a recording scope")
						return telemetry.SpanStatus{}
					})
				}
			}()
			if len(recorder.GetSpans()) != 0 {
				t.Fatal("opaque span mutation created a recording")
			}
			if err := span.StartSpan(telemetry.SpanOptions{Name: "child"}, func(*telemetry.Span) error { return nil }); err != nil {
				t.Fatal(err)
			}
			spans := recorder.GetSpans()
			if len(spans) != 1 || spans[0].ParentID != nil || spans[0].Name != "child" || !spans[0].Settled {
				t.Fatalf("child context was changed: %+v", spans)
			}
		})
	}
}
