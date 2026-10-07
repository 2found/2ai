package telemetry_test

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/telemetry"
)

func TestPiTelemetrySchemaOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/pi-schema.json")
	if err != nil {
		t.Fatal(err)
	}
	type typedAction struct {
		Op, Name, Target string
		Attributes       telemetry.Attributes
		Actions          []json.RawMessage
		Result           *int
	}
	var fixture struct {
		UpstreamCommit string
		Schemas        []json.RawMessage
		Cases          []struct {
			Input struct {
				Name                        string
				Noop, Duplicate, Unreadable bool
				Actions                     []json.RawMessage
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 7 {
		t.Fatal("unexpected oracle revision/coverage")
	}
	schemas := make([]*telemetry.SchemaDefinition, len(fixture.Schemas))
	for i, raw := range fixture.Schemas {
		schema := new(telemetry.SchemaDefinition)
		if err := json.Unmarshal(raw, schema); err != nil {
			t.Fatal(err)
		}
		if telemetry.DefineSchema(schema) != schema {
			t.Fatal("schema identity lost")
		}
		assertSchemaJSON(t, raw, schema)
		schemas[i] = schema
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			recorder := telemetry.NewInMemory()
			parent := recorder.Context
			if tc.Input.Noop {
				parent = telemetry.Context{}
			}
			inputs := schemas
			if tc.Input.Duplicate {
				inputs = []*telemetry.SchemaDefinition{schemas[0], schemas[0]}
			}
			if tc.Input.Unreadable {
				inputs = []*telemetry.SchemaDefinition{nil}
			}
			root := telemetry.CreateTypedSpanStarter(parent, inputs...)
			retained := map[string]telemetry.SpanStarter{"root": root}
			type result struct {
				Name   string `json:"name"`
				Result *int   `json:"result"`
			}
			results := []result{}
			var run func([]json.RawMessage, telemetry.SpanStarter, *telemetry.Span) error
			run = func(actions []json.RawMessage, starter telemetry.SpanStarter, current *telemetry.Span) error {
				for _, raw := range actions {
					var a typedAction
					if err := json.Unmarshal(raw, &a); err != nil {
						return err
					}
					switch a.Op {
					case "span":
						target := starter
						if a.Target != "" {
							target = retained[a.Target]
						}
						value, err := telemetry.StartTypedSpan(target, a.Name, a.Attributes, func(child *telemetry.Span, children telemetry.SpanStarter) (*int, error) {
							retained[a.Name] = children
							return a.Result, run(a.Actions, children, child)
						})
						if err != nil {
							return err
						}
						results = append(results, result{Name: a.Name, Result: value})
					case "attributes":
						current.SetAttributes(a.Attributes)
					case "event":
						current.AddEvent(a.Name, a.Attributes)
					default:
						return errors.New("unknown fixture action")
					}
				}
				return nil
			}
			if err := run(tc.Input.Actions, root, nil); err != nil {
				t.Fatal(err)
			}
			assertSchemaJSON(t, tc.Expected, map[string]any{"spans": recorder.GetSpans(), "results": results})
		})
	}
}

func assertSchemaJSON(t *testing.T, expected json.RawMessage, value any) {
	t.Helper()
	actual, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var got, want any
	if err := json.Unmarshal(actual, &got); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(expected, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Go: %s\nPi: %s", actual, expected)
	}
}

func TestTypedStarterPreservesValuesAndFailures(t *testing.T) {
	expected := &struct{ Value int }{42}
	failure := errors.New("same error")
	for _, parent := range []telemetry.Context{{}, telemetry.NewInMemory().Context} {
		starter := telemetry.CreateTypedSpanStarter(parent, nil)
		calls := 0
		value, err := telemetry.StartTypedSpan(starter, "operation", telemetry.Attributes{}, func(_ *telemetry.Span, children telemetry.SpanStarter) (*struct{ Value int }, error) {
			calls++
			return telemetry.StartTypedSpan(children, "request", telemetry.Attributes{}, func(*telemetry.Span, telemetry.SpanStarter) (*struct{ Value int }, error) {
				calls++
				return expected, failure
			})
		})
		if value != expected || err != failure || calls != 2 {
			t.Fatal("typed starter changed result, error, or callback count")
		}
		marker := &struct{ Kind string }{"panic"}
		func() {
			defer func() {
				if recover() != marker {
					t.Error("typed starter changed panic identity")
				}
			}()
			_ = starter.StartSpan("operation", telemetry.Attributes{}, func(*telemetry.Span, telemetry.SpanStarter) error { panic(marker) })
		}()
	}
}
