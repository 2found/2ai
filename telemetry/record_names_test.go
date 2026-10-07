package telemetry_test

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/2found/2ai/jsonjs"
	"github.com/2found/2ai/telemetry"
)

func TestPiRecordNames(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-record-names.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct{ Name, Options, Active, Settled, Event string }
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 10 {
		t.Fatal("unexpected name oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			check := func(label string, value any, expected string) {
				t.Helper()
				var actual []byte
				var err error
				if text, ok := value.(string); ok {
					actual = jsonjs.QuoteString(text)
				} else {
					actual, err = json.Marshal(value)
					if err == nil {
						actual, err = jsonjs.StringifyJSON(actual)
					}
				}
				if err != nil {
					t.Fatal(err)
				}
				if string(actual) != expected {
					t.Fatalf("%s\nGo: %s\nPi: %s", label, actual, expected)
				}
			}
			var options telemetry.SpanOptions
			if err := json.Unmarshal([]byte(tc.Options), &options); err != nil {
				t.Fatal(err)
			}
			// Serialize with the public JSON codec, then compare the name's raw
			// JSON; an ordinary string decoder would hide surrogate replacement.
			checkName := func(value any) {
				t.Helper()
				raw, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(raw, &fields); err != nil {
					t.Fatal(err)
				}
				var decoded telemetry.Attributes
				if err := json.Unmarshal([]byte(`{"name":`+string(fields["name"])+`}`), &decoded); err != nil {
					t.Fatal(err)
				}
				check("serialized name", decoded.Get("name"), tc.Name)
			}
			checkName(options)
			context := telemetry.NewInMemory()
			if err := context.StartSpan(options, func(span *telemetry.Span) error {
				span.AddEvent(options.Name, telemetry.NewAttributes(telemetry.Property{Name: "source", Value: "event"}))
				check("active", context.GetSpans(), tc.Active)
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			check("settled", context.GetSpans(), tc.Settled)
			var spans []telemetry.RecordedSpan
			if err := json.Unmarshal([]byte(tc.Settled), &spans); err != nil {
				t.Fatal(err)
			}
			check("imported snapshot", spans, tc.Settled)
			var event telemetry.RecordedEvent
			if err := json.Unmarshal([]byte(tc.Event), &event); err != nil {
				t.Fatal(err)
			}
			checkName(event)
		})
	}
}
