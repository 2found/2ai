package telemetry_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/2found/2ai/telemetry"
)

func TestPiStatusJSONOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-status-json.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name, Prior, StatusJSON string
				Status                  json.RawMessage
				Fail                    bool
			}
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 228 {
		t.Fatal("unexpected status JSON fixture coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(fmt.Sprintf("%s/%s/fail=%v", tc.Input.Name, tc.Input.Prior, tc.Input.Fail), func(t *testing.T) {
			var status telemetry.SpanStatus
			statusJSON := tc.Input.Status
			if tc.Input.StatusJSON != "" {
				statusJSON = json.RawMessage(tc.Input.StatusJSON)
			}
			if err := json.Unmarshal(statusJSON, &status); err != nil {
				t.Fatal(err)
			}
			recorder := telemetry.NewInMemory()
			var active []telemetry.RecordedSpan
			failure := errors.New("callback failed")
			err := recorder.StartSpan(telemetry.SpanOptions{Name: "status"}, func(span *telemetry.Span) error {
				switch tc.Input.Prior {
				case "ok":
					span.SetStatus(telemetry.SpanStatus{Status: "ok"})
				case "error":
					span.SetStatus(telemetry.SpanStatus{Status: "error", Error: &telemetry.ErrorDetails{Name: "Prior", Message: "retained"}})
				}
				span.SetStatus(status)
				active = recorder.GetSpans()
				if tc.Input.Fail {
					return failure
				}
				return nil
			})
			actual := map[string]any{"active": active, "settled": recorder.GetSpans()}
			if tc.Input.Fail {
				if err != failure {
					t.Fatal("callback error replaced")
				}
				actual["failure"] = err.Error()
			} else if err != nil {
				t.Fatal(err)
			}
			fields := map[string]string{}
			if details := recorder.GetSpans()[0].Status.Error; details != nil {
				raw, err := json.Marshal(details)
				if err != nil {
					t.Fatal(err)
				}
				var values map[string]json.RawMessage
				if err := json.Unmarshal(raw, &values); err != nil {
					t.Fatal(err)
				}
				for name, value := range values {
					var compact bytes.Buffer
					if err := json.Compact(&compact, value); err != nil {
						t.Fatal(err)
					}
					fields[name] = compact.String()
				}
			}
			actual["fields"] = fields
			statusBytes, err := json.Marshal(recorder.GetSpans()[0].Status)
			if err != nil {
				t.Fatal(err)
			}
			actual["statusJSON"] = string(statusBytes)
			encoded, err := json.Marshal(actual)
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err = json.Unmarshal(encoded, &got); err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", encoded, tc.Expected)
			}
		})
	}
}

func TestDecodedStatusCopiesRemainDetachedAndEditable(t *testing.T) {
	raw := []byte(`{"status":"error","error":{"name":null,"message":"original","ignored":true}}`)
	var status telemetry.SpanStatus
	if err := json.Unmarshal(raw, &status); err != nil {
		t.Fatal(err)
	}
	for i := range raw {
		raw[i] = 'x'
	}
	recorder := telemetry.NewInMemory()
	if err := recorder.StartSpan(telemetry.SpanOptions{Name: "decoded"}, func(span *telemetry.Span) error {
		span.SetStatus(status)
		status.Error.Name = "outside"
		status.Error.Message = "outside"
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	snapshot := recorder.GetSpans()[0]
	encoded, err := json.Marshal(snapshot.Status)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{"status":"error","error":{"name":null,"message":"original"}}` {
		t.Fatalf("decoded status lost shape or retained caller data: %s", encoded)
	}
	snapshot.Status.Error.Name = "inside"
	snapshot.Status.Error.Message = ""
	changed, err := json.Marshal(snapshot.Status)
	if err != nil {
		t.Fatal(err)
	}
	if string(changed) != `{"status":"error","error":{"name":"inside","message":""}}` {
		t.Fatalf("native fields did not override decoded values: %s", changed)
	}
	again, err := json.Marshal(recorder.GetSpans()[0].Status)
	if err != nil {
		t.Fatal(err)
	}
	if string(again) != string(encoded) {
		t.Fatalf("snapshot edit changed recorder: %s", again)
	}
	var roundTrip telemetry.SpanStatus
	if err = json.Unmarshal(encoded, &roundTrip); err != nil {
		t.Fatal(err)
	}
	restored, err := json.Marshal(roundTrip)
	if err != nil {
		t.Fatal(err)
	}
	if string(restored) != string(encoded) {
		t.Fatalf("status round trip changed field presence: %s", restored)
	}
	if err = json.Unmarshal([]byte(`{"status":`), &roundTrip); err == nil {
		t.Fatal("malformed JSON was accepted")
	}
	preserved, err := json.Marshal(roundTrip)
	if err != nil {
		t.Fatal(err)
	}
	if string(preserved) != string(encoded) {
		t.Fatal("failed decode partially replaced status")
	}
}
