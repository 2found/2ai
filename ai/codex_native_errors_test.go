package ai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"testing"
	"time"
)

func TestPiCodexErrorsOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-codex-errors.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Now            int64
		Retry          []struct {
			Status   int
			Text     string
			Expected bool
		}
		Delays []struct {
			Headers  map[string]string
			Expected *float64
		}
		Limits []struct {
			Delay float64
			Limit *float64
			Error *string
		}
		Errors []struct {
			Input struct {
				Raw, StatusText string
				Status          int
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Retry)+len(fixture.Delays)+len(fixture.Limits)+len(fixture.Errors) != 112 {
		t.Fatal("unexpected error oracle revision/coverage")
	}
	now := time.UnixMilli(fixture.Now)
	for i, tc := range fixture.Retry {
		t.Run(fmt.Sprintf("retry/%d", i), func(t *testing.T) {
			if got := codexRetryableError(tc.Status, tc.Text); got != tc.Expected {
				t.Fatalf("retry=%v want=%v", got, tc.Expected)
			}
		})
	}
	for i, tc := range fixture.Delays {
		t.Run(fmt.Sprintf("delay/%d", i), func(t *testing.T) {
			headers := http.Header{}
			for name, value := range tc.Headers {
				headers.Set(name, value)
			}
			got, ok := codexRetryAfter(headers, now)
			if (tc.Expected != nil) != ok || (ok && got != *tc.Expected) {
				t.Fatalf("delay=%v present=%v want=%v", got, ok, tc.Expected)
			}
		})
	}
	for i, tc := range fixture.Limits {
		t.Run(fmt.Sprintf("limit/%d", i), func(t *testing.T) {
			err := codexValidateRetryDelay(tc.Delay, tc.Limit)
			if tc.Error == nil {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || err.Error() != *tc.Error {
				t.Fatalf("error=%v want=%s", err, *tc.Error)
			}
		})
	}
	for i, tc := range fixture.Errors {
		t.Run(fmt.Sprintf("response/%d", i), func(t *testing.T) {
			assertPiJSON(t, tc.Expected, codexParseErrorResponse(tc.Input.Raw, tc.Input.StatusText, tc.Input.Status, now))
		})
	}
}
