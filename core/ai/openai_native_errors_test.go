package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"reflect"
	"testing"
	"time"
)

func TestPiCompletionsRetryOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-provider-retry.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name     string
				Status   int
				Index    float64
				MaxDelay *float64
				Headers  map[string]string
			}
			Expected struct {
				Provider, Retryable bool
				Delay               *int64
				Error               *string
			}
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 64 {
		t.Fatal("unexpected retry oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			provider := &completionsRequestError{status: tc.Input.Status, message: "provider failure", headers: http.Header{}}
			for name, value := range tc.Input.Headers {
				provider.headers.Set(name, value)
			}
			timing := completionsRetryTiming{now: func() time.Time { return time.UnixMilli(1000) }, random: func() float64 { return 0.5 }}
			delay, err := completionsRetryDelay(provider, tc.Input.Index, tc.Input.MaxDelay, timing)
			if tc.Expected.Error != nil {
				if err == nil || err.Error() != *tc.Expected.Error {
					t.Fatalf("delay error: %v, want %s", err, *tc.Expected.Error)
				}
			} else {
				if err != nil || tc.Expected.Delay == nil || delay.Milliseconds() != *tc.Expected.Delay {
					t.Fatalf("delay: %v %v, want %v", delay, err, tc.Expected.Delay)
				}
			}
			attempts := 0
			var sleeps []time.Duration
			timing.sleep = func(_ context.Context, delay time.Duration) error { sleeps = append(sleeps, delay); return nil }
			// Exercise the actual retry gate independently of a delay-cap rejection.
			_, err = retryCompletionsRequest(context.Background(), func() (*http.Response, error) {
				attempts++
				if attempts == 1 {
					return nil, provider
				}
				return &http.Response{}, nil
			}, 1, nil, timing)
			wantAttempts := 1
			if tc.Expected.Retryable {
				wantAttempts = 2
			}
			// A default server delay cap rejects before invoking a second request.
			if tc.Input.Headers["retry-after-ms"] == "60001" || tc.Input.Headers["retry-after-ms"] == "90000" {
				wantAttempts = 1
			}
			if attempts != wantAttempts {
				t.Fatalf("attempts %d want %d (error %v)", attempts, wantAttempts, err)
			}
		})
	}
}

func TestCompletionsRetryLifecycle(t *testing.T) {
	t.Run("ordinary errors are not retried", func(t *testing.T) {
		failure := errors.New("callback error")
		attempts := 0
		_, err := retryCompletionsRequest(context.Background(), func() (*http.Response, error) { attempts++; return nil, failure }, 3, nil, defaultCompletionsRetryTiming())
		if err != failure || attempts != 1 {
			t.Fatalf("retried ordinary error: %d %v", attempts, err)
		}
	})
	t.Run("abort interrupts retry delay", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		timing := defaultCompletionsRetryTiming()
		sleep := timing.sleep
		timing.sleep = func(ctx context.Context, delay time.Duration) error { cancel(); return sleep(ctx, delay) }
		attempts := 0
		_, err := retryCompletionsRequest(ctx, func() (*http.Response, error) {
			attempts++
			return nil, &completionsRequestError{status: 429, headers: http.Header{"Retry-After": []string{"60"}}}
		}, 3, nil, timing)
		if attempts != 1 || err == nil || err.Error() != "Request aborted" {
			t.Fatalf("retry abort: %d %v", attempts, err)
		}
	})
	t.Run("successful request is not replaced by late cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		response := &http.Response{}
		got, err := retryCompletionsRequest(ctx, func() (*http.Response, error) { cancel(); return response, nil }, 3, nil, defaultCompletionsRetryTiming())
		if got != response || err != nil {
			t.Fatalf("success replaced: %p %v", got, err)
		}
	})
	t.Run("bounded retries retain last error", func(t *testing.T) {
		attempts := 0
		delays := []time.Duration{}
		timing := defaultCompletionsRetryTiming()
		timing.random = func() float64 { return 0.5 }
		timing.sleep = func(_ context.Context, delay time.Duration) error { delays = append(delays, delay); return nil }
		failure := &completionsRequestError{status: 500, message: "last"}
		_, err := retryCompletionsRequest(context.Background(), func() (*http.Response, error) { attempts++; return nil, failure }, 3, nil, timing)
		if err != failure || attempts != 4 || !reflect.DeepEqual(delays, []time.Duration{437 * time.Millisecond, 875 * time.Millisecond, 1750 * time.Millisecond}) {
			t.Fatalf("retry history %d %v %v", attempts, delays, err)
		}
	})
}
