package ai

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"strconv"
	"testing"
	"time"
)

type oauthDeviceCodeFixture struct {
	UpstreamCommit string
	Cases          []struct {
		Input, Log, Output json.RawMessage
		Now                float64
	}
	Sleeps []struct {
		Mode   string
		Output json.RawMessage
	}
	PKCE []struct {
		Kind   string
		Bytes  []byte
		Calls  int
		Output PKCE
	}
}

func readOAuthDeviceCodeFixture(t *testing.T) oauthDeviceCodeFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-device-code.json")
	if err != nil {
		t.Fatal(err)
	}
	var f oauthDeviceCodeFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 127 || len(f.Sleeps) != 3 || len(f.PKCE) != 5 {
		t.Fatalf("unexpected device-code coverage: %d", len(f.Cases))
	}
	return f
}
func deviceCodeFixtureValue(value any) any {
	if object, ok := value.(*Object); ok {
		switch object.Get("special") {
		case "undefined":
			return Undefined
		case "NaN":
			return math.NaN()
		case "Infinity":
			return math.Inf(1)
		case "-Infinity":
			return math.Inf(-1)
		}
	}
	return value
}
func TestPiOAuthDeviceCodeFlow(t *testing.T) {
	for index, tc := range readOAuthDeviceCodeFixture(t).Cases {
		t.Run(strconv.Itoa(index), func(t *testing.T) {
			input := catalogDecode(t, tc.Input).(*Object)
			steps := input.Get("steps").(*Array)
			step := 0
			now := float64(1000)
			log := NewArray()
			value := NewObject(Property{Name: "token", Value: "value"})
			failure := errors.New("poll failed")
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			if input.Get("preabort") == true {
				cancel(errors.New("original cause"))
			}
			options := &OAuthDeviceCodePollOptions{Context: ctx, IntervalSeconds: deviceCodeFixtureValue(input.Get("interval")), ExpiresInSeconds: deviceCodeFixtureValue(input.Get("expires")), WaitBeforeFirstPoll: input.Get("wait") == true, Now: func() float64 { return now }}
			options.Sleep = func(ms float64, signal context.Context, message string) error {
				if signal.Err() != nil {
					return errors.New(message)
				}
				var recorded any = ms
				if math.IsNaN(ms) {
					recorded = "NaN"
				}
				if math.IsInf(ms, 1) {
					recorded = "Infinity"
				}
				log.Append(NewObject(Property{Name: "sleep", Value: recorded}))
				if input.Get("abortSleep") == true {
					cancel(errors.New("sleep cause"))
					return errors.New(message)
				}
				if math.IsNaN(ms) || math.IsInf(ms, 0) || ms < 1 || ms > math.MaxInt32 {
					now++
				} else {
					now += math.Floor(ms)
				}
				return nil
			}
			options.Poll = func() (OAuthDeviceCodePollResult, error) {
				log.Append(NewObject(Property{Name: "poll", Value: now}))
				if step >= steps.Len() {
					return OAuthDeviceCodePollResult{}, errors.New("Unexpected extra poll")
				}
				entry := steps.Get(step).(*Object)
				step++
				if entry.Get("action") == "abort" {
					cancel(errors.New("poll cause"))
				}
				if entry.Get("action") == "expire" {
					now += 2000
				}
				if entry.Get("mutate") == true {
					options.IntervalSeconds = 30
					options.ExpiresInSeconds = 0
					options.WaitBeforeFirstPoll = true
				}
				if entry.Get("replacePoll") == true {
					options.Poll = func() (OAuthDeviceCodePollResult, error) {
						log.Append(NewObject(Property{Name: "replacement", Value: now}))
						return OAuthDeviceCodePollResult{Status: "complete", Value: value}, nil
					}
				}
				if entry.Get("replaceSignal") == true {
					replacement, stop := context.WithCancel(context.Background())
					stop()
					options.Context = replacement
				}
				if entry.Get("action") == "throw" {
					panic(failure)
				}
				if entry.Get("action") == "reject" {
					return OAuthDeviceCodePollResult{}, failure
				}
				message, _ := entry.Get("message").(string)
				return OAuthDeviceCodePollResult{Status: entry.Get("status").(string), Value: value, Message: message, IntervalSeconds: deviceCodeFixtureValue(entry.Get("interval"))}, nil
			}
			result, err := PollOAuthDeviceCodeFlow(options)
			var output any
			if err != nil {
				output = map[string]any{"error": err.Error(), "same": err == failure}
			} else {
				output = map[string]any{"value": result, "same": result == value}
			}
			catalogCompare(t, output, tc.Output)
			catalogCompare(t, log, tc.Log)
			if now != tc.Now {
				t.Fatalf("clock: %v, expected %v", now, tc.Now)
			}
		})
	}
}
func TestPiOAuthAbortableSleep(t *testing.T) {
	for _, tc := range readOAuthDeviceCodeFixture(t).Sleeps {
		t.Run(tc.Mode, func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			cause := errors.New("source reason")
			if tc.Mode == "preabort" {
				cancel(cause)
			}
			ms := float64(10000)
			if tc.Mode == "complete" {
				ms = 0
			}
			if tc.Mode == "abort" {
				timer := time.AfterFunc(10*time.Millisecond, func() { cancel(cause) })
				defer timer.Stop()
			}
			err := AbortableSleep(ms, ctx, "custom cancelled")
			var output any = map[string]any{"success": true}
			if err != nil {
				if errors.Is(err, cause) {
					t.Fatal("abort cause was forwarded")
				}
				output = map[string]any{"error": err.Error()}
			}
			catalogCompare(t, output, tc.Output)
		})
	}
}
func TestOAuthDeviceCodeWaitsForInFlightPoll(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	value := NewObject()
	result := make(chan any, 1)
	failure := make(chan error, 1)
	go func() {
		v, err := PollOAuthDeviceCodeFlow(&OAuthDeviceCodePollOptions{Context: ctx, Poll: func() (OAuthDeviceCodePollResult, error) {
			close(started)
			<-release
			return OAuthDeviceCodePollResult{Status: "complete", Value: value}, nil
		}})
		result <- v
		failure <- err
	}()
	<-started
	cancel()
	select {
	case <-result:
		t.Fatal("cancelled a poll owned by the provider")
	default:
	}
	close(release)
	select {
	case got := <-result:
		if got != value {
			t.Fatal("completion identity lost")
		}
	case <-time.After(time.Second):
		t.Fatal("poll did not complete")
	}
	if err := <-failure; err != nil {
		t.Fatal(err)
	}
}
