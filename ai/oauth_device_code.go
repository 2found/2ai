package ai

import (
	"context"
	"errors"
	"math"
	"reflect"
	"time"

	"github.com/2found/2ai/jsonjs"
)

const deviceCodeCancelMessage = "Login cancelled"
const deviceCodeTimeoutMessage = "Device flow timed out"
const deviceCodeSlowTimeoutMessage = "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again."

type OAuthDeviceCodePollResult struct {
	Status          string
	Value           any
	Message         string
	IntervalSeconds any
}

// OAuthDeviceCodePollOptions keeps polling callbacks and cancellation live.
// Numeric controls accept JSON values to preserve Pi's coercion and number-only
// expiry/server-interval checks. Nil/Undefined mean omitted for initial timing.
// Now and Sleep are concrete native clock dependencies; defaults use real time.
type OAuthDeviceCodePollOptions struct {
	IntervalSeconds     any
	ExpiresInSeconds    any
	WaitBeforeFirstPoll bool
	Poll                func() (OAuthDeviceCodePollResult, error)
	Context             context.Context
	Now                 func() float64
	Sleep               func(float64, context.Context, string) error
}

// AbortableSleep replaces the abort cause with the requested cancellation
// message, as Pi does. Delay conversion follows the native timer boundary.
func AbortableSleep(ms float64, ctx context.Context, cancelMessage string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if ctx.Err() != nil {
		return errors.New(cancelMessage)
	}
	if math.IsNaN(ms) || ms < 1 || ms > math.MaxInt32 {
		ms = 1
	}
	timer := time.NewTimer(time.Duration(math.Floor(ms)) * time.Millisecond)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return errors.New(cancelMessage)
	case <-timer.C:
		return nil
	}
}

func deviceCodeNumber(value any) (float64, bool) {
	if jsonjs.IsNullish(value) {
		return 0, false
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(v.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return float64(v.Uint()), true
	case reflect.Float32, reflect.Float64:
		return v.Float(), true
	}
	return 0, false
}

func PollOAuthDeviceCodeFlow(options *OAuthDeviceCodePollOptions) (any, error) {
	return invokeAuth(func() (any, error) {
		now, sleep := options.Now, options.Sleep
		if now == nil {
			now = func() float64 { return float64(time.Now().UnixMilli()) }
		}
		if sleep == nil {
			sleep = AbortableSleep
		}
		signal := func() context.Context {
			if options.Context != nil {
				return options.Context
			}
			return context.Background()
		}
		deadline := math.Inf(1)
		if expiry, ok := deviceCodeNumber(options.ExpiresInSeconds); ok {
			deadline = now() + expiry*1000
		}
		intervalSeconds := options.IntervalSeconds
		if jsonjs.IsNullish(intervalSeconds) {
			intervalSeconds = 5
		}
		interval, err := authNumber(intervalSeconds)
		if err != nil {
			return nil, err
		}
		interval = math.Max(1000, math.Floor(interval*1000))
		slowDown := false
		if options.WaitBeforeFirstPoll {
			if remaining := deadline - now(); remaining > 0 {
				if err := sleep(math.Min(interval, remaining), signal(), deviceCodeCancelMessage); err != nil {
					return nil, err
				}
			}
		}
		for now() < deadline {
			if signal().Err() != nil {
				return nil, errors.New(deviceCodeCancelMessage)
			}
			result, err := options.Poll()
			if err != nil {
				return nil, err
			}
			switch result.Status {
			case "complete":
				return result.Value, nil
			case "failed":
				return nil, errors.New(result.Message)
			case "slow_down":
				slowDown = true
				seconds, numeric := deviceCodeNumber(result.IntervalSeconds)
				if numeric && !math.IsNaN(seconds) && !math.IsInf(seconds, 0) && seconds > 0 {
					interval = math.Max(1000, math.Floor(seconds*1000))
				} else {
					interval = math.Max(1000, interval+5000)
				}
			}
			remaining := deadline - now()
			if remaining <= 0 {
				break
			}
			if err := sleep(math.Min(interval, remaining), signal(), deviceCodeCancelMessage); err != nil {
				return nil, err
			}
		}
		if slowDown {
			return nil, errors.New(deviceCodeSlowTimeoutMessage)
		}
		return nil, errors.New(deviceCodeTimeoutMessage)
	})
}
