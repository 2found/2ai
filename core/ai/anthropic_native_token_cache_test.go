package ai

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
)

func TestPiAnthropicTokenCacheOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-anthropic-token-cache.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, SDKVersion string
		Cases                      []struct {
			Input struct {
				Name      string
				Actions   []json.RawMessage
				Responses []struct {
					Token, Error string
					ExpiresAt    *float64
				}
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || fixture.SDKVersion != "0.129.0" || len(fixture.Cases) != 16 {
		t.Fatal("unexpected cache oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			var now atomic.Int64
			now.Store(1000)
			var mu sync.Mutex
			calls := []bool{}
			cache := &anthropicTokenCache{now: func() float64 { return float64(now.Load()) }}
			cache.provider = func(force bool) (anthropicAccessToken, error) {
				mu.Lock()
				defer mu.Unlock()
				calls = append(calls, force)
				if len(calls) <= len(tc.Input.Responses) {
					response := tc.Input.Responses[len(calls)-1]
					if response.Error != "" {
						return anthropicAccessToken{}, errors.New(response.Error)
					}
					return anthropicAccessToken{response.Token, response.ExpiresAt}, nil
				}
				expires := float64(now.Load() + 1000)
				return anthropicAccessToken{fmt.Sprintf("token-%d", len(calls)), &expires}, nil
			}
			events := []map[string]any{}
			for _, raw := range tc.Input.Actions {
				if len(raw) > 0 && raw[0] != '"' {
					var time int64
					_ = json.Unmarshal(raw, &time)
					now.Store(time)
					continue
				}
				if samplingString(raw) == "invalidate" {
					cache.invalidate()
					continue
				}
				value, err := cache.getToken()
				cache.mu.Lock()
				pending := cache.pending
				cache.mu.Unlock()
				if pending != nil {
					<-pending.done
				}
				var result, failure any
				if err != nil {
					failure = err.Error()
				} else {
					result = value
				}
				mu.Lock()
				copied := append([]bool{}, calls...)
				mu.Unlock()
				events = append(events, map[string]any{"value": result, "error": failure, "calls": copied})
			}
			actual, _ := json.Marshal(events)
			var a, b any
			_ = json.Unmarshal(actual, &a)
			_ = json.Unmarshal(tc.Expected, &b)
			if !reflect.DeepEqual(a, b) {
				t.Fatalf("Go: %s\nSDK: %s", actual, tc.Expected)
			}
		})
	}
}

func TestAnthropicTokenCacheConcurrentAndForcedRefresh(t *testing.T) {
	type call struct {
		force   bool
		release chan struct{}
	}
	started := make(chan call, 4)
	var count atomic.Int32
	cache := &anthropicTokenCache{now: func() float64 { return 1000 }}
	cache.provider = func(force bool) (anthropicAccessToken, error) {
		n := count.Add(1)
		release := make(chan struct{})
		started <- call{force, release}
		<-release
		expires := 2000.0
		return anthropicAccessToken{fmt.Sprintf("token-%d", n), &expires}, nil
	}
	results := make(chan string, 33)
	for i := 0; i < 32; i++ {
		go func() {
			value, err := cache.getToken()
			if err != nil {
				results <- err.Error()
			} else {
				results <- value
			}
		}()
	}
	first := <-started
	if first.force {
		t.Fatal("initial refresh forced")
	}
	close(first.release)
	for i := 0; i < 32; i++ {
		if result := <-results; result != "token-1" {
			t.Fatalf("coalesced result: %s", result)
		}
	}
	if count.Load() != 1 {
		t.Fatal("refresh was not coalesced")
	}
	cache.invalidate()
	go func() { value, _ := cache.getToken(); results <- value }()
	forced := <-started
	if !forced.force {
		t.Fatal("invalidation did not force refresh")
	}
	close(forced.release)
	if result := <-results; result != "token-2" {
		t.Fatal(result)
	}
}

func TestAnthropicTokenCacheForcedRefreshDoesNotJoinPending(t *testing.T) {
	type call struct {
		force   bool
		release chan struct{}
	}
	started := make(chan call, 2)
	var count atomic.Int32
	cache := &anthropicTokenCache{now: func() float64 { return 1000 }}
	cache.provider = func(force bool) (anthropicAccessToken, error) {
		n := count.Add(1)
		release := make(chan struct{})
		started <- call{force, release}
		<-release
		expires := 2000.0
		return anthropicAccessToken{fmt.Sprintf("token-%d", n), &expires}, nil
	}
	firstResult, secondResult := make(chan string, 1), make(chan string, 1)
	go func() { value, _ := cache.getToken(); firstResult <- value }()
	first := <-started
	cache.invalidate()
	go func() { value, _ := cache.getToken(); secondResult <- value }()
	second := <-started
	if first.force || !second.force {
		t.Fatal("forced refresh joined the old request")
	}
	close(second.release)
	if value := <-secondResult; value != "token-2" {
		t.Fatal(value)
	}
	close(first.release)
	if value := <-firstResult; value != "token-1" {
		t.Fatal(value)
	}
	// The pinned SDK lets the older completion replace the cached token.
	// Retain this ordering rather than inventing generation fencing.
	if value, err := cache.getToken(); err != nil || value != "token-1" {
		t.Fatalf("late completion: %q %v", value, err)
	}
}
