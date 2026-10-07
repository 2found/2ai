package ai

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestPiCodexWebSocketStreamOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-codex-websocket-stream.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name  string
				Idle  int
				Steps []struct {
					Kind   string
					Data   json.RawMessage
					Binary bool
				}
			}
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 22 {
		t.Fatal("unexpected WebSocket stream oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			p := newCodexWebSocketParser()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			for _, step := range tc.Input.Steps {
				switch step.Kind {
				case "message":
					var text string
					if err := json.Unmarshal(step.Data, &text); err != nil {
						t.Fatal(err)
					}
					p.message([]byte(text), step.Binary)
				case "error":
					p.fail(codexSocketEventError(step.Data))
				case "close":
					p.closed(step.Data)
				case "abort":
					cancel()
				}
			}
			events := []json.RawMessage{}
			closes := []any{}
			err := p.run(ctx, time.Duration(tc.Input.Idle)*time.Millisecond, func() { closes = append(closes, map[string]any{"code": 1000, "reason": "idle_timeout"}) }, func(raw json.RawMessage) (bool, error) { events = append(events, raw); return false, nil })
			result := map[string]any{"events": events, "closes": closes}
			if err != nil {
				failure := map[string]any{"message": err.Error()}
				var closed *codexWebSocketCloseError
				if errors.As(err, &closed) {
					if closed.Code != nil {
						failure["code"] = *closed.Code
					}
					if closed.Reason != nil {
						failure["reason"] = *closed.Reason
					}
					if closed.WasClean != nil {
						failure["wasClean"] = *closed.WasClean
					}
				}
				result["error"] = failure
			}
			// Listener removal belongs to the future transport adapter, not this queue.
			var expected map[string]json.RawMessage
			if err := json.Unmarshal(tc.Expected, &expected); err != nil {
				t.Fatal(err)
			}
			delete(expected, "listeners")
			expectedRaw, _ := json.Marshal(expected)
			assertPiJSON(t, expectedRaw, result)
		})
	}
}
func TestCodexWebSocketParserConcurrentDelivery(t *testing.T) {
	p := newCodexWebSocketParser()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	done := make(chan error, 1)
	count := 0
	go func() { done <- p.run(ctx, 0, nil, func(json.RawMessage) (bool, error) { count++; return false, nil }) }()
	for i := 0; i < 100; i++ {
		p.message([]byte(`{"type":"delta"}`), false)
	}
	p.message([]byte(`{"type":"response.done"}`), false)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if count != 101 {
		t.Fatalf("lost queued messages: %d", count)
	}
}
func TestCodexWebSocketParserCancellationAndMalformed(t *testing.T) {
	t.Run("waiting cancellation", func(t *testing.T) {
		p := newCodexWebSocketParser()
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan error, 1)
		go func() {
			done <- p.run(ctx, 0, nil, func(json.RawMessage) (bool, error) { t.Error("unexpected event"); return false, nil })
		}()
		cancel()
		select {
		case err := <-done:
			if err == nil || err.Error() != "Request was aborted" {
				t.Fatal(err)
			}
		case <-time.After(time.Second):
			t.Fatal("cancellation did not wake parser")
		}
	})
	for _, payload := range []string{"{", " null ", "[DONE]"} {
		t.Run(payload, func(t *testing.T) {
			p := newCodexWebSocketParser()
			p.message([]byte(payload), false)
			err := p.run(context.Background(), 0, nil, func(json.RawMessage) (bool, error) { t.Error("invalid message admitted"); return false, nil })
			if err == nil || !strings.HasPrefix(err.Error(), "Invalid Codex WebSocket JSON:") {
				t.Fatal(err)
			}
		})
	}
}
