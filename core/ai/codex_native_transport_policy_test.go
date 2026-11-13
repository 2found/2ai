package ai

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestPiCodexTransportPolicyOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-codex-transport-policy.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name, Transport string
				Session         *string
				Scenarios       []string
				Calls           int
				Reset           bool
			}
			Expected []json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 19 {
		t.Fatal("unexpected transport policy oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			policy := newCodexTransportPolicy()
			session := "s"
			if tc.Input.Session != nil {
				session = *tc.Input.Session
			}
			attempts, sse := 0, 0
			calls := tc.Input.Calls
			if calls == 0 {
				calls = 1
			}
			for call := 0; call < calls; call++ {
				if tc.Input.Reset && call > 0 {
					policy.reset("s")
				}
				ctx, cancel := context.WithCancel(context.Background())
				before := attempts
				starts := 0
				diagnostics := []any{}
				fallback, err := policy.run(ctx, session, tc.Input.Transport, func(start func()) error {
					scenario := "success"
					if attempts < len(tc.Input.Scenarios) {
						scenario = tc.Input.Scenarios[attempts]
					}
					attempts++
					policy.recordRequest(session, false, tc.Input.Transport == "" || tc.Input.Transport == "auto" || tc.Input.Transport == "websocket-cached", json.RawMessage(`{"input":[],"store":false}`))
					if strings.HasPrefix(scenario, "after-") {
						start()
					}
					switch {
					case scenario == "abort":
						cancel()
						return errors.New("Request was aborted")
					case strings.HasSuffix(scenario, "network"):
						return errors.New("network failed")
					case strings.HasSuffix(scenario, "limit"):
						return &codexAPIError{Code: "websocket_connection_limit_reached", Message: "Codex error: websocket_connection_limit_reached"}
					case strings.HasSuffix(scenario, "missing"):
						return &codexAPIError{Code: "previous_response_not_found", Message: "Codex error: previous_response_not_found"}
					case scenario == "api":
						return &codexAPIError{Code: "invalid_request_error", Message: "Codex error: invalid_request_error"}
					case scenario == "callback":
						return &codexProviderCallbackError{errors.New("callback failed")}
					default:
						start()
						return nil
					}
				}, func() { starts++ }, func(_ error, started bool) {
					phase := "before_message_stream_start"
					if started {
						phase = "after_message_stream_start"
					}
					d := map[string]any{"type": "provider_transport_failure", "eventsEmitted": started, "phase": phase}
					if !started {
						d["fallbackTransport"] = "sse"
					}
					diagnostics = append(diagnostics, d)
				}, nil)
				if fallback {
					sse++
					starts++
				}
				stop := "stop"
				if err != nil {
					stop = "error"
					if ctx.Err() != nil {
						stop = "aborted"
					}
				}
				result := map[string]any{"attempts": attempts - before, "sse": sse, "starts": starts, "stopReason": stop, "diagnostics": diagnostics}
				if err != nil {
					result["error"] = err.Error()
				}
				if stats := policy.snapshot(session); stats != nil {
					result["stats"] = stats
				}
				assertPiJSON(t, tc.Expected[call], result)
				cancel()
			}
		})
	}
}
func TestCodexProtocolErrorsDoNotFallback(t *testing.T) {
	p := newCodexTransportPolicy()
	failure := &codexProtocolError{Message: "invalid JSON"}
	fallback, err := p.run(context.Background(), "s", "auto", func(func()) error { return failure }, nil, func(error, bool) { t.Error("protocol error classified as transport") }, nil)
	if fallback || err != failure || p.disabled("s") {
		t.Fatal("protocol error enabled fallback")
	}
}

func TestCodexTransportRetryOwnsFreshSocketAndOneStart(t *testing.T) {
	policy := newCodexTransportPolicy()
	cache := newCodexSocketCache()
	defer cache.closeSessions("")
	connects, closes, starts := 0, 0, 0
	connect := func(context.Context) (*codexSocket, error) {
		connects++
		attempt := connects
		var parser *codexWebSocketParser
		return &codexSocket{close: func(int, string) { closes++ }, listen: func(p *codexWebSocketParser) func() { parser = p; return func() { parser = nil } }, send: func(context.Context, []byte) error {
			if attempt == 1 {
				parser.message([]byte(`{"type":"response.created","response":{"id":"old"}}`), false)
				parser.message([]byte(`{"type":"error","code":"previous_response_not_found","message":"missing"}`), false)
			} else {
				parser.message([]byte(`{"type":"response.done","response":{"id":"new","status":"completed","output":[]}}`), false)
			}
			return nil
		}}, nil
	}
	rawModel := json.RawMessage(`{"id":"test","api":"openai-codex-responses","provider":"openai-codex","input":["text"]}`)
	var model completionsModel
	_ = json.Unmarshal(rawModel, &model)
	acc := newCodexResponsesAccumulator(model, nil, NewAssistantMessageEventStream(), 0)
	acc.push = func(e AssistantMessageEvent) {
		if e.Type == "start" {
			starts++
		}
	}
	fallback, err := runCodexWebSocketTransport(context.Background(), policy, cache, "s", "a", connect, json.RawMessage(`{"model":"test","input":[]}`), rawModel, acc, CodexResponsesStreamOptions{Options: json.RawMessage(`{"transport":"auto"}`)}, 0, func(error, bool) { t.Error("retryable continuation failure recorded as transport failure") }, nil)
	if err != nil || fallback || connects != 2 || closes != 1 || starts != 1 {
		t.Fatalf("err=%v fallback=%v connects=%d closes=%d starts=%d", err, fallback, connects, closes, starts)
	}
	entry := cache.entries[codexSocketKey{"s", "a"}]
	if entry == nil || entry.continuation == nil || entry.continuation.LastResponseID != "new" {
		t.Fatal("replacement continuation not retained")
	}
}
