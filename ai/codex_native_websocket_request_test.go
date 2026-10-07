package ai

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/gorilla/websocket"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestPiCodexWebSocketRequestOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-codex-websocket-request.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          map[string]json.RawMessage
		Cases          []struct {
			Input struct {
				Name, Transport string
				Model, Options  map[string]json.RawMessage
				Compat, Body    json.RawMessage
				Chunks          []json.RawMessage
				CallbackFail    bool
			}
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 110 {
		t.Fatal("unexpected WebSocket request oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			fields := map[string]json.RawMessage{}
			for k, v := range fixture.Model {
				fields[k] = v
			}
			for k, v := range tc.Input.Model {
				fields[k] = v
			}
			if len(tc.Input.Compat) > 0 {
				fields["compat"] = tc.Input.Compat
			}
			modelRaw, _ := json.Marshal(fields)
			var model completionsModel
			if err := json.Unmarshal(modelRaw, &model); err != nil {
				t.Fatal(err)
			}
			stream := NewAssistantMessageEventStream()
			acc := newCodexResponsesAccumulator(model, nil, stream, 123)
			events := []json.RawMessage{}
			acc.push = func(event AssistantMessageEvent) {
				raw, err := json.Marshal(event)
				if err != nil {
					t.Fatal(err)
				}
				events = append(events, raw)
			}
			options := tc.Input.Options
			if options == nil {
				options = map[string]json.RawMessage{}
			}
			options["transport"], _ = json.Marshal(tc.Input.Transport)
			encodedOptions, _ := json.Marshal(options)
			acc.serviceTier = options["serviceTier"]
			cache := newCodexSocketCache()
			defer cache.closeSessions("")
			requests := []json.RawMessage{}
			closes := []any{}
			var parser *codexWebSocketParser
			state := 1
			socket := &codexSocket{ready: func() *int { return &state }, close: func(code int, reason string) {
				state = 3
				closes = append(closes, map[string]any{"code": code, "reason": reason})
			}, listen: func(p *codexWebSocketParser) func() { parser = p; return func() { parser = nil } }, send: func(context.Context, []byte) error { return nil }}
			socket.send = func(_ context.Context, payload []byte) error {
				requests = append(requests, append(json.RawMessage(nil), payload...))
				for _, event := range tc.Input.Chunks {
					parser.message(event, false)
				}
				return nil
			}
			body, _ := json.Marshal(map[string]any{"model": model.ID, "store": false, "input": []any{}})
			if len(tc.Input.Body) > 0 {
				body = tc.Input.Body
			}
			opts := CodexResponsesStreamOptions{Options: encodedOptions}
			if tc.Input.CallbackFail {
				opts.OnProviderStreamEvent = func(context.Context, *json.RawMessage, json.RawMessage) error { return errors.New("callback failed") }
			}
			policy := newCodexTransportPolicy()
			err := processCodexWebSocketStarted(context.Background(), cache, "s", "a", func(context.Context) (*codexSocket, error) { return socket, nil }, body, modelRaw, acc, opts, time.Second, acc.start, policy)
			result := map[string]any{"stats": policy.snapshot("s"), "requests": requests, "events": events, "output": acc.output, "closes": closes, "cached": cache.entries[codexSocketKey{"s", "a"}] != nil}
			if err != nil {
				name := "Error"
				var api *codexAPIError
				var callback *codexProviderCallbackError
				if errors.As(err, &api) {
					name = "CodexApiError"
				}
				if errors.As(err, &callback) {
					name = "ProviderStreamEventCallbackError"
				}
				result["error"] = map[string]any{"message": err.Error(), "name": name}
			}
			if entry := cache.entries[codexSocketKey{"s", "a"}]; entry != nil && entry.continuation != nil {
				result["continuation"] = entry.continuation
			}
			if parser != nil {
				t.Fatal("request listener retained")
			}
			assertPiJSON(t, tc.Expected, result)
		})
	}
}

func TestCodexWebSocketRequestContinuationOnWire(t *testing.T) {
	requests := make(chan json.RawMessage, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for i := 0; i < 2; i++ {
			_, raw, err := conn.ReadMessage()
			if err != nil {
				return
			}
			requests <- raw
			if err = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.done","response":{"id":"r1","status":"completed","output":[]}}`)); err != nil {
				return
			}
		}
		_, _, _ = conn.ReadMessage()
	}))
	defer server.Close()
	cache := newCodexSocketCache()
	defer cache.closeSessions("")
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	modelRaw := json.RawMessage(`{"id":"test","api":"openai-codex-responses","provider":"openai-codex","input":["text"]}`)
	var model completionsModel
	_ = json.Unmarshal(modelRaw, &model)
	connect := func(ctx context.Context) (*codexSocket, error) {
		return connectCodexWebSocket(ctx, "ws"+strings.TrimPrefix(server.URL, "http"), nil, time.Second, nil)
	}
	bodies := []json.RawMessage{json.RawMessage(`{"model":"test","store":false,"input":[{"role":"user","content":"first"}]}`), json.RawMessage(`{"model":"test","store":false,"input":[{"role":"user","content":"first"},{"role":"user","content":"next"}]}`)}
	for _, body := range bodies {
		acc := newCodexResponsesAccumulator(model, nil, NewAssistantMessageEventStream(), 0)
		err := processCodexWebSocket(ctx, cache, "s", "a", connect, body, modelRaw, acc, CodexResponsesStreamOptions{Options: json.RawMessage(`{"transport":"websocket-cached"}`)}, time.Second)
		if err != nil {
			t.Fatal(err)
		}
	}
	first, second := <-requests, <-requests
	assertPiJSON(t, json.RawMessage(`{"type":"response.create","model":"test","store":false,"input":[{"role":"user","content":"first"}]}`), first)
	assertPiJSON(t, json.RawMessage(`{"type":"response.create","model":"test","store":false,"input":[{"role":"user","content":"next"}],"previous_response_id":"r1"}`), second)
	entry := cache.entries[codexSocketKey{"s", "a"}]
	if entry == nil || entry.busy || entry.continuation == nil {
		t.Fatal("continuation not retained on idle socket")
	}
	assertPiJSON(t, bodies[1], entry.continuation.LastRequestBody)
}

func TestCodexWebSocketCallbackPanicReleasesLease(t *testing.T) {
	cache := newCodexSocketCache()
	defer cache.closeSessions("")
	var parser *codexWebSocketParser
	closed := false
	socket := &codexSocket{close: func(int, string) { closed = true }, listen: func(p *codexWebSocketParser) func() { parser = p; return func() { parser = nil } }, send: func(context.Context, []byte) error {
		parser.message([]byte(`{"type":"response.done"}`), false)
		return nil
	}}
	acc := newCodexResponsesAccumulator(completionsModel{}, nil, NewAssistantMessageEventStream(), 0)
	func() {
		defer func() {
			if recover() != "callback panic" {
				t.Error("callback panic not propagated")
			}
		}()
		_ = processCodexWebSocket(context.Background(), cache, "s", "a", func(context.Context) (*codexSocket, error) { return socket, nil }, json.RawMessage(`{}`), json.RawMessage(`{}`), acc, CodexResponsesStreamOptions{OnProviderStreamEvent: func(context.Context, *json.RawMessage, json.RawMessage) error { panic("callback panic") }}, time.Second)
	}()
	if !closed || len(cache.entries) != 0 || parser != nil {
		t.Fatal("panic retained socket, cache entry or listener")
	}
}
