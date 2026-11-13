package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/klauspost/compress/zstd"
)

func TestCodexCombinedTransportLifecycle(t *testing.T) {
	for _, mode := range []string{"websocket", "fallback", "sse"} {
		t.Run(mode, func(t *testing.T) {
			var mu sync.Mutex
			wsCalls, httpCalls := 0, 0
			requests := []json.RawMessage{}
			end := []byte(`{"type":"response.done","response":{"id":"r","status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}`)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if websocket.IsWebSocketUpgrade(r) {
					mu.Lock()
					wsCalls++
					mu.Unlock()
					if mode == "fallback" {
						http.Error(w, "unavailable", 503)
						return
					}
					conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
					if err != nil {
						return
					}
					defer conn.Close()
					for {
						_, raw, err := conn.ReadMessage()
						if err != nil {
							return
						}
						mu.Lock()
						requests = append(requests, raw)
						mu.Unlock()
						if err = conn.WriteMessage(websocket.TextMessage, end); err != nil {
							return
						}
					}
				}
				raw, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
					return
				}
				decoder, err := zstd.NewReader(nil)
				if err != nil {
					t.Error(err)
					return
				}
				defer decoder.Close()
				body, err := decoder.DecodeAll(raw, nil)
				if err != nil {
					t.Error(err)
					return
				}
				mu.Lock()
				httpCalls++
				requests = append(requests, body)
				mu.Unlock()
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "data: %s\n\n", end)
			}))
			defer server.Close()
			session := t.Name()
			defer codexNativePolicy.reset(session)
			defer codexNativeSockets.closeSessions(session)
			token := "h." + base64.StdEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_account_id":"account"}}`)) + ".s"
			modelRaw, _ := json.Marshal(map[string]any{"id": "test", "api": "openai-codex-responses", "provider": "openai-codex", "baseUrl": server.URL, "input": []string{"text"}})
			transport := "auto"
			if mode == "sse" {
				transport = "sse"
			}
			optionsRaw, _ := json.Marshal(map[string]any{"apiKey": token, "transport": transport, "sessionId": session, "timeoutMs": 1000, "websocketConnectTimeoutMs": 1000})
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			payloadCalls, responseCalls := 0, 0
			options := CodexResponsesStreamOptions{Options: optionsRaw, OnPayload: func(context.Context, json.RawMessage, json.RawMessage) (json.RawMessage, error) {
				payloadCalls++
				return json.RawMessage(`{"model":"replacement","store":false,"input":[],"marker":"same-payload"}`), nil
			}, OnResponse: func(context.Context, CompletionsResponse, json.RawMessage) error { responseCalls++; return nil }, Now: func() int64 { return 100 }}
			for call := 0; call < 2; call++ {
				stream := StreamCodexResponses(ctx, modelRaw, NormalizeContext(Context{}), options)
				result, err := stream.Result(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if result.StopReason != "stop" {
					t.Fatalf("stream failed: %+v", result)
				}
				if payloadCalls != call+1 {
					t.Fatalf("payload callback repeated during fallback: %d", payloadCalls)
				}
				var diagnostics []map[string]json.RawMessage
				_ = json.Unmarshal(result.Diagnostics, &diagnostics)
				expected := 0
				if mode == "fallback" && call == 0 {
					expected = 1
				}
				if len(diagnostics) != expected {
					t.Fatalf("diagnostics=%s", result.Diagnostics)
				}
				if expected == 1 {
					var details map[string]any
					_ = json.Unmarshal(diagnostics[0]["details"], &details)
					if details["fallbackTransport"] != "sse" || details["eventsEmitted"] != false {
						t.Fatalf("wrong diagnostic details: %v", details)
					}
				}
			}
			mu.Lock()
			defer mu.Unlock()
			expectedWS, expectedHTTP := 1, 0
			if mode == "fallback" {
				expectedHTTP = 2
			}
			if mode == "sse" {
				expectedWS = 0
				expectedHTTP = 2
			}
			if wsCalls != expectedWS || httpCalls != expectedHTTP || responseCalls != expectedHTTP {
				t.Fatalf("ws=%d http=%d response callbacks=%d", wsCalls, httpCalls, responseCalls)
			}
			if len(requests) != 2 {
				t.Fatalf("requests=%d", len(requests))
			}
			for _, raw := range requests {
				var body map[string]json.RawMessage
				_ = json.Unmarshal(raw, &body)
				if string(body["marker"]) != `"same-payload"` || string(body["model"]) != `"replacement"` {
					t.Fatalf("prepared payload lost: %s", raw)
				}
			}
		})
	}
}
