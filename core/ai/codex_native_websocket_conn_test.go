package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestCodexWebSocketLocalReuse(t *testing.T) {
	var connections atomic.Int32
	requests := make(chan string, 2)
	serverDone := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test" {
			t.Error("missing auth header")
		}
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		connections.Add(1)
		defer conn.Close()
		defer close(serverDone)
		for i := 0; i < 2; i++ {
			_, payload, err := conn.ReadMessage()
			if err != nil {
				t.Error(err)
				return
			}
			requests <- string(payload)
			if err = conn.WriteMessage(websocket.BinaryMessage, []byte(`{"type":"response.created","text":"Chào 🌱"}`)); err != nil {
				t.Error(err)
				return
			}
			if err = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.done"}`)); err != nil {
				t.Error(err)
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
	connect := func(ctx context.Context) (*codexSocket, error) {
		return connectCodexWebSocket(ctx, "ws"+strings.TrimPrefix(server.URL, "http"), http.Header{"Authorization": []string{"Bearer test"}}, time.Second, nil)
	}
	for i := 0; i < 2; i++ {
		lease, err := cache.acquire(ctx, "s", "a", connect)
		if err != nil {
			t.Fatal(err)
		}
		if lease.reused != (i == 1) {
			t.Fatal("unexpected reuse state")
		}
		count := 0
		err = runCodexWebSocketRequest(ctx, lease.socket, []byte(`{"type":"response.create"}`), time.Second, func(raw json.RawMessage) (bool, error) { count++; return false, nil })
		if err != nil {
			lease.release(false)
			t.Fatal(err)
		}
		if count != 2 {
			t.Fatalf("events=%d", count)
		}
		lease.release(true)
		if request := <-requests; request != `{"type":"response.create"}` {
			t.Fatal(request)
		}
	}
	if connections.Load() != 1 {
		t.Fatal("socket not reused")
	}
	cache.closeSessions("s")
	select {
	case <-serverDone:
	case <-ctx.Done():
		t.Fatal("server did not observe close")
	}
}

func TestCodexWebSocketLocalFailure(t *testing.T) {
	for _, mode := range []string{"early-close", "idle", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			received := make(chan struct{})
			serverDone := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
				if err != nil {
					return
				}
				defer conn.Close()
				defer close(serverDone)
				if _, _, err = conn.ReadMessage(); err != nil {
					return
				}
				close(received)
				if mode == "early-close" {
					_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(1009, ""), time.Now().Add(time.Second))
					return
				}
				_, _, _ = conn.ReadMessage()
			}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			socket, err := connectCodexWebSocket(ctx, "ws"+strings.TrimPrefix(server.URL, "http"), nil, time.Second, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer socket.closeSilently("done")
			runCtx, abort := context.WithCancel(ctx)
			defer abort()
			result := make(chan error, 1)
			go func() {
				result <- runCodexWebSocketRequest(runCtx, socket, []byte(`{}`), 20*time.Millisecond, func(json.RawMessage) (bool, error) { return false, nil })
			}()
			<-received
			if mode == "cancel" {
				abort()
			}
			select {
			case err = <-result:
			case <-ctx.Done():
				t.Fatal("request stuck")
			}
			if err == nil {
				t.Fatal("missing failure")
			}
			switch mode {
			case "early-close":
				var closed *codexWebSocketCloseError
				if !errors.As(err, &closed) || closed.Code == nil || *closed.Code != 1009 || err.Error() != "WebSocket closed 1009 message too big" {
					t.Fatal(err)
				}
			case "idle":
				if err.Error() != "WebSocket idle timeout after 20ms" {
					t.Fatal(err)
				}
			case "cancel":
				if err.Error() != "Request was aborted" {
					t.Fatal(err)
				}
			}
			socket.closeSilently("done")
			select {
			case <-serverDone:
			case <-ctx.Done():
				t.Fatal("connection reader leaked")
			}
		})
	}
}
func TestCodexWebSocketHandshakeTimeoutAndCancel(t *testing.T) {
	for _, mode := range []string{"timeout", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			entered := make(chan struct{})
			done := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { close(entered); <-r.Context().Done(); close(done) }))
			defer server.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			result := make(chan error, 1)
			go func() {
				timeout := 30 * time.Millisecond
				if mode == "cancel" {
					timeout = 10 * time.Second
				}
				_, err := connectCodexWebSocket(ctx, "ws"+strings.TrimPrefix(server.URL, "http"), nil, timeout, nil)
				result <- err
			}()
			<-entered
			if mode == "cancel" {
				cancel()
			}
			select {
			case err := <-result:
				expected := "WebSocket connect timeout after 30ms"
				if mode == "cancel" {
					expected = "Request was aborted"
				}
				if err == nil || err.Error() != expected {
					t.Fatalf("got %v, want %s", err, expected)
				}
			case <-time.After(time.Second):
				t.Fatal("handshake stuck")
			}
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("handshake request leaked")
			}
		})
	}
}
