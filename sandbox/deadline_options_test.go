package sandbox

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"
)

func TestExplicitUnlimitedNetworkClientsRemainCancellable(t *testing.T) {
	clients := []*http.Client{
		NewHTTPRequestTool(nil, WithHTTPTimeout(0)).client,
		NewWebFetchTool(nil, nil, WithWebFetchTimeout(0)).client,
		NewGuardedClient(-1),
	}
	for _, client := range clients {
		transport := client.Transport.(*http.Transport)
		if client.Timeout != 0 || transport.ResponseHeaderTimeout != 0 || transport.TLSHandshakeTimeout != 0 {
			t.Fatal("unlimited client retained a deadline")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://example.com", nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := client.Do(req); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation lost: %v", err)
		}
	}
	if got := NewWebFetchTool(nil, nil, WithWebFetchTimeout(time.Minute)).client.Timeout; got != time.Minute {
		t.Fatalf("explicit timeout ignored: %v", got)
	}
	if NewWebFetchTool(nil, nil).client.Timeout <= 0 || NewGuardedClient(0).Timeout <= 0 {
		t.Fatal("unrelated callers lost their existing defaults")
	}
}
