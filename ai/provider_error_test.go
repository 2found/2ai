package ai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

func streamError(t *testing.T, ch <-chan protocol.ChatDelta) error {
	t.Helper()
	var got error
	for delta := range ch {
		if delta.Err != nil {
			got = delta.Err
		}
	}
	return got
}

func TestOpenAIStreamClassifiesInBandRateLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Retry-After-Ms", "2500")
		_, _ = w.Write([]byte("data: {\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow down\"}}\n\n"))
	}))
	defer srv.Close()

	p := NewOpenAIProvider("key", srv.URL, DefaultCompat())
	p.StreamHTTP = srv.Client()
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	err = streamError(t, ch)
	var providerErr *protocol.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("stream error = %T %v, want ProviderError", err, err)
	}
	if providerErr.Status != http.StatusTooManyRequests || providerErr.RetryAfter != 2500*time.Millisecond {
		t.Fatalf("provider error = %+v, want status=429 retry_after=2.5s", providerErr)
	}
	if !protocol.IsRetryable(err) {
		t.Fatalf("in-band rate limit is not retryable: %v", err)
	}
}

func TestOpenAIStreamInBandQuotaIsTerminal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"error\":{\"type\":\"insufficient_quota\",\"message\":\"billing limit\"}}\n\n"))
	}))
	defer srv.Close()

	p := NewOpenAIProvider("key", srv.URL, DefaultCompat())
	p.StreamHTTP = srv.Client()
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	err = streamError(t, ch)
	if err == nil || protocol.IsRetryable(err) {
		t.Fatalf("quota error = %v, want terminal", err)
	}
}

func TestAnthropicStreamClassifiesInBandOverload(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n"))
	}))
	defer srv.Close()

	p := NewAnthropicProvider("key", srv.URL)
	p.StreamHTTP = srv.Client()
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	err = streamError(t, ch)
	var providerErr *protocol.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Status != http.StatusServiceUnavailable {
		t.Fatalf("stream error = %T %+v, want ProviderError status=503", err, providerErr)
	}
	if !protocol.IsRetryable(err) {
		t.Fatalf("in-band overload is not retryable: %v", err)
	}
}

func TestOpenAIStreamRequiresTerminalEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"))
	}))
	defer srv.Close()

	p := NewOpenAIProvider("key", srv.URL, DefaultCompat())
	p.StreamHTTP = srv.Client()
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	err = streamError(t, ch)
	if err == nil || !protocol.IsRetryable(err) {
		t.Fatalf("truncated stream error = %v, want retryable", err)
	}
}

func TestAnthropicStreamRequiresMessageStop(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"))
	}))
	defer srv.Close()

	p := NewAnthropicProvider("key", srv.URL)
	p.StreamHTTP = srv.Client()
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	err = streamError(t, ch)
	if err == nil || !protocol.IsRetryable(err) {
		t.Fatalf("truncated stream error = %v, want retryable", err)
	}
}

func TestProviderHTTPErrorBodyIsBounded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(strings.Repeat("x", maxProviderErrorBody*2)))
	}))
	defer srv.Close()

	p := NewOpenAIProvider("key", srv.URL, DefaultCompat())
	p.StreamHTTP = srv.Client()
	_, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err == nil {
		t.Fatal("Stream succeeded, want HTTP error")
	}
	if !strings.Contains(err.Error(), "provider error body truncated") {
		t.Fatalf("error body was not visibly bounded: len=%d", len(err.Error()))
	}
	if len(err.Error()) > maxProviderErrorBody+256 {
		t.Fatalf("error retained too much body: len=%d", len(err.Error()))
	}
}

func TestInBandProviderErrorAcceptsFlatAndCamelCaseSignals(t *testing.T) {
	for _, payload := range []string{
		`{"code":429,"message":"busy"}`,
		`{"type":"ThrottlingAllocationQuota","message":"busy"}`,
		`{"error":{"type":"rateLimitError","message":"busy"}}`,
	} {
		err, ok := inBandProviderError("compat", nil, []byte(payload))
		if !ok || err.Status != http.StatusTooManyRequests || !protocol.IsRetryable(err) {
			t.Errorf("payload %s => ok=%v err=%+v, want retryable 429", payload, ok, err)
		}
	}
	if err, ok := inBandProviderError("compat", nil, []byte(`{"choices":[],"error":null}`)); ok || err != nil {
		t.Fatalf("null error field classified as failure: ok=%v err=%v", ok, err)
	}
}

func TestInBandProviderErrorClassifiesStructuredAuthentication(t *testing.T) {
	for payload, want := range map[string]int{
		`{"type":"error","error":{"type":"authentication_error","message":"expired"}}`: http.StatusUnauthorized,
		`{"error":{"code":"permission_denied","message":"not allowed"}}`:               http.StatusForbidden,
		`{"error":{"status":401,"message":"expired"}}`:                                 http.StatusUnauthorized,
	} {
		err, ok := inBandProviderError("compat", nil, []byte(payload))
		if !ok || err.Status != want {
			t.Errorf("payload %s => ok=%v status=%v, want %d", payload, ok, err, want)
		}
	}
	if err, ok := inBandProviderError("compat", nil, []byte(`authentication failed with 401`)); ok || err != nil {
		t.Fatalf("free-form auth prose classified as failure: ok=%v err=%v", ok, err)
	}
}

func TestProviderErrorWithStatusPreservesRetryHeaders(t *testing.T) {
	resp := &http.Response{StatusCode: http.StatusOK, Header: make(http.Header)}
	resp.Header.Set("Retry-After-Ms", "1750")
	err := providerErrorWithStatus("stream", resp, http.StatusTooManyRequests, "busy")
	if err.Status != http.StatusTooManyRequests || err.RetryAfter != 1750*time.Millisecond {
		t.Fatalf("provider error = %+v, want 429 with 1.75s retry hint", err)
	}
}
