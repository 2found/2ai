package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func adaptiveRequestFixture(session *protocol.ProviderSession) protocol.ChatRequest {
	return protocol.ChatRequest{
		Model: "reasoning-model", ProviderSession: session,
		Messages:        []protocol.Message{{Role: protocol.RoleUser, Content: "hello"}},
		ReasoningEffort: "high", CacheKey: "conversation", CacheRetention: "long",
		OutputSchema: verdictSchema(),
	}
}

func TestOpenAIChatLearnsUnsupportedHintWithinProviderSession(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		if _, has := body["reasoning_effort"]; has {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"Unsupported parameter: reasoning_effort"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"{\"verdict\":\"ok\"}"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	p := NewOpenAIProvider("k", srv.URL, DefaultCompat())
	session := protocol.NewProviderSession()
	defer session.Close()
	req := adaptiveRequestFixture(session)
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	// Runtime rebuilds providers for the next short-lived Agent/run. The lesson
	// must live in ProviderSession, not on the old wire client.
	p = NewOpenAIProvider("k", srv.URL, DefaultCompat())
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("second Chat: %v", err)
	}

	mu.Lock()
	got := append([]map[string]any(nil), bodies...)
	mu.Unlock()
	if len(got) != 3 {
		t.Fatalf("wire calls = %d, want rejected + fallback + learned call = 3", len(got))
	}
	if _, ok := got[0]["reasoning_effort"]; !ok {
		t.Fatal("first request did not carry reasoning_effort")
	}
	for i := 1; i < len(got); i++ {
		if _, ok := got[i]["reasoning_effort"]; ok {
			t.Fatalf("request %d retained learned-unsupported reasoning_effort: %+v", i+1, got[i])
		}
		if _, ok := got[i]["prompt_cache_key"]; !ok {
			t.Fatalf("request %d lost unrelated prompt_cache_key: %+v", i+1, got[i])
		}
		if _, ok := got[i]["response_format"]; !ok {
			t.Fatalf("request %d lost unrelated response_format: %+v", i+1, got[i])
		}
	}

	// A different logical conversation does not inherit private request history.
	fresh := protocol.NewProviderSession()
	defer fresh.Close()
	p = NewOpenAIProvider("k", srv.URL, DefaultCompat())
	if _, err := p.Chat(context.Background(), adaptiveRequestFixture(fresh)); err != nil {
		t.Fatalf("fresh-session Chat: %v", err)
	}
	mu.Lock()
	got = append([]map[string]any(nil), bodies...)
	mu.Unlock()
	if len(got) != 5 {
		t.Fatalf("fresh session made %d total wire calls, want a new rejected + fallback pair", len(got))
	}
	if _, ok := got[3]["reasoning_effort"]; !ok {
		t.Fatal("fresh session incorrectly inherited the prior session's demotion")
	}
}

func TestOpenAIChatLearnsMaxCompletionTokenDialect(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		if _, legacy := body["max_tokens"]; legacy {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"Unsupported parameter: max_tokens; use max_completion_tokens"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	p := NewOpenAIProvider("key", srv.URL, DefaultCompat())
	p.HTTP = srv.Client()
	session := protocol.NewProviderSession()
	req := protocol.ChatRequest{
		Model: "reasoning-model", MaxTokens: 2048, ProviderSession: session,
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "hello"}},
	}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("second Chat: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 3 {
		t.Fatalf("requests = %d, want legacy rejection + retry + learned call", len(bodies))
	}
	if _, ok := bodies[0]["max_tokens"]; !ok {
		t.Fatalf("first request did not use max_tokens: %v", bodies[0])
	}
	for i := 1; i < 3; i++ {
		if _, ok := bodies[i]["max_completion_tokens"]; !ok {
			t.Fatalf("request %d did not use learned max_completion_tokens: %v", i+1, bodies[i])
		}
		if _, legacy := bodies[i]["max_tokens"]; legacy {
			t.Fatalf("request %d retained max_tokens: %v", i+1, bodies[i])
		}
	}
}

func TestOpenAIStreamRetriesUnsupportedHintBeforeExposingOutput(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if _, has := body["prompt_cache_key"]; has {
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = w.Write([]byte(`{"error":{"message":"unknown parameter prompt_cache_key"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	p := NewOpenAIProvider("k", srv.URL, DefaultCompat())
	session := protocol.NewProviderSession()
	defer session.Close()
	req := adaptiveRequestFixture(session)
	req.ReasoningEffort = ""
	req.OutputSchema = nil
	ch, err := p.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var text string
	for delta := range ch {
		if delta.Err != nil {
			t.Fatalf("stream delta: %v", delta.Err)
		}
		text += delta.ContentDelta
	}
	if text != "ok" {
		t.Fatalf("stream text = %q, want ok", text)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("wire calls = %d, want rejected + fallback = 2", got)
	}

	ch, err = p.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("learned Stream: %v", err)
	}
	for range ch {
	}
	if got := atomic.LoadInt32(&calls); got != 3 {
		t.Fatalf("learned stream made %d total calls, want one additional call", got)
	}
}

func TestOpenAIAdaptiveFallbackDoesNotHideInvalidSchema(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"Invalid schema for response_format: required is missing"}}`))
	}))
	defer srv.Close()

	p := NewOpenAIProvider("k", srv.URL, DefaultCompat())
	_, err := p.Chat(context.Background(), adaptiveRequestFixture(protocol.NewProviderSession()))
	if err == nil {
		t.Fatal("invalid schema was silently retried without response_format")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("invalid schema wire calls = %d, want 1", got)
	}
}

type accountResetProbe struct {
	resets int32
	closes int32
}

func (p *accountResetProbe) Close()              { atomic.AddInt32(&p.closes, 1) }
func (p *accountResetProbe) ResetAccountScoped() { atomic.AddInt32(&p.resets, 1) }

type endpointResetProbe struct{ closes int32 }

func (p *endpointResetProbe) Close() { atomic.AddInt32(&p.closes, 1) }

func TestOAuthAccountSwitchResetsOnlyAccountScopedProviderState(t *testing.T) {
	session := protocol.NewProviderSession()
	defer session.Close()
	account := &accountResetProbe{}
	endpoint := &endpointResetProbe{}
	session.State("chain", func() protocol.ProviderSessionState { return account })
	session.State("endpoint", func() protocol.ProviderSessionState { return endpoint })

	bindOAuthProviderSession(session, VendorOpenAICodex, "provider-row-1", OAuthToken{AccountID: "acct-1"})
	bindOAuthProviderSession(session, VendorOpenAICodex, "provider-row-1", OAuthToken{AccountID: "acct-1"})
	if got := atomic.LoadInt32(&account.resets); got != 0 {
		t.Fatalf("same account reset state %d times", got)
	}
	bindOAuthProviderSession(session, VendorOpenAICodex, "provider-row-1", OAuthToken{AccountID: "acct-2"})
	if got := atomic.LoadInt32(&account.resets); got != 1 {
		t.Fatalf("rotated account resets = %d, want 1", got)
	}
	if got := atomic.LoadInt32(&endpoint.closes); got != 0 {
		t.Fatalf("endpoint lesson was closed on account rotation: %d", got)
	}
	// A second configured provider row may use the same vendor with an entirely
	// different account pool. Its first bind is not a rotation of row 1.
	bindOAuthProviderSession(session, VendorOpenAICodex, "provider-row-2", OAuthToken{AccountID: "acct-9"})
	if got := atomic.LoadInt32(&account.resets); got != 1 {
		t.Fatalf("independent provider pool caused a false account reset: %d", got)
	}
}
