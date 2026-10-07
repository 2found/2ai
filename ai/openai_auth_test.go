package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// Local OpenAI-compatible engines normally have no key. Absence must mean no
// Authorization header, not a syntactically present but empty Bearer grant.
func TestOpenAICompatibleEmptyKeyOmitsAuthorizationEverywhere(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("%s Authorization = %q, want header omitted", r.URL.Path, got)
		}
		switch r.URL.Path {
		case "/chat/completions":
			var req oaiRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode request: %v", err)
			}
			if req.Stream {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Write([]byte("data: [DONE]\n\n"))
				return
			}
			w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
		case "/embeddings":
			w.Write([]byte(`{"data":[{"index":0,"embedding":[1]}]}`))
		case "/models":
			w.Write([]byte(`{"data":[{"id":"local-model"}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	p := NewOpenAIProvider("", srv.URL, DefaultCompat())
	p.HTTP = srv.Client()
	p.StreamHTTP = srv.Client()
	if _, err := p.Chat(context.Background(), protocol.ChatRequest{Model: "local"}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "local"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	for range ch {
	}

	e := NewOpenAIEmbedder("", srv.URL, "local-embed")
	e.HTTP = srv.Client()
	if _, err := e.Embed(context.Background(), []string{"hello"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if _, err := listOpenAIModels(context.Background(), srv.Client(), srv.URL, ""); err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(paths) != 4 {
		t.Fatalf("requests = %v, want chat + stream + embeddings + models", paths)
	}
}

func TestOpenAIConcurrentKeyRotationUsesWholeCredential(t *testing.T) {
	var serverErr error
	var errMu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer key-a" && auth != "Bearer key-b" {
			errMu.Lock()
			serverErr = fmt.Errorf("unexpected Authorization header %q", auth)
			errMu.Unlock()
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	p := NewOpenAIProvider("key-a", srv.URL, DefaultCompat())
	p.HTTP = srv.Client()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 500; i++ {
			if i%2 == 0 {
				p.UpdateAPIKey("key-a")
			} else {
				p.UpdateAPIKey("key-b")
			}
		}
	}()
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				if _, err := p.Chat(context.Background(), protocol.ChatRequest{Model: "m"}); err != nil {
					errMu.Lock()
					if serverErr == nil {
						serverErr = err
					}
					errMu.Unlock()
				}
			}
		}()
	}
	wg.Wait()
	errMu.Lock()
	defer errMu.Unlock()
	if serverErr != nil {
		t.Fatal(serverErr)
	}
}

func TestAPIKeyOptionalOnlyForExplicitLocalEngines(t *testing.T) {
	for _, vendor := range []string{"ollama", "LM-Studio", "llama.cpp", "vllm", "localai", "litellm"} {
		if !APIKeyOptional(vendor) {
			t.Errorf("APIKeyOptional(%q) = false", vendor)
		}
	}
	for _, vendor := range []string{"openai", "anthropic", "google", "openai-compat", "remote-gateway"} {
		if APIKeyOptional(vendor) {
			t.Errorf("APIKeyOptional(%q) = true; arbitrary cloud gateways still require explicit credentials", vendor)
		}
	}
}
