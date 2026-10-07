package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func TestNewOpenAIEmbedder_FillsDefaults(t *testing.T) {
	e := NewOpenAIEmbedder("k", "", "")
	if e.BaseURL != defaultOpenAIBaseURL {
		t.Fatalf("base URL = %q, want the vendor default", e.BaseURL)
	}
	if e.Model != defaultOpenAIEmbedModel {
		t.Fatalf("model = %q, want the default embed model", e.Model)
	}
	// A trailing slash would produce "…//embeddings" on every request.
	if e = NewOpenAIEmbedder("k", "https://gw.example/v1/", "m"); e.BaseURL != "https://gw.example/v1" {
		t.Fatalf("base URL = %q, want the trailing slash trimmed", e.BaseURL)
	}
}

// The API may return data out of order, and the contract is that the caller
// gets vectors index-aligned with the texts it passed. Sorting by arrival would
// silently mismatch every embedding with the wrong text.
func TestEmbed_AlignsVectorsToInputIndex(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var in oaiEmbedRequest
		json.NewDecoder(r.Body).Decode(&in)
		if len(in.Input) != 3 {
			t.Errorf("server saw %d inputs, want 3", len(in.Input))
		}
		// Deliberately out of order.
		w.Write([]byte(`{"data":[
			{"index":2,"embedding":[3,3]},
			{"index":0,"embedding":[1,1]},
			{"index":1,"embedding":[2,2]}]}`))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("k", srv.URL, "m")
	got, err := e.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	want := [][]float32{{1, 1}, {2, 2}, {3, 3}}
	for i := range want {
		if len(got[i]) != 2 || got[i][0] != want[i][0] {
			t.Fatalf("vector %d = %v, want %v", i, got[i], want[i])
		}
	}
}

// No inputs must not become a request: an empty embeddings call is a 400 at
// most providers and costs a round trip for nothing.
func TestEmbed_EmptyInputSkipsTheRequest(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("k", srv.URL, "m")
	got, err := e.Embed(context.Background(), nil)
	if err != nil || got != nil {
		t.Fatalf("Embed(nil) = %v, %v; want nil, nil", got, err)
	}
	if called {
		t.Fatal("empty input still issued an HTTP request")
	}
}

// A body severed mid-flight must surface as a retryable ProviderError rather
// than a JSON decode error, which IsRetryable cannot classify — that turns a
// transient truncation into a permanent embeddings failure.
func TestEmbed_TruncatedBodyIsRetryable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Promise more than we send, then hang up: the client's read fails.
		w.Header().Set("Content-Length", "4096")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"data":[{"index":0,"embed`))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		panic(http.ErrAbortHandler) // sever the connection
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("k", srv.URL, "m")
	_, err := e.Embed(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("severed embeddings body decoded as success")
	}
	if !protocol.IsRetryable(err) {
		t.Fatalf("truncated embeddings error is not retryable: %v", err)
	}
	if !strings.Contains(err.Error(), "truncated embeddings body") {
		t.Fatalf("error = %v, want it to name the truncation", err)
	}
}

// A provider-reported error in the JSON envelope must win over the empty data
// array, otherwise the caller gets a slice of nil vectors and no error at all.
func TestEmbed_ProviderErrorEnvelopeSurfaces(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"model not found"}}`))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("k", srv.URL, "m")
	_, err := e.Embed(context.Background(), []string{"a"})
	if err == nil || !strings.Contains(err.Error(), "model not found") {
		t.Fatalf("err = %v, want the provider message", err)
	}
}
