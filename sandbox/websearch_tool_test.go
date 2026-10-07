package sandbox

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const ddgFixture = `<html><body>
<div class="result results_links">
  <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone">Example One</a></h2>
  <a class="result__snippet">First <b>result</b> snippet</a>
</div>
<div class="result results_links">
  <h2 class="result__title"><a class="result__a" href="https://example.com/two">Example Two</a></h2>
  <a class="result__snippet">Second result snippet</a>
</div>
</body></html>`

func TestWebSearchRendersRankedResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("q"); got != "agentray" {
			t.Errorf("query param = %q, want agentray", got)
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(ddgFixture))
	}))
	defer srv.Close()

	p := NewDuckDuckGoSearch(nil).(*duckDuckGoSearch)
	p.endpoint = srv.URL
	tool := NewWebSearchTool(p, 0)
	tool.AllowAllIPsForTest() // httptest is on loopback, normally refused

	out, err := tool.Run(context.Background(), `{"query":"agentray"}`)
	if err != nil {
		t.Fatalf("web_search Run: %v", err)
	}
	if !strings.Contains(out, "1. Example One") || !strings.Contains(out, "https://example.com/one") {
		t.Fatalf("first result missing or uddg not resolved: %q", out)
	}
	if !strings.Contains(out, "2. Example Two") || !strings.Contains(out, "https://example.com/two") {
		t.Fatalf("second result missing: %q", out)
	}
	if !strings.Contains(out, "First result snippet") {
		t.Fatalf("snippet missing (markup should be stripped): %q", out)
	}
}

func TestWebSearchRefusesLoopbackSSRF(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(ddgFixture))
	}))
	defer srv.Close()

	// No AllowAllIPsForTest: the guarded dialer must refuse loopback even
	// though the endpoint was redirected at a test server.
	p := NewDuckDuckGoSearch(nil).(*duckDuckGoSearch)
	p.endpoint = srv.URL
	tool := NewWebSearchTool(p, 0)
	if _, err := tool.Run(context.Background(), `{"query":"x"}`); err == nil {
		t.Fatal("expected loopback search to be refused")
	}
}

func TestWebSearchClampsResults(t *testing.T) {
	many := make([]SearchResult, 0, 20)
	for i := range 20 {
		many = append(many, SearchResult{
			Title:   fmt.Sprintf("Result %d", i),
			URL:     fmt.Sprintf("https://example.com/%d", i),
			Snippet: strings.Repeat("s", 500),
		})
	}
	tool := NewWebSearchTool(stubSearchProvider{results: many}, 0)
	out, err := tool.Run(context.Background(), `{"query":"x","max_results":50}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if strings.Contains(out, "Result 10") {
		t.Fatalf("more than %d results rendered: %q", webSearchMaxResults, out)
	}
	if !strings.Contains(out, "Result 9") {
		t.Fatalf("expected %d results: %q", webSearchMaxResults, out)
	}
	if strings.Contains(out, strings.Repeat("s", webSearchMaxSnippetLen+1)) {
		t.Fatal("snippet was not clamped")
	}
}

func TestWebSearchRequiresQuery(t *testing.T) {
	tool := NewWebSearchTool(stubSearchProvider{}, 0)
	for _, args := range []string{`{"query":""}`, `{"query":"   "}`, `{}`} {
		if _, err := tool.Run(context.Background(), args); err == nil {
			t.Fatalf("expected error for args %s", args)
		}
	}
}

func TestWebSearchNoResults(t *testing.T) {
	tool := NewWebSearchTool(stubSearchProvider{}, 0)
	out, err := tool.Run(context.Background(), `{"query":"nothing"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, "No results") {
		t.Fatalf("expected no-results message, got %q", out)
	}
}

func TestParseDuckDuckGoResults(t *testing.T) {
	got := parseDuckDuckGoResults([]byte(ddgFixture), 10)
	if len(got) != 2 {
		t.Fatalf("want 2 results, got %d: %+v", len(got), got)
	}
	if got[0].URL != "https://example.com/one" {
		t.Fatalf("uddg redirect not resolved: %q", got[0].URL)
	}
	if got[1].URL != "https://example.com/two" {
		t.Fatalf("absolute href not kept: %q", got[1].URL)
	}
}

type stubSearchProvider struct {
	results []SearchResult
	err     error
}

func (s stubSearchProvider) Search(context.Context, string, int) ([]SearchResult, error) {
	return s.results, s.err
}
