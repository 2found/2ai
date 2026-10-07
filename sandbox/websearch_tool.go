package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"

	"golang.org/x/net/html"

	"github.com/2found/2ai/agentcore"
)

// ToolWebSearch is the stable name of the web-search tool. Like web_fetch it
// must be permitted by policy before the model sees it.
const ToolWebSearch = "web_search"

const (
	webSearchTimeout      = 15 * time.Second
	webSearchMaxBodyBytes = 256 * 1024
	// webSearchDefaultMaxResults / webSearchMaxResults bound the result list so
	// the rendered answer stays far under the loop's tool-result cap and spill
	// remains a fallback, not the common path.
	webSearchDefaultMaxResults = 5
	webSearchMaxResults        = 10
	webSearchMaxTitleLen       = 200
	webSearchMaxSnippetLen     = 300
)

// SearchResult is one ranked hit: title, URL, and a short snippet.
type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

// SearchProvider is the pluggable backend behind web_search. The provider is
// configuration, not code: the registry picks an implementation from the
// agent's tool config and hands it here, so adding a provider never touches
// the tool. Implementations must make their own egress safe — the built-in
// ones dial through NewGuardedClient, which re-checks every resolved IP.
type SearchProvider interface {
	// Search runs one query and returns ranked results, best first, at most
	// maxResults of them.
	Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error)
}

// WebSearchTool answers a query with ranked web results (title/url/snippet).
// It is the discovery counterpart to web_fetch: search finds the pages,
// web_fetch reads one. There is no host allowlist — the result set is whatever
// the provider returns — but the provider's own egress rides the same guarded
// dialer as every other host-side outbound tool, so loopback / private /
// link-local / metadata addresses are refused at connect time.
//
// Host substrate only, for the same reason as http_request: the default
// sandbox image has no HTTP client to run the request through.
type WebSearchTool struct {
	provider   SearchProvider
	maxResults int
}

// NewWebSearchTool builds the web_search tool over the given provider.
// maxResults is the per-call default and ceiling source: 0 uses the package
// default, anything above the package maximum is clamped.
func NewWebSearchTool(p SearchProvider, maxResults int) *WebSearchTool {
	if maxResults <= 0 {
		maxResults = webSearchDefaultMaxResults
	}
	if maxResults > webSearchMaxResults {
		maxResults = webSearchMaxResults
	}
	return &WebSearchTool{provider: p, maxResults: maxResults}
}

func (t *WebSearchTool) Name() string   { return ToolWebSearch }
func (t *WebSearchTool) Parallel() bool { return true }

func (t *WebSearchTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolWebSearch,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Search the public web and return ranked results as title / URL / snippet lines. " +
			"Use this to find pages, then web_fetch to read one. Internal/loopback/private addresses " +
			"are refused by the egress guard.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string", "description": "The search query."},
				"max_results": map[string]any{
					"type":        "integer",
					"description": fmt.Sprintf("Maximum results to return (default %d, at most %d).", webSearchDefaultMaxResults, webSearchMaxResults),
				},
			},
			"required": []string{"query"},
		},
	}
}

func (t *WebSearchTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Query      string `json:"query"`
		MaxResults int    `json:"max_results"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("web_search: invalid arguments: %w", err)
	}
	query := strings.TrimSpace(in.Query)
	if query == "" {
		return "", fmt.Errorf("web_search: query is required")
	}
	max := in.MaxResults
	if max <= 0 {
		max = t.maxResults
	}
	if max > webSearchMaxResults {
		max = webSearchMaxResults
	}

	results, err := t.provider.Search(ctx, query, max)
	if err != nil {
		return "", fmt.Errorf("web_search: %w", err)
	}
	if len(results) == 0 {
		return fmt.Sprintf("No results for %q.", query), nil
	}
	if len(results) > max {
		results = results[:max]
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Results for %q:\n", query)
	for i, r := range results {
		fmt.Fprintf(&b, "%d. %s\n   %s\n", i+1, clampRunes(strings.TrimSpace(r.Title), webSearchMaxTitleLen), r.URL)
		if s := clampRunes(strings.TrimSpace(r.Snippet), webSearchMaxSnippetLen); s != "" {
			fmt.Fprintf(&b, "   %s\n", s)
		}
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

// clampRunes trims s to at most n runes, appending an ellipsis when it cuts.
func clampRunes(s string, n int) string {
	if n <= 0 || utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return string(r[:n]) + "…"
}

// --- DuckDuckGo HTML provider ------------------------------------------------

// duckDuckGoEndpoint is the keyless HTML endpoint. It is code, not config: the
// scheme is fixed https so a config value can never downgrade egress to plain
// HTTP or point the provider at an internal host.
const duckDuckGoEndpoint = "https://html.duckduckgo.com/html/"

// duckDuckGoSearch is the keyless provider: it scrapes DuckDuckGo's HTML
// results page. The markup is a third party's and can change without notice,
// which is exactly why it sits behind SearchProvider — a swap is one file.
type duckDuckGoSearch struct {
	client *http.Client
	// endpoint is the request URL; a field (not the const) so tests can point
	// the provider at an httptest server.
	endpoint string
}

// NewDuckDuckGoSearch builds the DuckDuckGo provider. client is optional: nil
// installs the shared SSRF-guarded client (resolved-IP re-check, no redirects
// followed), which is the only client a production build should ever use.
func NewDuckDuckGoSearch(client *http.Client) SearchProvider {
	if client == nil {
		client = NewGuardedClient(webSearchTimeout)
	}
	return &duckDuckGoSearch{client: client, endpoint: duckDuckGoEndpoint}
}

func (p *duckDuckGoSearch) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	u, err := url.Parse(p.endpoint)
	if err != nil {
		return nil, fmt.Errorf("provider endpoint: %w", err)
	}
	q := u.Query()
	q.Set("q", query)
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "agentray-web-search/1.0")
	req.Header.Set("Accept", "text/html,application/xhtml+xml")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("provider returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, webSearchMaxBodyBytes))
	if err != nil {
		return nil, err
	}
	return parseDuckDuckGoResults(body, maxResults), nil
}

// parseDuckDuckGoResults extracts one SearchResult per result container. DDG's
// HTML marks each hit with a "result" class token; inside it the title anchor
// carries "result__a" and the snippet "result__snippet". The anchor href is a
// redirect shim (//duckduckgo.com/l/?uddg=<real url>) — the real target is the
// uddg parameter; a href that is already absolute is used as-is.
func parseDuckDuckGoResults(body []byte, maxResults int) []SearchResult {
	doc, err := html.Parse(strings.NewReader(string(body)))
	if err != nil {
		return nil
	}
	var out []SearchResult
	var walk func(n *html.Node)
	walk = func(n *html.Node) {
		if len(out) >= maxResults {
			return
		}
		if n.Type == html.ElementNode && hasClassToken(n, "result") {
			if r, err := extractDDGResult(n); err == nil {
				out = append(out, r)
			}
			return // result containers do not nest
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)
	return out
}

// extractDDGResult pulls the title anchor and snippet out of one result
// container's subtree.
func extractDDGResult(container *html.Node) (SearchResult, error) {
	var r SearchResult
	var find func(n *html.Node)
	find = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch {
			case r.Title == "" && hasClassToken(n, "result__a"):
				r.Title = nodeText(n)
				r.URL = ddgResultURL(attr(n, "href"))
			case r.Snippet == "" && hasClassToken(n, "result__snippet"):
				r.Snippet = nodeText(n)
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			find(c)
		}
	}
	find(container)
	if r.URL == "" {
		return SearchResult{}, fmt.Errorf("no result url")
	}
	return r, nil
}

// ddgResultURL resolves a result href to the real target: the uddg parameter
// of DDG's redirect shim, or the href itself when already absolute.
func ddgResultURL(href string) string {
	if href == "" {
		return ""
	}
	if strings.HasPrefix(href, "//") {
		href = "https:" + href
	}
	u, err := url.Parse(href)
	if err != nil {
		return ""
	}
	if uddg := u.Query().Get("uddg"); uddg != "" {
		return uddg
	}
	if u.IsAbs() && (u.Scheme == "https" || u.Scheme == "http") {
		return u.String()
	}
	return ""
}

// hasClassToken reports whether n's class attribute carries the exact token —
// token equality, not substring, so "result" does not match "result__a".
func hasClassToken(n *html.Node, token string) bool {
	for _, c := range strings.Fields(attr(n, "class")) {
		if c == token {
			return true
		}
	}
	return false
}

func attr(n *html.Node, key string) string {
	for _, a := range n.Attr {
		if a.Key == key {
			return a.Val
		}
	}
	return ""
}

// nodeText flattens an element's text content, collapsing whitespace runs.
func nodeText(n *html.Node) string {
	var b strings.Builder
	var walk func(*html.Node)
	walk = func(x *html.Node) {
		if x.Type == html.TextNode {
			b.WriteString(x.Data)
			b.WriteByte(' ')
		}
		for c := x.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return strings.Join(strings.Fields(b.String()), " ")
}
