package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
	"unicode"

	"golang.org/x/net/html"

	"github.com/2found/2ai/agentcore"
)

// ToolWebFetch is the stable name of the open web-fetch tool. Like http_request
// it must be permitted by policy before the model sees it.
const ToolWebFetch = "web_fetch"

const (
	webFetchTimeout      = 20 * time.Second
	webFetchMaxBodyBytes = 512 * 1024
	webFetchMaxRedirects = 5
)

// WebFetchTool fetches an arbitrary public URL and returns its readable text
// (Claude Code's WebFetch). It is the open-egress counterpart to http_request:
// where http_request is host-allowlisted for talking to specific APIs, web_fetch
// is meant for reading the open web, so it has no host allowlist. SSRF is still
// closed off at the IP layer — the same guarded dialer as http_request re-checks
// every resolved address (including each redirect hop) and refuses
// loopback / private / link-local / metadata, so "no allowlist" does not mean
// "can reach internal services". HTML is reduced to text to keep results small.
//
// With a sandbox injected the fetch is made from inside the container instead.
// Because the tool has no host allowlist of its own, the container's egress is
// pinned per call to the requested URL's host (and its subdomains) — an empty
// egress allowlist would hand the container an open network with no IP guard,
// which is exactly the SSRF surface this tool exists to close. The visible
// consequence is that a redirect off that host is refused by the egress proxy
// rather than followed.
type WebFetchTool struct {
	sb           agentcore.Sandbox
	client       *http.Client
	maxBodyBytes int64
	// sink writes a fetched document into the agent's workspace when the call
	// asks for save_as. Zero value = no workspace, and save_as is refused.
	sink responseSink
}

// NewWebFetchTool builds the web_fetch tool with the SSRF-guarded dialer
// installed. sb is optional: nil fetches from this host process (the default),
// non-nil fetches from inside the sandbox. It follows a bounded number of
// redirects because every hop is re-validated at connect time — by the dialer on
// the host path, by the egress proxy on the sandbox path — unlike http_request
// which cannot (its allowlist can't re-check a redirected host).
//
// ws is optional and enables save_as: with a workspace the fetched document can
// be written where read_file, grep, and run_shell can reach it, which is what a
// page too long to read in one context needs.
func NewWebFetchTool(sb agentcore.Sandbox, ws *Workspace, options ...WebFetchOption) *WebFetchTool {
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	t := &WebFetchTool{
		sb:           sb,
		maxBodyBytes: webFetchMaxBodyBytes,
		client: &http.Client{
			Timeout: webFetchTimeout,
			CheckRedirect: func(_ *http.Request, via []*http.Request) error {
				if len(via) >= webFetchMaxRedirects {
					return fmt.Errorf("stopped after %d redirects", webFetchMaxRedirects)
				}
				return nil
			},
			Transport: &http.Transport{
				DialContext:           guardedDialFunc(dialer, blockedIP),
				TLSHandshakeTimeout:   10 * time.Second,
				ResponseHeaderTimeout: webFetchTimeout,
				MaxIdleConns:          10,
			},
		},
	}
	if ws != nil {
		t.sink = responseSink{fs: newWorkspaceFS(sb, ws)}
	}
	for _, option := range options {
		option(t)
	}
	return t
}

type WebFetchOption func(*WebFetchTool)

// WithWebFetchTimeout sets the host fetch deadline; zero leaves cancellation to the caller.
func WithWebFetchTimeout(timeout time.Duration) WebFetchOption {
	return func(t *WebFetchTool) {
		if timeout < 0 {
			return
		}
		t.client.Timeout = timeout
		transport := t.client.Transport.(*http.Transport)
		transport.DialContext = guardedDialFunc(&net.Dialer{Timeout: timeout}, blockedIP)
		transport.TLSHandshakeTimeout = timeout
		transport.ResponseHeaderTimeout = timeout
	}
}

func (t *WebFetchTool) Name() string   { return ToolWebFetch }
func (t *WebFetchTool) Parallel() bool { return true }

func (t *WebFetchTool) Schema() agentcore.ToolSchema {
	props := map[string]any{
		"url": map[string]any{"type": "string", "description": "Absolute https:// URL to fetch."},
	}
	// Advertised only when there is a workspace behind it — a run whose workspace
	// could not be created still gets the tool, and offering a save it must then
	// refuse spends a real fetch to learn that.
	if t.sink.available() {
		props["save_as"] = saveAsParam()
	}
	return agentcore.ToolSchema{
		Name:   ToolWebFetch,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Fetch a public web page over HTTPS and return its readable text content. " +
			"HTML is stripped to text; non-HTML text is returned as-is. Use this to read documentation, " +
			"articles, or API docs. Internal/loopback/private addresses are refused. For authenticated " +
			"calls to a specific API, use http_request instead.",
		Parameters: map[string]any{
			"type":       "object",
			"properties": props,
			"required":   []string{"url"},
		},
	}
}

func (t *WebFetchTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		URL    string `json:"url"`
		SaveAs string `json:"save_as"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("web_fetch: invalid arguments: %w", err)
	}
	u, err := parseAbsoluteURL(in.URL)
	if err != nil {
		return "", fmt.Errorf("web_fetch: %w", err)
	}
	if u.Scheme != "https" && u.Scheme != "http" {
		return "", fmt.Errorf("web_fetch: unsupported scheme %q", u.Scheme)
	}

	headers := map[string]string{
		"User-Agent": "agentray-web-fetch/1.0",
		"Accept":     "text/html,application/xhtml+xml,text/plain,*/*",
	}

	var (
		body []byte
		ct   string
	)
	if t.sb != nil {
		resp, b, err := execSandboxHTTP(ctx, t.sb, ToolWebFetch, sandboxHTTPRequest{
			Method:  http.MethodGet,
			URL:     u.String(),
			Headers: headers,
			// No host allowlist exists for this tool, so the egress grant is scoped
			// to the one host the model asked for rather than left open.
			AllowHosts:      []string{u.Hostname()},
			FollowRedirects: true,
			MaxRedirects:    webFetchMaxRedirects,
			TimeoutSeconds:  int(webFetchTimeout / time.Second),
			MaxBodyBytes:    t.maxBodyBytes,
		})
		if err != nil {
			return "", err
		}
		body, ct = b, resp.Header.Get("Content-Type")
		return t.respond(ctx, in.SaveAs, u.String(), resp.Status, ct, body)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return "", fmt.Errorf("web_fetch: %w", err)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("web_fetch: %w", err)
	}
	defer resp.Body.Close()

	body, _ = io.ReadAll(io.LimitReader(resp.Body, t.maxBodyBytes))
	ct = resp.Header.Get("Content-Type")
	return t.respond(ctx, in.SaveAs, u.String(), resp.Status, ct, body)
}

// respond returns the readable document, or saves it and returns a receipt.
//
// What gets saved is the RAW body, not the HTML-stripped text: the stripping
// exists to fit a page into a context window, and a file the agent is about to
// parse should be the document the server sent, not a lossy rendering of it.
func (t *WebFetchTool) respond(ctx context.Context, saveAs, url, status, ct string, body []byte) (string, error) {
	// Equal to the limit means ReadAll stopped at the cap, so the document very
	// likely continues past what we hold.
	truncated := int64(len(body)) >= t.maxBodyBytes
	return t.sink.respond(ctx, ToolWebFetch, saveAs, body, truncated, func(b []byte) string {
		return formatWebFetch(url, status, ct, b)
	})
}

// formatWebFetch renders one fetched document for the model, identically for
// both substrates.
func formatWebFetch(url, status, ct string, body []byte) string {
	var content string
	if isHTML(ct, body) {
		content = htmlToText(body)
	} else {
		content = string(body)
	}
	content = strings.TrimSpace(content)

	var b strings.Builder
	fmt.Fprintf(&b, "url: %s\nstatus: %s\n", url, status)
	if ct != "" {
		fmt.Fprintf(&b, "content-type: %s\n", ct)
	}
	if content != "" {
		fmt.Fprintf(&b, "content:\n%s", content)
	}
	return strings.TrimRight(b.String(), "\n")
}

func isHTML(contentType string, body []byte) bool {
	if strings.Contains(strings.ToLower(contentType), "html") {
		return true
	}
	head := strings.ToLower(string(body[:min(len(body), 512)]))
	return strings.Contains(head, "<html") || strings.Contains(head, "<!doctype html")
}

// htmlToText extracts visible text from an HTML document, dropping script/style
// content and collapsing whitespace so the result is compact for the model.
func htmlToText(body []byte) string {
	doc, err := html.Parse(strings.NewReader(string(body)))
	if err != nil {
		return string(body)
	}
	var b strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.Data {
			case "script", "style", "noscript", "head", "svg":
				return
			}
		}
		if n.Type == html.TextNode {
			text := strings.TrimSpace(n.Data)
			if text != "" {
				b.WriteString(text)
				b.WriteString(" ")
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
		// Break lines after common block elements so output stays readable.
		if n.Type == html.ElementNode {
			switch n.Data {
			case "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article":
				b.WriteString("\n")
			}
		}
	}
	walk(doc)
	return collapseBlankLines(b.String())
}

func collapseBlankLines(s string) string {
	var out []string
	blank := 0
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimRightFunc(line, unicode.IsSpace)
		if strings.TrimSpace(line) == "" {
			blank++
			if blank > 1 {
				continue
			}
			out = append(out, "")
			continue
		}
		blank = 0
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}
