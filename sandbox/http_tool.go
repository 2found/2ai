package sandbox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/2found/2ai/agentcore"
)

// ToolHTTPRequest is the stable tool name. A consumer's Policy must permit it
// before the model is shown the tool.
//
// It is the legitimate-egress counterpart to the container tools in this
// package: where run_shell runs untrusted code with --network none, this makes
// *controlled* outbound calls to an operator-approved host allowlist. Where the
// call is made from depends on the substrate it was built with — see HTTPTool.
const ToolHTTPRequest = "http_request"

const (
	httpDefaultTimeout      = 15 * time.Second
	httpDefaultMaxBodyBytes = 256 * 1024
)

// HTTPTool makes guarded outbound HTTP requests. It is the worked consumer of
// the credential vault (governance F7): the place a {{cred:NAME}} secret
// actually gets used.
//
// The tool is deliberately dumb about credentials: the agentcore loop resolves
// {{cred:NAME}} placeholders in the argument JSON at the trust boundary before
// Run is ever called, so an Authorization header the model wrote as "Bearer
// {{cred:API_KEY}}" arrives here already resolved to the real value — the model
// never saw the literal, and this tool never needs the vault.
//
// Where the request is made from depends on the substrate. With no sandbox (the
// default) it goes out from the host process over net/http. With a sandbox it
// goes out from inside the container, so a compromised request cannot reach
// anything the container's network envelope does not permit. The resolved
// secret then has to cross that boundary, and it does so on the container's
// stdin as a curl config — never in argv (readable via `ps` by every process in
// the container) and never in an environment variable. See httpsandbox.go.
//
// What it is careful about either way is SSRF. An agent that can make arbitrary
// outbound requests can reach cloud metadata (169.254.169.254), internal
// services, and localhost. Defenses, default-deny:
//   - scheme must be https (http is opt-in)
//   - the URL host must be in the configured allowlist
//   - a guarded dialer re-checks the resolved IP at connect time and refuses
//     loopback / private / link-local / unspecified addresses, which also closes
//     the DNS-rebinding TOCTOU gap (allowlisted name re-pointed at a blocked IP).
//     On the sandbox path the same check runs in the egress proxy the container's
//     traffic is confined to, so the backstop survives the move.
//   - redirects are not followed (a 3xx is surfaced to the model as-is)
//
// Safe for concurrent use: the allowlist is read-only after construction and
// http.Client is concurrency-safe.
type HTTPTool struct {
	sb             agentcore.Sandbox
	client         *http.Client
	allowHosts     map[string]struct{}
	allowPlainHTTP bool
	maxBodyBytes   int64
	// ipBlocked decides whether a resolved IP is off-limits. Defaults to
	// blockedIP (refuses loopback/private/link-local); a test seam can relax it
	// to exercise the happy path against an httptest server on 127.0.0.1. Read at
	// dial time, so an option may override it after the client is built.
	ipBlocked func(net.IP) (bool, string)
	// sink writes a response body into the agent's workspace when the call asks
	// for save_as. Zero value = no workspace, and save_as is refused.
	sink responseSink
}

// HTTPOption configures an HTTPTool. It is distinct from Option, which
// configures the DockerSandbox backend.
type HTTPOption func(*HTTPTool)

// WithHTTPAllowHosts sets the exact-match host allowlist (case-insensitive, port
// stripped). A request to any other host is refused.
func WithHTTPAllowHosts(hosts []string) HTTPOption {
	return func(t *HTTPTool) {
		for _, h := range hosts {
			h = strings.ToLower(strings.TrimSpace(h))
			if h != "" {
				t.allowHosts[h] = struct{}{}
			}
		}
	}
}

// WithHTTPAllowPlain permits plain http:// URLs (default: https only).
func WithHTTPAllowPlain(allow bool) HTTPOption {
	return func(t *HTTPTool) { t.allowPlainHTTP = allow }
}

// WithHTTPWorkspace gives the tool the run's workspace, enabling save_as.
// Without it the tool still works — it just cannot hand a body to the file and
// shell tools, which is the difference between fetching data and being able to
// use it.
func WithHTTPWorkspace(ws *Workspace) HTTPOption {
	return func(t *HTTPTool) {
		if ws != nil {
			t.sink = responseSink{fs: newWorkspaceFS(t.sb, ws)}
		}
	}
}

// WithHTTPTimeout overrides the per-request timeout. Zero disables it while
// preserving caller cancellation; negative values leave the default intact.
func WithHTTPTimeout(d time.Duration) HTTPOption {
	return func(t *HTTPTool) {
		if d >= 0 {
			t.client.Timeout = d
			if transport, ok := t.client.Transport.(*http.Transport); ok {
				transport.DialContext = t.guardedDial(&net.Dialer{Timeout: d})
				transport.TLSHandshakeTimeout = d
				transport.ResponseHeaderTimeout = d
			}
		}
	}
}

// NewHTTPRequestTool builds the http_request tool. sb is optional: nil makes
// the request from this host process (the default), non-nil makes it from
// inside the sandbox with egress confined to the same allowlist. The guarded
// dialer is installed either way — for the host path directly on the client, for
// the sandbox path in the egress proxy — so every request is IP-checked at
// connect.
func NewHTTPRequestTool(sb agentcore.Sandbox, opts ...HTTPOption) *HTTPTool {
	t := &HTTPTool{
		sb:           sb,
		allowHosts:   make(map[string]struct{}),
		maxBodyBytes: httpDefaultMaxBodyBytes,
		ipBlocked:    blockedIP,
	}
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	t.client = &http.Client{
		Timeout: httpDefaultTimeout,
		// Do not follow redirects: a 3xx to an allowlisted host could bounce to a
		// blocked one. Surface the redirect response to the model instead.
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		Transport: &http.Transport{
			DialContext:           t.guardedDial(dialer),
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: httpDefaultTimeout,
			MaxIdleConns:          10,
		},
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// AllowHosts returns the configured allowlist (for startup logging / tests).
func (t *HTTPTool) AllowHosts() []string {
	out := make([]string, 0, len(t.allowHosts))
	for h := range t.allowHosts {
		out = append(out, h)
	}
	sort.Strings(out)
	return out
}

func (t *HTTPTool) Name() string { return ToolHTTPRequest }

func (t *HTTPTool) Schema() agentcore.ToolSchema {
	props := map[string]any{
		"method": map[string]any{
			"type":        "string",
			"description": "HTTP method (GET, POST, PUT, PATCH, DELETE). Defaults to GET.",
		},
		"url": map[string]any{
			"type":        "string",
			"description": "Absolute https:// URL whose host is on the allowlist.",
		},
		"headers": map[string]any{
			"type":        "object",
			"description": "Optional request headers. Secret values may use {{cred:NAME}}.",
		},
		"body": map[string]any{
			"type":        "string",
			"description": "Optional request body (sent verbatim).",
		},
	}
	// Only offered when there is a workspace to save into. The host-global
	// instance is built once at startup, long before any run has one, and an
	// advertised save_as it cannot honour would cost a real request per attempt.
	if t.sink.available() {
		props["save_as"] = saveAsParam()
	}
	return agentcore.ToolSchema{
		Name:   ToolHTTPRequest,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Make an outbound HTTP request to an approved host. Use a " +
			"{{cred:NAME}} placeholder for any secret (e.g. an Authorization " +
			"header value); it is resolved securely and never exposed. Only " +
			"allowlisted hosts over HTTPS are reachable. Returns the status, " +
			"response headers, and body.",
		Parameters: map[string]any{
			"type":       "object",
			"properties": props,
			"required":   []string{"url"},
		},
	}
}

type httpRequestArgs struct {
	Method  string            `json:"method"`
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`
	Body    string            `json:"body"`
	SaveAs  string            `json:"save_as"`
}

// Run executes the guarded request. args has already had {{cred:NAME}} resolved
// by the loop, so any secret in Headers is the real value here.
func (t *HTTPTool) Run(ctx context.Context, args string) (string, error) {
	var in httpRequestArgs
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("http_request: invalid arguments: %w", err)
	}

	method := strings.ToUpper(strings.TrimSpace(in.Method))
	if method == "" {
		method = http.MethodGet
	}
	if !allowedHTTPMethod(method) {
		return "", fmt.Errorf("http_request: method %q not allowed", method)
	}
	if err := t.validateURL(in.URL); err != nil {
		return "", err
	}

	if t.sb != nil {
		resp, body, err := execSandboxHTTP(ctx, t.sb, ToolHTTPRequest, sandboxHTTPRequest{
			Method:  method,
			URL:     in.URL,
			Headers: in.Headers,
			Body:    in.Body,
			// The egress allowlist is the tool's own host allowlist, so the
			// container can reach exactly the hosts the operator approved and
			// nothing else — including on a redirect, which is not followed here.
			AllowHosts:     t.AllowHosts(),
			TimeoutSeconds: int(t.client.Timeout / time.Second),
			MaxBodyBytes:   t.maxBodyBytes,
		})
		if err != nil {
			return "", err
		}
		return t.respond(ctx, in.SaveAs, resp, body)
	}

	var bodyReader io.Reader
	if in.Body != "" {
		bodyReader = strings.NewReader(in.Body)
	}
	req, err := http.NewRequestWithContext(ctx, method, in.URL, bodyReader)
	if err != nil {
		return "", fmt.Errorf("http_request: %w", err)
	}
	for k, v := range in.Headers {
		req.Header.Set(k, v)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("http_request: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, t.maxBodyBytes))
	return t.respond(ctx, in.SaveAs, resp, body)
}

// respond returns the body inline, or writes it to the workspace and returns the
// status plus a receipt. The status and content-type are reported either way —
// a save is not a reason to hide that the server answered 404 and the "file" is
// an error page.
func (t *HTTPTool) respond(ctx context.Context, saveAs string, resp *http.Response, body []byte) (string, error) {
	// Equal to the limit means ReadAll stopped at the cap, so the document very
	// likely continues past what we hold.
	truncated := int64(len(body)) >= t.maxBodyBytes
	return t.sink.respond(ctx, ToolHTTPRequest, saveAs, body, truncated, func(b []byte) string {
		return formatHTTPResponse(resp, b)
	})
}

// validateURL enforces scheme + host allowlist. The IP-level guard is the
// dialer's job (it sees the resolved address at connect time).
func (t *HTTPTool) validateURL(raw string) error {
	u, err := parseAbsoluteURL(raw)
	if err != nil {
		return fmt.Errorf("http_request: %w", err)
	}
	switch u.Scheme {
	case "https":
	case "http":
		if !t.allowPlainHTTP {
			return errors.New("http_request: only https:// URLs are allowed")
		}
	default:
		return fmt.Errorf("http_request: unsupported scheme %q", u.Scheme)
	}
	host := strings.ToLower(u.Hostname())
	if _, ok := t.allowHosts[host]; !ok {
		return fmt.Errorf("http_request: host %q is not on the allowlist", host)
	}
	return nil
}

func allowedHTTPMethod(m string) bool {
	switch m {
	case http.MethodGet, http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete, http.MethodHead:
		return true
	default:
		return false
	}
}

func formatHTTPResponse(resp *http.Response, body []byte) string {
	var b strings.Builder
	fmt.Fprintf(&b, "status: %s\n", resp.Status)
	if loc := resp.Header.Get("Location"); loc != "" {
		fmt.Fprintf(&b, "location: %s\n", loc)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		fmt.Fprintf(&b, "content-type: %s\n", ct)
	}
	if len(body) > 0 {
		fmt.Fprintf(&b, "body:\n%s", string(body))
	}
	return strings.TrimRight(b.String(), "\n")
}
