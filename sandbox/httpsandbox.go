package sandbox

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/2found/2ai/agentcore"
)

// This file is the sandboxed half of the two outbound tools (http_request,
// web_fetch). Their host half lives in http_tool.go / webfetch_tool.go and uses
// net/http with the guarded dialer; this half makes the same request from
// *inside* the sandbox, so egress is confined by the container's network
// envelope instead of the host process's.
//
// The credential problem this has to solve: the agentcore loop resolves
// {{cred:NAME}} placeholders in the argument JSON before Run is called, so by
// the time a request is built the Authorization header holds the real secret.
// Handing that to a container is only safe if the secret is not readable from
// inside it by anything other than the process that needs it — so the request
// is written as a curl config on **stdin**. It never appears in argv (visible
// to `ps` for every process in the container), never in an environment variable
// (visible in /proc/<pid>/environ), and never on the container filesystem.

const (
	// curlExitNotFound is the shell's "command not found" exit. The default
	// sandbox image is busybox-based alpine with no curl, so this is the failure
	// a deployment hits first — it must surface as an explicit, actionable error
	// rather than an empty body.
	curlExitNotFound = 127
	// sandboxHTTPTimeoutPad gives the container a little longer than curl's own
	// --max-time so the timeout is reported by curl (with a usable message)
	// rather than by the sandbox killing the process.
	sandboxHTTPTimeoutPad = 5
)

// sandboxHTTPRequest is one outbound request to make from inside the sandbox.
type sandboxHTTPRequest struct {
	Method  string
	URL     string
	Headers map[string]string
	Body    string
	// TimeoutSeconds bounds the request inside the container.
	TimeoutSeconds int
	// MaxBodyBytes truncates the response body, mirroring the host path's
	// io.LimitReader.
	MaxBodyBytes int64
	// AllowHosts confines the container's egress to these hosts (and their
	// subdomains) via the sandbox's filtering proxy. It must never be empty: an
	// empty list means "open network" to the backend, which would let a redirect
	// or a rebound DNS name reach the host's internal services.
	AllowHosts []string
	// FollowRedirects enables curl -L (web_fetch); http_request surfaces a 3xx
	// to the model instead of following it.
	FollowRedirects bool
	MaxRedirects    int
}

// execSandboxHTTP runs one request inside sb and returns the parsed response.
// The status line and headers come back in-band via curl --include, which is
// parsed here into the same shape the host path gets from net/http, so the two
// substrates render identically.
func execSandboxHTTP(ctx context.Context, sb agentcore.Sandbox, tool string, req sandboxHTTPRequest) (*http.Response, []byte, error) {
	if len(req.AllowHosts) == 0 {
		// Fail closed. Without an allowlist the container gets an open network and
		// no IP-level guard, which is strictly worse than the host path it replaced.
		return nil, nil, fmt.Errorf("%s: refusing to run in the sandbox without an egress allowlist", tool)
	}
	timeout := req.TimeoutSeconds
	if timeout <= 0 {
		timeout = 15
	}
	cfg := curlConfig(req, timeout)

	res, err := sb.Exec(ctx, agentcore.SandboxExec{
		// --config - reads the whole request (url, method, headers, body) from
		// stdin, so a resolved credential never reaches argv or env.
		Argv:  []string{"curl", "--config", "-"},
		Stdin: cfg,
		Constraints: agentcore.SandboxLimits{
			Network:        true,
			NetworkAllow:   req.AllowHosts,
			TimeoutSeconds: float64(timeout + sandboxHTTPTimeoutPad),
		},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", tool, err)
	}
	if res.Killed {
		return nil, nil, fmt.Errorf("%s: sandbox killed the request: %s", tool, res.KillReason)
	}
	if res.ExitCode == curlExitNotFound {
		return nil, nil, fmt.Errorf("%s: the sandbox image has no curl, so the request cannot be made from inside it — "+
			"configure a curl-capable sandbox image or run this tool without a sandbox", tool)
	}
	if res.ExitCode != 0 {
		msg := strings.TrimSpace(res.Stderr)
		if msg == "" {
			msg = fmt.Sprintf("curl exited %d", res.ExitCode)
		}
		return nil, nil, fmt.Errorf("%s: %s", tool, msg)
	}
	return parseCurlResponse(res.Stdout, req.MaxBodyBytes)
}

// curlConfig renders the request as a curl config file. Every value is quoted
// and escaped per curl's config syntax, so a header or body containing a quote,
// a backslash or a newline survives intact.
func curlConfig(req sandboxHTTPRequest, timeout int) string {
	var b strings.Builder
	b.WriteString("url = " + curlQuote(req.URL) + "\n")
	if m := strings.TrimSpace(req.Method); m != "" {
		b.WriteString("request = " + curlQuote(strings.ToUpper(m)) + "\n")
	}
	hasContentType := false
	for k, v := range req.Headers {
		if strings.EqualFold(strings.TrimSpace(k), "Content-Type") {
			hasContentType = true
		}
		b.WriteString("header = " + curlQuote(k+": "+v) + "\n")
	}
	if req.Body != "" {
		// data-raw, never data/data-binary: those treat a leading '@' as "read
		// this file", so a model-authored body of "@/etc/passwd" would post the
		// container's file instead of the string it wrote.
		b.WriteString("data-raw = " + curlQuote(req.Body) + "\n")
		if !hasContentType {
			// curl adds application/x-www-form-urlencoded to any request with a
			// body; net/http adds nothing. An empty value tells curl to send no
			// such header, so both substrates put the same request on the wire.
			b.WriteString(`header = "Content-Type:"` + "\n")
		}
	}
	if req.FollowRedirects {
		b.WriteString("location\n")
		if req.MaxRedirects > 0 {
			b.WriteString("max-redirs = " + strconv.Itoa(req.MaxRedirects) + "\n")
		}
	}
	// include: response headers arrive on stdout ahead of the body, which is how
	// the status and headers get back without a second channel.
	// silent + show-error: no progress meter on stdout, real errors on stderr.
	b.WriteString("include\nsilent\nshow-error\n")
	b.WriteString("max-time = " + strconv.Itoa(timeout) + "\n")
	return b.String()
}

// curlQuote renders s as a curl config double-quoted value. curl understands
// \\, \", \t, \n, \r and \v inside such a value; everything else is literal.
func curlQuote(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for i := 0; i < len(s); i++ {
		switch c := s[i]; c {
		case '\\':
			b.WriteString(`\\`)
		case '"':
			b.WriteString(`\"`)
		case '\t':
			b.WriteString(`\t`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\v':
			b.WriteString(`\v`)
		default:
			b.WriteByte(c)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// parseCurlResponse splits curl --include output into the final response's
// status/headers and its body. Informational (1xx) blocks and the header blocks
// of followed redirects are skipped, so what comes back is the response the
// model should see — the same one net/http would have handed the host path.
func parseCurlResponse(out string, maxBodyBytes int64) (*http.Response, []byte, error) {
	rest := out
	var resp *http.Response
	for {
		block, body, ok := splitHeaderBlock(rest)
		if !ok {
			if resp == nil {
				return nil, nil, fmt.Errorf("no HTTP response headers in sandbox output")
			}
			break
		}
		parsed, err := parseHeaderBlock(block)
		if err != nil {
			return nil, nil, err
		}
		resp = parsed
		rest = body
		// Only an informational (1xx) or a redirect curl followed is succeeded by
		// another header block. Anything else means `rest` is the body — and the
		// body is not allowed to be re-parsed as headers just because it happens to
		// start with "HTTP/", which a fetched log or spec file easily does.
		interim := parsed.StatusCode < 200 || (parsed.StatusCode >= 300 && parsed.StatusCode < 400)
		if !interim || !startsWithStatusLine(rest) {
			break
		}
	}
	body := []byte(rest)
	if maxBodyBytes > 0 && int64(len(body)) > maxBodyBytes {
		body = body[:maxBodyBytes]
	}
	return resp, body, nil
}

// splitHeaderBlock cuts s at the first blank line, returning the header block
// and the remainder. It tolerates both CRLF (what servers send) and LF.
func splitHeaderBlock(s string) (block, rest string, ok bool) {
	if !startsWithStatusLine(s) {
		return "", "", false
	}
	if i := strings.Index(s, "\r\n\r\n"); i >= 0 {
		return s[:i], s[i+4:], true
	}
	if i := strings.Index(s, "\n\n"); i >= 0 {
		return s[:i], s[i+2:], true
	}
	return s, "", true // headers with no body
}

func startsWithStatusLine(s string) bool { return strings.HasPrefix(s, "HTTP/") }

// parseHeaderBlock turns one raw status-line-plus-headers block into an
// http.Response. curl's HTTP/2 status line carries no reason phrase, so one is
// synthesized from the code — keeping "status: 200 OK" identical to what the
// host path prints for the same response.
func parseHeaderBlock(block string) (*http.Response, error) {
	lines := strings.Split(strings.ReplaceAll(block, "\r\n", "\n"), "\n")
	if len(lines) == 0 {
		return nil, fmt.Errorf("empty HTTP response header block")
	}
	fields := strings.SplitN(strings.TrimSpace(lines[0]), " ", 3)
	if len(fields) < 2 {
		return nil, fmt.Errorf("malformed HTTP status line %q", lines[0])
	}
	code, err := strconv.Atoi(fields[1])
	if err != nil {
		return nil, fmt.Errorf("malformed HTTP status code %q", fields[1])
	}
	reason := ""
	if len(fields) == 3 {
		reason = strings.TrimSpace(fields[2])
	}
	if reason == "" {
		reason = http.StatusText(code)
	}
	resp := &http.Response{
		StatusCode: code,
		Status:     strings.TrimSpace(strconv.Itoa(code) + " " + reason),
		Header:     http.Header{},
	}
	for _, line := range lines[1:] {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		resp.Header.Add(strings.TrimSpace(k), strings.TrimSpace(v))
	}
	return resp, nil
}
