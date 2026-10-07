package ai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

// Timeouts for the chat providers. There are two knobs, they do different jobs,
// and which of them applies depends on whether the request is STREAMED.
//
// Go's http.Client.Timeout is absolute: it covers dialing, the response headers,
// AND the full body read. For a chat provider the body IS the model generating —
// a turn that produces a complete HTML page can legitimately spend minutes
// there. Sizing that cap like a handshake (the old 120s) does not protect
// against a dead gateway; it kills the slowest *healthy* turns, and because the
// kill is deterministic the retry ladder then reproduces it on every attempt.
//
// ResponseHeaderTimeout looks like the safe replacement — time to first response
// header is a real liveness signal — but only on an SSE request. There the
// gateway writes its 200 immediately and the tokens follow, so headers precede
// generation. On a NON-streamed completion the response is a single buffered
// JSON document: headers do not leave the gateway until generation has finished,
// so a "header" deadline is a generation deadline wearing a disguise. Bench #4
// proved it — the first turn generated 9.5k tokens, the header deadline fired at
// exactly 60.3s, and the (successful) retry then spent 135s regenerating the
// same answer: 60s of dead wall clock and a doubled bill for a healthy call.
//
// So: the absolute cap applies to both paths and is sized for the slowest
// legitimate turn; the header deadline applies to the streamed path ONLY.
const (
	// DefaultChatTimeout bounds one chat exchange end to end, body included.
	DefaultChatTimeout = 10 * time.Minute
	// DefaultResponseHeaderTimeout bounds the wait for response headers on a
	// STREAMED request — the "is this gateway alive" check the absolute cap was
	// mistakenly doing. It is meaningless on a buffered completion.
	DefaultResponseHeaderTimeout = 60 * time.Second
)

// setOptionalBearerAuth adds a Bearer credential only when one exists. Several
// OpenAI-compatible local engines accept unauthenticated requests; sending an
// empty `Authorization: Bearer ` header can make an otherwise valid local call
// fail at a proxy or middleware layer.
func setOptionalBearerAuth(req *http.Request, apiKey string) {
	if key := strings.TrimSpace(apiKey); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
}

// NewChatHTTPClient builds the HTTP client for NON-streamed chat completions: a
// generous absolute deadline and no header deadline, because the headers of a
// buffered completion arrive only once the model has finished. A zero timeout
// uses DefaultChatTimeout.
//
// Exported so a caller that knows its own latency envelope can size the cap
// itself and assign it to the provider's HTTP field, rather than inheriting a
// default chosen for the slowest case.
func NewChatHTTPClient(timeout time.Duration) *http.Client {
	if timeout <= 0 {
		timeout = DefaultChatTimeout
	}
	// Clone the stock transport so connection pooling, proxy support and HTTP/2
	// stay at their defaults.
	return &http.Client{Timeout: timeout, Transport: http.DefaultTransport.(*http.Transport).Clone()}
}

// NewStreamHTTPClient builds the HTTP client for SSE chat requests: the same
// absolute cap plus the header deadline, which is honest here — an SSE gateway
// flushes its headers before the first token, so a silent wait past
// DefaultResponseHeaderTimeout means the endpoint is not answering at all.
func NewStreamHTTPClient(timeout time.Duration) *http.Client {
	c := NewChatHTTPClient(timeout)
	c.Transport.(*http.Transport).ResponseHeaderTimeout = DefaultResponseHeaderTimeout
	return c
}

// describeReadErr names which deadline actually killed a body read. Go reports
// both as the same string — "context deadline exceeded (Client.Timeout or
// context cancellation while reading body)" — which is useless when you are
// trying to work out whether the caller gave up or we hung up on a healthy
// stream. The two have opposite fixes, so say which one it was, and for our own
// deadline say what it was set to.
func describeReadErr(ctx context.Context, client *http.Client, readErr error) string {
	if readErr == nil {
		return ""
	}
	// The caller's context wins: if it is already done, its deadline is what the
	// transport observed, whatever the client cap happens to be.
	if cerr := ctx.Err(); cerr != nil {
		return fmt.Sprintf("caller context ended mid-body (%v)", cerr)
	}
	if isTimeout(readErr) && client != nil && client.Timeout > 0 {
		return fmt.Sprintf("client timeout of %s elapsed mid-body; raise it for long "+
			"generations (ai.NewChatHTTPClient) — the model was still streaming", client.Timeout)
	}
	return readErr.Error()
}

// isTimeout reports whether err is a deadline expiry rather than a severed
// connection, checking the net.Error contract and os.ErrDeadlineExceeded, which
// is what Client.Timeout surfaces on a body read.
func isTimeout(err error) bool {
	if errors.Is(err, os.ErrDeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var ne interface{ Timeout() bool }
	return errors.As(err, &ne) && ne.Timeout()
}
