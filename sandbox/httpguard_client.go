package sandbox

import (
	"net"
	"net/http"
	"time"
)

// NewGuardedClient returns an http.Client that shares this package's SSRF
// backstop — every dial resolves the host and refuses the connection if any
// resolved IP is loopback/private/link-local (the cloud-metadata endpoint) —
// but WITHOUT a host allowlist. It is for outbound delivery to user-configured
// destinations (alert webhooks, Slack incoming-webhooks) where the host set is
// open-ended but the private-network SSRF guard must still hold. Redirects are
// not followed, closing the "3xx bounces to a blocked host" gap. Zero uses the
// default timeout; a negative timeout disables deadlines but keeps cancellation.
func NewGuardedClient(timeout time.Duration) *http.Client {
	connectTimeout := 10 * time.Second
	if timeout == 0 {
		timeout = httpDefaultTimeout
	}
	if timeout < 0 {
		timeout, connectTimeout = 0, 0
	}
	dialer := &net.Dialer{Timeout: connectTimeout}
	return &http.Client{
		Timeout:       timeout,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		Transport: &http.Transport{
			DialContext:           guardedDialFunc(dialer, blockedIP),
			TLSHandshakeTimeout:   connectTimeout,
			ResponseHeaderTimeout: timeout,
			MaxIdleConns:          10,
		},
	}
}
