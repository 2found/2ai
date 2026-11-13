package sandbox

import (
	"crypto/tls"
	"crypto/x509"
	"net"
	"net/http"
	"time"
)

// AllowAllIPsForTest relaxes the web_fetch IP guard so a test can hit an
// httptest server on 127.0.0.1 (normally refused as loopback). Test-only.
func (t *WebFetchTool) AllowAllIPsForTest() {
	tr, ok := t.client.Transport.(*http.Transport)
	if !ok {
		return
	}
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	tr.DialContext = guardedDialFunc(dialer, func(net.IP) (bool, string) { return false, "" })
}

// AllowAllIPsForTest relaxes the web_search provider's IP guard so a test can
// hit an httptest server on 127.0.0.1 (normally refused as loopback). It only
// applies when the provider is the built-in DuckDuckGo one. Test-only.
func (t *WebSearchTool) AllowAllIPsForTest() {
	p, ok := t.provider.(*duckDuckGoSearch)
	if !ok {
		return
	}
	tr, ok := p.client.Transport.(*http.Transport)
	if !ok {
		return
	}
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	tr.DialContext = guardedDialFunc(dialer, func(net.IP) (bool, string) { return false, "" })
}

// AllowAllIPsForTest relaxes the IP guard so tests can hit an httptest server on
// 127.0.0.1 (normally refused as loopback). Test-only: compiled only under _test.
func (t *HTTPTool) AllowAllIPsForTest() {
	t.ipBlocked = func(net.IP) (bool, string) { return false, "" }
}

// TrustCertsForTest makes the guarded client trust the given roots, so a test can
// reach an httptest.NewTLSServer (self-signed) over real https — the scheme a
// live model expects. The guarded dialer and allowlist are untouched. Test-only:
// compiled only under _test.
func (t *HTTPTool) TrustCertsForTest(roots *x509.CertPool) {
	tr, ok := t.client.Transport.(*http.Transport)
	if !ok {
		return
	}
	if tr.TLSClientConfig == nil {
		tr.TLSClientConfig = &tls.Config{}
	}
	tr.TLSClientConfig.RootCAs = roots
}
