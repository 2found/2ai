package sandbox

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// Egress allowlist (#5b). When a session is granted network AND a non-empty
// NetworkAllow list, its container must reach only the listed hosts. The
// enforcement has two halves:
//
//  1. This host-side filtering forward-proxy, which hard-denies any host not on
//     the allowlist for both CONNECT (HTTPS) and plain HTTP — the security core,
//     unit-tested independently of Docker.
//  2. docker.go routes the container's traffic through this proxy (HTTP(S)_PROXY
//     env) and, on hosts where it is available, confines the container to an
//     internal network whose only reachable peer is the proxy, so a client that
//     ignores the proxy env still cannot reach the open internet.
//
// The allowlist match is suffix-based on the DNS name: an entry "pypi.org"
// matches "pypi.org" and "sub.pypi.org" but never "notpypi.org" (the boundary is
// a dot or the whole string), and never an IP literal unless the entry is that
// exact IP. This is deliberately strict — egress is the blast-radius control for
// a compromised tool.

// egressAllow is a compiled, immutable allowlist checked per-request.
type egressAllow struct {
	hosts []string // lower-cased, deduped, sorted (for a stable cache key)
}

func newEgressAllow(entries []string) egressAllow {
	seen := map[string]struct{}{}
	var hosts []string
	for _, e := range entries {
		h := strings.ToLower(strings.TrimSpace(e))
		if h == "" {
			continue
		}
		if _, ok := seen[h]; ok {
			continue
		}
		seen[h] = struct{}{}
		hosts = append(hosts, h)
	}
	sort.Strings(hosts)
	return egressAllow{hosts: hosts}
}

// key is a stable identifier for this allowlist (used to cache one proxy per
// distinct set of hosts).
func (a egressAllow) key() string { return strings.Join(a.hosts, ",") }

// permits reports whether host (a bare hostname, no port) is allowed. Matching is
// case-insensitive and boundary-safe: an allow entry matches the host itself or
// any subdomain of it.
func (a egressAllow) permits(host string) bool {
	host = strings.ToLower(strings.TrimSuffix(strings.TrimSpace(host), "."))
	if host == "" {
		return false
	}
	for _, allowed := range a.hosts {
		if host == allowed {
			return true
		}
		// Subdomain: host ends with "."+allowed so "notpypi.org" does not match
		// "pypi.org" but "files.pypi.org" does.
		if strings.HasSuffix(host, "."+allowed) {
			return true
		}
	}
	return false
}

// egressProxy is a host-side HTTP forward proxy that only relays to allowlisted
// hosts. It serves the CONNECT method (HTTPS tunnels) and plain HTTP proxying.
type egressProxy struct {
	allow    egressAllow
	server   *http.Server
	listener net.Listener
	dialer   *net.Dialer
}

// newEgressProxy binds a proxy on addr (":0" for an ephemeral port) enforcing
// allow. Call Addr() after Start to learn the bound port.
func newEgressProxy(allow egressAllow) *egressProxy {
	return &egressProxy{allow: allow, dialer: &net.Dialer{Timeout: 10 * time.Second}}
}

// Start binds and serves in the background. addr is a host:port to listen on.
func (p *egressProxy) Start(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	p.listener = ln
	p.server = &http.Server{Handler: http.HandlerFunc(p.handle), ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = p.server.Serve(ln) }()
	return nil
}

// Addr returns the bound listen address (host:port), or "" before Start.
func (p *egressProxy) Addr() string {
	if p.listener == nil {
		return ""
	}
	return p.listener.Addr().String()
}

// Stop shuts the proxy down.
func (p *egressProxy) Stop() {
	if p.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = p.server.Shutdown(ctx)
	}
}

func (p *egressProxy) handle(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodConnect {
		p.handleConnect(w, r)
		return
	}
	p.handleHTTP(w, r)
}

// hostOnly strips any :port from an authority.
func hostOnly(authority string) string {
	if h, _, err := net.SplitHostPort(authority); err == nil {
		return h
	}
	return authority
}

func (p *egressProxy) handleConnect(w http.ResponseWriter, r *http.Request) {
	host := hostOnly(r.Host)
	if !p.allow.permits(host) {
		http.Error(w, fmt.Sprintf("egress to %q is not on the allowlist", host), http.StatusForbidden)
		return
	}
	// The name is allowlisted; the address it resolves to still has to pass the
	// same IP guard the host-process tools use, or an allowlisted name pointed at
	// 169.254.169.254 (or the host's private network) would tunnel straight
	// through. This is also what lets http_request and web_fetch run inside the
	// sandbox without giving up their SSRF backstop.
	dst, err := guardedDialFunc(p.dialer, blockedIP)(r.Context(), "tcp", r.Host)
	if err != nil {
		http.Error(w, "upstream dial failed", http.StatusBadGateway)
		return
	}
	hj, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "hijack unsupported", http.StatusInternalServerError)
		_ = dst.Close()
		return
	}
	client, _, err := hj.Hijack()
	if err != nil {
		_ = dst.Close()
		return
	}
	_, _ = client.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
	go func() { _, _ = io.Copy(dst, client); _ = dst.Close() }()
	go func() { _, _ = io.Copy(client, dst); _ = client.Close() }()
}

func (p *egressProxy) handleHTTP(w http.ResponseWriter, r *http.Request) {
	host := hostOnly(r.Host)
	if !p.allow.permits(host) {
		http.Error(w, fmt.Sprintf("egress to %q is not on the allowlist", host), http.StatusForbidden)
		return
	}
	// Rebuild an outbound request; proxied requests carry an absolute URI.
	outReq, err := http.NewRequestWithContext(r.Context(), r.Method, r.URL.String(), r.Body)
	if err != nil {
		http.Error(w, "bad proxied request", http.StatusBadRequest)
		return
	}
	for k, vs := range r.Header {
		for _, v := range vs {
			outReq.Header.Add(k, v)
		}
	}
	client := &http.Client{
		Timeout:       30 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		// Same IP backstop as the CONNECT path: allowlisting the name is not
		// enough if the name resolves somewhere internal.
		Transport: &http.Transport{DialContext: guardedDialFunc(p.dialer, blockedIP)},
	}
	resp, err := client.Do(outReq)
	if err != nil {
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	for k, vs := range resp.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

// egressProxyPool caches one running proxy per distinct allowlist so repeated
// sessions with the same grants reuse a single listener instead of leaking one
// per run.
type egressProxyPool struct {
	mu      sync.Mutex
	proxies map[string]*egressProxy
}

func newEgressProxyPool() *egressProxyPool {
	return &egressProxyPool{proxies: map[string]*egressProxy{}}
}

// get returns a running proxy for allow, starting one bound to the loopback host
// on an ephemeral port if none exists yet. An empty allowlist returns (nil, nil)
// — the caller keeps the current no-allowlist behavior.
func (pl *egressProxyPool) get(allow egressAllow) (*egressProxy, error) {
	if len(allow.hosts) == 0 {
		return nil, nil
	}
	pl.mu.Lock()
	defer pl.mu.Unlock()
	if p, ok := pl.proxies[allow.key()]; ok {
		return p, nil
	}
	p := newEgressProxy(allow)
	// Bind on all interfaces so the container can reach it via the host gateway.
	if err := p.Start("0.0.0.0:0"); err != nil {
		return nil, err
	}
	pl.proxies[allow.key()] = p
	return p, nil
}

// stopAll tears down every pooled proxy (called on sandbox shutdown).
func (pl *egressProxyPool) stopAll() {
	pl.mu.Lock()
	defer pl.mu.Unlock()
	for k, p := range pl.proxies {
		p.Stop()
		delete(pl.proxies, k)
	}
}
