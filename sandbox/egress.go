package sandbox

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"
)

// Egress allowlist. When a session is granted network AND a non-empty
// NetworkAllow list, its container must reach only the listed hosts. The
// enforcement has two halves:
//
//  1. This host-side filtering forward-proxy, which hard-denies any host or port
//     not on the allowlist for both CONNECT (HTTPS) and plain HTTP — the
//     security core, unit-tested independently of Docker.
//  2. docker.go attaches the container to an internal network (no route off the
//     bridge) whose gateway address is the only place this proxy listens, so
//     the proxy is the container's one way out. A client that ignores the proxy
//     env has no route at all.
//
// The allowlist match is suffix-based on the DNS name: an entry "pypi.org"
// matches "pypi.org" and "sub.pypi.org" but never "notpypi.org" (the boundary is
// a dot or the whole string), and never an IP literal unless the entry is that
// exact IP. An entry without a port permits 80 and 443; "host:8443" permits that
// port only. This is deliberately strict — egress is the blast-radius control
// for a compromised tool.

// AnyHost is the NetworkAllow entry that grants unrestricted egress. Network
// with an empty allowlist is refused rather than silently opened, so open
// internet access is always this explicit, greppable choice.
const AnyHost = "*"

// egressDefaultPorts are what an allowlist entry without a port permits.
var egressDefaultPorts = []string{"80", "443"}

// egressRule is one compiled allowlist entry.
type egressRule struct {
	host string // lower-cased, no trailing dot
	port string // "" = egressDefaultPorts
}

// egressAllow is a compiled, immutable allowlist checked per-request.
type egressAllow struct {
	rules        []egressRule // deduped, sorted (for a stable cache key)
	unrestricted bool         // AnyHost was listed
}

func newEgressAllow(entries []string) egressAllow {
	seen := map[egressRule]struct{}{}
	var allow egressAllow
	for _, e := range entries {
		e = strings.ToLower(strings.TrimSpace(e))
		if e == "" {
			continue
		}
		if e == AnyHost {
			allow.unrestricted = true
			continue
		}
		rule := egressRule{host: e}
		if h, p, err := net.SplitHostPort(e); err == nil {
			rule = egressRule{host: h, port: p}
		}
		rule.host = strings.TrimSuffix(rule.host, ".")
		if _, ok := seen[rule]; ok || rule.host == "" {
			continue
		}
		seen[rule] = struct{}{}
		allow.rules = append(allow.rules, rule)
	}
	sort.Slice(allow.rules, func(i, j int) bool { return allow.rules[i].key() < allow.rules[j].key() })
	return allow
}

func (r egressRule) key() string {
	if r.port == "" {
		return r.host
	}
	return net.JoinHostPort(r.host, r.port)
}

// key is a stable identifier for this allowlist (used to cache one proxy per
// distinct set of rules).
func (a egressAllow) key() string {
	keys := make([]string, len(a.rules))
	for i, r := range a.rules {
		keys[i] = r.key()
	}
	return strings.Join(keys, ",")
}

// permits reports whether host:port is allowed. Matching is case-insensitive
// and boundary-safe: an entry matches the host itself or any subdomain of it.
func (a egressAllow) permits(host, port string) bool {
	host = strings.ToLower(strings.TrimSuffix(strings.TrimSpace(host), "."))
	if host == "" || port == "" {
		return false
	}
	for _, r := range a.rules {
		if host != r.host && !strings.HasSuffix(host, "."+r.host) {
			continue
		}
		if r.port == port || (r.port == "" && slices.Contains(egressDefaultPorts, port)) {
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

// newEgressProxy builds a proxy enforcing allow. Start binds it; Addr then
// reports the bound port.
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

func (p *egressProxy) deny(w http.ResponseWriter, host, port string) {
	http.Error(w, fmt.Sprintf("egress to %s is not on the allowlist", net.JoinHostPort(host, port)), http.StatusForbidden)
}

func (p *egressProxy) handleConnect(w http.ResponseWriter, r *http.Request) {
	// A CONNECT target is always host:port; anything else is not a tunnel
	// request this proxy can check.
	host, port, err := net.SplitHostPort(r.Host)
	if err != nil {
		http.Error(w, "CONNECT target must be host:port", http.StatusBadRequest)
		return
	}
	if !p.allow.permits(host, port) {
		p.deny(w, host, port)
		return
	}
	// The name is allowlisted; the address it resolves to still has to pass the
	// same IP guard the host-process tools use, or an allowlisted name pointed at
	// 169.254.169.254 (or the host's private network) would tunnel straight
	// through. This is also what lets http_request and web_fetch run inside the
	// sandbox without giving up their SSRF backstop.
	dst, err := guardedDialFunc(p.dialer, blockedIP)(r.Context(), "tcp", net.JoinHostPort(host, port))
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

// hopHeaders are connection-scoped and never forwarded (RFC 9110 §7.6.1), plus
// Proxy-Authorization, which is for this proxy and must not reach upstream.
var hopHeaders = []string{"Connection", "Keep-Alive", "Proxy-Authorization", "Proxy-Connection", "Te", "Trailer", "Transfer-Encoding", "Upgrade"}

func (p *egressProxy) handleHTTP(w http.ResponseWriter, r *http.Request) {
	// A proxied request carries an absolute URI. Check the host the request will
	// actually be sent to (the URL), not a Host header that may disagree.
	if !r.URL.IsAbs() || r.URL.Scheme != "http" || r.URL.Hostname() == "" {
		http.Error(w, "only absolute http:// URLs are proxied; use CONNECT for https", http.StatusBadRequest)
		return
	}
	host, port := r.URL.Hostname(), r.URL.Port()
	if port == "" {
		port = "80"
	}
	if !p.allow.permits(host, port) {
		p.deny(w, host, port)
		return
	}
	outReq, err := http.NewRequestWithContext(r.Context(), r.Method, r.URL.String(), r.Body)
	if err != nil {
		http.Error(w, "bad proxied request", http.StatusBadRequest)
		return
	}
	outReq.Header = r.Header.Clone()
	for _, h := range hopHeaders {
		outReq.Header.Del(h)
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

// egressProxyPool caches one running proxy per (listen address, allowlist) so
// repeated sessions with the same grants reuse a single listener instead of
// leaking one per run.
type egressProxyPool struct {
	mu      sync.Mutex
	proxies map[string]*egressProxy
}

func newEgressProxyPool() *egressProxyPool {
	return &egressProxyPool{proxies: map[string]*egressProxy{}}
}

// get returns a running proxy for allow listening on bindIP (an ephemeral
// port), starting one if none exists yet. bindIP is the narrowest address the
// container can reach — the egress network's gateway — never a wildcard.
func (pl *egressProxyPool) get(allow egressAllow, bindIP string) (*egressProxy, error) {
	if len(allow.rules) == 0 {
		return nil, fmt.Errorf("sandbox: egress proxy needs a non-empty allowlist")
	}
	if ip := net.ParseIP(bindIP); ip == nil || ip.IsUnspecified() {
		return nil, fmt.Errorf("sandbox: egress proxy needs a concrete listen address, got %q", bindIP)
	}
	key := bindIP + "|" + allow.key()
	pl.mu.Lock()
	defer pl.mu.Unlock()
	if p, ok := pl.proxies[key]; ok {
		return p, nil
	}
	p := newEgressProxy(allow)
	if err := p.Start(net.JoinHostPort(bindIP, "0")); err != nil {
		return nil, err
	}
	pl.proxies[key] = p
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
