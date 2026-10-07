package sandbox

// This file holds the host-side outbound egress guard: the SSRF backstop shared
// by every tool here that makes a request from the host process (http_request,
// web_fetch) and by hosts needing the same protection for their own outbound
// calls (see NewGuardedClient). It is the counterpart to egress.go, which
// controls what a *container* may reach.

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"strings"
)

// parseAbsoluteURL parses raw and rejects anything that is not an absolute URL
// with a host — a relative or host-less URL can never be allowlist-checked.
func parseAbsoluteURL(raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("invalid url: %w", err)
	}
	if !u.IsAbs() || u.Host == "" {
		return nil, fmt.Errorf("url must be absolute with a host")
	}
	return u, nil
}

// guardedDial wraps a dialer so the IP actually connected to is checked at
// connect time. This is the SSRF backstop and the DNS-rebinding fix: even if an
// allowlisted hostname resolves to a blocked address (loopback, private,
// link-local — including the 169.254.169.254 cloud-metadata endpoint), the dial
// is refused. If any resolved IP for the host is blocked, the whole connection
// is rejected rather than racing to a "good" one. The IP policy is read from
// t.ipBlocked at dial time.
func (t *HTTPTool) guardedDial(dialer *net.Dialer) func(ctx context.Context, network, addr string) (net.Conn, error) {
	// Read t.ipBlocked at dial time (not construction) so a test seam that
	// relaxes the IP policy after New still takes effect.
	return guardedDialFunc(dialer, func(ip net.IP) (bool, string) { return t.ipBlocked(ip) })
}

// guardedDialFunc is the SSRF-guarded DialContext shared by every host-process
// outbound tool here (http_request, web_fetch). It resolves the host, refuses the
// connection if any resolved IP is blocked by ipBlocked, then dials a checked
// address directly — closing the DNS-rebinding TOCTOU gap.
func guardedDialFunc(dialer *net.Dialer, ipBlocked func(net.IP) (bool, string)) func(ctx context.Context, network, addr string) (net.Conn, error) {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, err
		}
		ips, err := net.DefaultResolver.LookupIP(ctx, "ip", host)
		if err != nil {
			return nil, err
		}
		if len(ips) == 0 {
			return nil, fmt.Errorf("no addresses for host %q", host)
		}
		for _, ip := range ips {
			if blocked, why := ipBlocked(ip); blocked {
				return nil, fmt.Errorf("refusing to connect to %s (%s)", ip, why)
			}
		}
		// All resolved IPs passed; dial the first one directly so the connection
		// goes to a checked address, not a re-resolved one.
		return dialer.DialContext(ctx, network, net.JoinHostPort(ips[0].String(), port))
	}
}

// blockedIP reports whether ip is one an outbound agent tool must never reach,
// with a human-readable reason for the model/audit.
func blockedIP(ip net.IP) (bool, string) {
	switch {
	case ip.IsLoopback():
		return true, "loopback"
	case ip.IsLinkLocalUnicast(), ip.IsLinkLocalMulticast():
		// 169.254.0.0/16 and fe80::/10 — covers the cloud-metadata endpoint.
		return true, "link-local"
	case ip.IsPrivate():
		return true, "private network"
	case ip.IsUnspecified():
		return true, "unspecified"
	case ip.IsMulticast():
		return true, "multicast"
	default:
		return false, ""
	}
}
