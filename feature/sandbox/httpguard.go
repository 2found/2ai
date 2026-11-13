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
	"net/netip"
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

// blockedRanges is every address range an outbound agent tool must never
// reach: anything that is not ordinary public unicast. IPv4-mapped IPv6
// (::ffff:a.b.c.d) is unmapped before the lookup, so it is checked as the IPv4
// address it is. Transition prefixes that embed or tunnel to an IPv4 address
// (NAT64, 6to4, Teredo, IPv4-compatible) are refused whole: the embedded
// address is attacker-chosen and may be private.
var blockedRanges = []struct {
	prefix netip.Prefix
	why    string
}{
	{netip.MustParsePrefix("0.0.0.0/8"), "this-network"},
	{netip.MustParsePrefix("10.0.0.0/8"), "private network"},
	{netip.MustParsePrefix("100.64.0.0/10"), "carrier-grade NAT"},
	{netip.MustParsePrefix("127.0.0.0/8"), "loopback"},
	{netip.MustParsePrefix("169.254.0.0/16"), "link-local"}, // cloud metadata
	{netip.MustParsePrefix("172.16.0.0/12"), "private network"},
	{netip.MustParsePrefix("192.0.0.0/24"), "IETF protocol assignment"},
	{netip.MustParsePrefix("192.0.2.0/24"), "documentation"},
	{netip.MustParsePrefix("192.88.99.0/24"), "6to4 relay"},
	{netip.MustParsePrefix("192.168.0.0/16"), "private network"},
	{netip.MustParsePrefix("198.18.0.0/15"), "benchmarking"},
	{netip.MustParsePrefix("198.51.100.0/24"), "documentation"},
	{netip.MustParsePrefix("203.0.113.0/24"), "documentation"},
	{netip.MustParsePrefix("224.0.0.0/4"), "multicast"},
	{netip.MustParsePrefix("240.0.0.0/4"), "reserved"}, // includes 255.255.255.255
	{netip.MustParsePrefix("::/128"), "unspecified"},
	{netip.MustParsePrefix("::1/128"), "loopback"},
	{netip.MustParsePrefix("::/96"), "IPv4-compatible"},
	{netip.MustParsePrefix("64:ff9b::/96"), "NAT64"},
	{netip.MustParsePrefix("64:ff9b:1::/48"), "NAT64"},
	{netip.MustParsePrefix("100::/64"), "discard"},
	{netip.MustParsePrefix("2001::/32"), "Teredo"},
	{netip.MustParsePrefix("2001:db8::/32"), "documentation"},
	{netip.MustParsePrefix("2002::/16"), "6to4"},
	{netip.MustParsePrefix("fc00::/7"), "private network"},
	{netip.MustParsePrefix("fe80::/10"), "link-local"},
	{netip.MustParsePrefix("fec0::/10"), "site-local"},
	{netip.MustParsePrefix("ff00::/8"), "multicast"},
}

// blockedIP reports whether ip is one an outbound agent tool must never reach,
// with a human-readable reason for the model/audit. An unparseable address is
// blocked.
func blockedIP(ip net.IP) (bool, string) {
	addr, ok := netip.AddrFromSlice(ip)
	if !ok {
		return true, "invalid address"
	}
	addr = addr.Unmap()
	for _, r := range blockedRanges {
		if r.prefix.Contains(addr) {
			return true, r.why
		}
	}
	return false, ""
}
