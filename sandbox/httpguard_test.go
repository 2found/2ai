package sandbox

import (
	"net"
	"testing"
)

func TestBlockedIP(t *testing.T) {
	cases := []struct {
		ip      string
		blocked bool
	}{
		{"169.254.169.254", true},        // cloud-metadata endpoint (link-local)
		{"127.0.0.1", true},              // loopback
		{"127.255.0.1", true},            // whole loopback /8
		{"10.1.2.3", true},               // private
		{"192.168.0.5", true},            // private
		{"172.16.4.4", true},             // private
		{"172.31.255.255", true},         // private upper edge
		{"0.0.0.0", true},                // unspecified
		{"0.1.2.3", true},                // this-network /8
		{"100.64.0.1", true},             // CGNAT
		{"100.127.255.254", true},        // CGNAT upper edge
		{"198.18.0.1", true},             // benchmarking
		{"198.19.255.255", true},         // benchmarking upper edge
		{"192.0.0.8", true},              // IETF protocol assignments
		{"192.0.2.1", true},              // TEST-NET-1
		{"198.51.100.7", true},           // TEST-NET-2
		{"203.0.113.9", true},            // TEST-NET-3
		{"192.88.99.1", true},            // 6to4 relay anycast
		{"224.0.0.1", true},              // multicast
		{"239.255.255.250", true},        // SSDP multicast
		{"240.0.0.1", true},              // reserved
		{"255.255.255.255", true},        // broadcast
		{"::", true},                     // ipv6 unspecified
		{"::1", true},                    // ipv6 loopback
		{"fe80::1", true},                // ipv6 link-local
		{"fec0::1", true},                // deprecated site-local
		{"fd00::1", true},                // unique local
		{"ff02::1", true},                // ipv6 multicast
		{"::ffff:127.0.0.1", true},       // IPv4-mapped loopback
		{"::ffff:169.254.169.254", true}, // IPv4-mapped metadata
		{"::ffff:10.0.0.1", true},        // IPv4-mapped private
		{"::10.0.0.1", true},             // IPv4-compatible (deprecated)
		{"64:ff9b::a9fe:a9fe", true},     // NAT64 of 169.254.169.254
		{"64:ff9b::808:808", true},       // NAT64 is refused whole
		{"64:ff9b:1::1", true},           // local-use NAT64
		{"2002:a9fe:a9fe::1", true},      // 6to4 of the metadata endpoint
		{"2001:0:4136:e378::1", true},    // Teredo
		{"2001:db8::1", true},            // documentation
		{"100::1", true},                 // discard-only
		{"8.8.8.8", false},               // public
		{"1.1.1.1", false},               // public
		{"100.63.255.255", false},        // just below CGNAT
		{"100.128.0.0", false},           // just above CGNAT
		{"198.20.0.1", false},            // just above benchmarking
		{"172.32.0.1", false},            // just above 172.16/12
		{"::ffff:8.8.8.8", false},        // IPv4-mapped public
		{"2606:4700:4700::1111", false},  // public IPv6
		{"2001:4860:4860::8888", false},  // public IPv6 outside Teredo /32
	}
	for _, c := range cases {
		ip := net.ParseIP(c.ip)
		if ip == nil {
			t.Fatalf("bad test ip %q", c.ip)
		}
		got, why := blockedIP(ip)
		if got != c.blocked {
			t.Errorf("blockedIP(%s) = %v (%s), want %v", c.ip, got, why, c.blocked)
		}
		if got && why == "" {
			t.Errorf("blockedIP(%s) blocked without a reason", c.ip)
		}
	}
	if blocked, _ := blockedIP(net.IP{1, 2, 3}); !blocked {
		t.Error("a malformed address must be blocked")
	}
}

func TestParseAbsoluteURLRejectsRelative(t *testing.T) {
	for _, raw := range []string{"/path/only", "example.com/x", "", "   "} {
		if _, err := parseAbsoluteURL(raw); err == nil {
			t.Errorf("expected error for %q", raw)
		}
	}
	if _, err := parseAbsoluteURL("https://api.example.com/v1"); err != nil {
		t.Errorf("valid absolute url rejected: %v", err)
	}
}
