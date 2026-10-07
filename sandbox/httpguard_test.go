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
		{"169.254.169.254", true}, // cloud-metadata endpoint (link-local)
		{"127.0.0.1", true},       // loopback
		{"10.1.2.3", true},        // private
		{"192.168.0.5", true},     // private
		{"172.16.4.4", true},      // private
		{"0.0.0.0", true},         // unspecified
		{"fe80::1", true},         // ipv6 link-local
		{"::1", true},             // ipv6 loopback
		{"8.8.8.8", false},        // public
		{"1.1.1.1", false},        // public
	}
	for _, c := range cases {
		ip := net.ParseIP(c.ip)
		if ip == nil {
			t.Fatalf("bad test ip %q", c.ip)
		}
		got, _ := blockedIP(ip)
		if got != c.blocked {
			t.Errorf("blockedIP(%s) = %v, want %v", c.ip, got, c.blocked)
		}
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
