package sandbox

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

// The allowlist decision is the security core: it must accept the exact host and
// its subdomains, and reject look-alikes, unrelated hosts, and IP literals.
func TestEgressAllowPermits(t *testing.T) {
	allow := newEgressAllow([]string{"pypi.org", "files.pythonhosted.org", " PyPI.ORG "})
	cases := []struct {
		host string
		ok   bool
	}{
		{"pypi.org", true},
		{"PyPI.org", true},       // case-insensitive
		{"pypi.org.", true},      // trailing dot (FQDN form)
		{"files.pypi.org", true}, // subdomain
		{"a.b.pypi.org", true},   // nested subdomain
		{"files.pythonhosted.org", true},
		{"notpypi.org", false},       // suffix look-alike must not match
		{"pypi.org.evil.com", false}, // host that merely contains the entry
		{"evil.com", false},
		{"", false},
		{"1.2.3.4", false}, // IP literal, not on list
	}
	for _, tc := range cases {
		if got := allow.permits(tc.host); got != tc.ok {
			t.Errorf("permits(%q) = %v, want %v", tc.host, got, tc.ok)
		}
	}
}

// newEgressAllow dedupes and sorts so the pool key is stable regardless of input
// order or duplicates.
func TestEgressAllowKeyStable(t *testing.T) {
	a := newEgressAllow([]string{"b.com", "a.com", "b.com", "  a.com "})
	if a.key() != "a.com,b.com" {
		t.Fatalf("key = %q, want a.com,b.com", a.key())
	}
	if len(a.hosts) != 2 {
		t.Fatalf("hosts = %v, want 2 deduped", a.hosts)
	}
}

// The live proxy must tunnel/relay allowlisted hosts and 403 everything else,
// for both CONNECT (HTTPS) and plain HTTP.
func TestEgressProxyBlocksNonAllowlisted(t *testing.T) {
	proxy := newEgressProxy(newEgressAllow([]string{"allowed.example"}))
	if err := proxy.Start("127.0.0.1:0"); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer proxy.Stop()

	// Plain-HTTP proxying to a disallowed host → 403 from the proxy itself.
	proxyURL, _ := url.Parse("http://" + proxy.Addr())
	client := &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(proxyURL)}}
	resp, err := client.Get("http://blocked.example/")
	if err != nil {
		t.Fatalf("proxied GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("blocked host status = %d, want 403 (body: %s)", resp.StatusCode, body)
	}

	// A CONNECT to a disallowed host must also be refused before any dial.
	connResp, err := connectViaProxy(proxy.Addr(), "blocked.example:443")
	if err != nil {
		t.Fatalf("CONNECT: %v", err)
	}
	if !strings.Contains(connResp, "403") {
		t.Fatalf("CONNECT to blocked host = %q, want 403", connResp)
	}
}

// The pool caches one proxy per distinct allowlist and returns (nil,nil) for an
// empty list (the caller keeps the no-allowlist default behavior).
func TestEgressProxyPool(t *testing.T) {
	pool := newEgressProxyPool()
	defer pool.stopAll()

	if p, err := pool.get(newEgressAllow(nil)); err != nil || p != nil {
		t.Fatalf("empty allowlist = (%v,%v), want (nil,nil)", p, err)
	}
	a := newEgressAllow([]string{"pypi.org"})
	p1, err := pool.get(a)
	if err != nil || p1 == nil {
		t.Fatalf("get: %v %v", p1, err)
	}
	p2, _ := pool.get(newEgressAllow([]string{"pypi.org"}))
	if p1 != p2 {
		t.Fatal("pool must reuse one proxy per identical allowlist")
	}
	p3, _ := pool.get(newEgressAllow([]string{"npmjs.org"}))
	if p3 == p1 {
		t.Fatal("distinct allowlists must get distinct proxies")
	}
}

// --- docker arg construction (no live Docker needed) ---

// run_shell never gets network, so egress args must always be --network none with
// no proxy env — offline regardless of any allowlist config.
func TestEgressArgsRunShellIsOffline(t *testing.T) {
	s := NewDockerSandbox()
	args, env := s.egressNetworkArgs(agentcore.SandboxLimits{Network: false, NetworkAllow: []string{"pypi.org"}})
	if strings.Join(args, " ") != "--network none" {
		t.Fatalf("no-network args = %v, want [--network none]", args)
	}
	if len(env) != 0 {
		t.Fatalf("no-network env = %v, want none", env)
	}
}

// computer_use with network but no allowlist keeps the current default-network
// behavior: no --network flag, no proxy env.
func TestEgressArgsOpenNetworkUnchanged(t *testing.T) {
	s := NewDockerSandbox()
	args, env := s.egressNetworkArgs(agentcore.SandboxLimits{Network: true})
	if len(args) != 0 || len(env) != 0 {
		t.Fatalf("open network = args %v env %v, want both empty", args, env)
	}
}

// computer_use with network + allowlist routes through the proxy: proxy env is
// injected and the container gets a host-gateway host entry.
func TestEgressArgsAllowlistRoutesThroughProxy(t *testing.T) {
	s := NewDockerSandbox()
	defer s.StopEgress()
	args, env := s.egressNetworkArgs(agentcore.SandboxLimits{Network: true, NetworkAllow: []string{"pypi.org"}})
	if !containsPair(args, "--add-host", "host.docker.internal:host-gateway") {
		t.Fatalf("args %v missing host-gateway mapping", args)
	}
	for _, key := range []string{"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"} {
		v, ok := env[key]
		if !ok || !strings.HasPrefix(v, "http://host.docker.internal:") {
			t.Fatalf("env[%s] = %q, want host.docker.internal proxy URL", key, v)
		}
	}
	if env["NO_PROXY"] == "" {
		t.Fatal("NO_PROXY should exempt loopback so the container never proxies to itself")
	}
}

// --- helpers ---

// connectViaProxy sends a raw CONNECT and returns the proxy's status line.
func connectViaProxy(proxyAddr, target string) (string, error) {
	conn, err := net.DialTimeout("tcp", proxyAddr, 3*time.Second)
	if err != nil {
		return "", err
	}
	defer conn.Close()
	if _, err := conn.Write([]byte("CONNECT " + target + " HTTP/1.1\r\nHost: " + target + "\r\n\r\n")); err != nil {
		return "", err
	}
	_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	line, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		return "", err
	}
	return line, nil
}

func containsPair(args []string, a, b string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == a && args[i+1] == b {
			return true
		}
	}
	return false
}
