package sandbox

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

// The allowlist decision is the security core: it must accept the exact host and
// its subdomains on permitted ports, and reject look-alikes, unrelated hosts, IP
// literals, and every port the entry does not grant.
func TestEgressAllowPermits(t *testing.T) {
	allow := newEgressAllow([]string{"pypi.org", "files.pythonhosted.org", " PyPI.ORG ", "registry.example:8443"})
	cases := []struct {
		host, port string
		ok         bool
	}{
		{"pypi.org", "443", true},
		{"pypi.org", "80", true},
		{"PyPI.org", "443", true},       // case-insensitive
		{"pypi.org.", "443", true},      // trailing dot (FQDN form)
		{"files.pypi.org", "443", true}, // subdomain
		{"a.b.pypi.org", "443", true},   // nested subdomain
		{"files.pythonhosted.org", "443", true},
		{"pypi.org", "22", false},          // port not granted
		{"pypi.org", "8443", false},        // port not granted
		{"pypi.org", "", false},            // no port is never a match
		{"registry.example", "8443", true}, // explicit port
		{"registry.example", "443", false}, // explicit port replaces the defaults
		{"notpypi.org", "443", false},      // suffix look-alike must not match
		{"pypi.org.evil.com", "443", false},
		{"evil.com", "443", false},
		{"", "443", false},
		{"1.2.3.4", "443", false}, // IP literal, not on list
	}
	for _, tc := range cases {
		if got := allow.permits(tc.host, tc.port); got != tc.ok {
			t.Errorf("permits(%q, %q) = %v, want %v", tc.host, tc.port, got, tc.ok)
		}
	}
	if allow.unrestricted {
		t.Fatal("an allowlist without AnyHost must not be unrestricted")
	}
	if !newEgressAllow([]string{"pypi.org", AnyHost}).unrestricted {
		t.Fatal("AnyHost must mark the allowlist unrestricted")
	}
}

// newEgressAllow dedupes and sorts so the pool key is stable regardless of input
// order or duplicates.
func TestEgressAllowKeyStable(t *testing.T) {
	a := newEgressAllow([]string{"b.com", "a.com", "b.com", "  a.com ", "a.com:8080", ""})
	if a.key() != "a.com,a.com:8080,b.com" {
		t.Fatalf("key = %q", a.key())
	}
}

// The live proxy must relay only allowlisted host:port pairs and refuse
// everything else, for both CONNECT (HTTPS) and plain HTTP.
func TestEgressProxyBlocksNonAllowlisted(t *testing.T) {
	proxy := newEgressProxy(newEgressAllow([]string{"allowed.example", "localhost"}))
	if err := proxy.Start("127.0.0.1:0"); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer proxy.Stop()

	proxyURL, _ := url.Parse("http://" + proxy.Addr())
	client := &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(proxyURL)}}
	for _, target := range []string{"http://blocked.example/", "http://allowed.example:22/"} {
		resp, err := client.Get(target)
		if err != nil {
			t.Fatalf("proxied GET %s: %v", target, err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Fatalf("GET %s status = %d, want 403 (body: %s)", target, resp.StatusCode, body)
		}
	}

	cases := []struct{ target, want string }{
		{"blocked.example:443", "403"},
		{"allowed.example:22", "403"}, // allowed host, port not granted
		{"allowed.example", "400"},    // CONNECT without a port
		// Allowlisted name that resolves to loopback: the IP guard still refuses.
		{"localhost:443", "502"},
	}
	for _, tc := range cases {
		status, err := connectViaProxy(proxy.Addr(), tc.target)
		if err != nil {
			t.Fatalf("CONNECT %s: %v", tc.target, err)
		}
		if !strings.Contains(status, tc.want) {
			t.Errorf("CONNECT %s = %q, want %s", tc.target, status, tc.want)
		}
	}

	// An origin-form request (what a client sends when it is not proxying) has
	// no absolute URL to check and must be refused.
	conn, err := net.DialTimeout("tcp", proxy.Addr(), 3*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	_, _ = io.WriteString(conn, "GET / HTTP/1.1\r\nHost: allowed.example\r\n\r\n")
	_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	line, _ := bufio.NewReader(conn).ReadString('\n')
	if !strings.Contains(line, "400") {
		t.Fatalf("origin-form request = %q, want 400", line)
	}
}

// The pool caches one proxy per distinct allowlist, never binds a wildcard
// address, and refuses an empty allowlist.
func TestEgressProxyPool(t *testing.T) {
	pool := newEgressProxyPool()
	defer pool.stopAll()

	if _, err := pool.get(newEgressAllow(nil), "127.0.0.1"); err == nil {
		t.Fatal("empty allowlist must be refused")
	}
	for _, bind := range []string{"0.0.0.0", "::", "", "not-an-ip"} {
		if _, err := pool.get(newEgressAllow([]string{"pypi.org"}), bind); err == nil {
			t.Fatalf("bind %q must be refused", bind)
		}
	}
	a := newEgressAllow([]string{"pypi.org"})
	p1, err := pool.get(a, "127.0.0.1")
	if err != nil || p1 == nil {
		t.Fatalf("get: %v %v", p1, err)
	}
	if host, _, _ := net.SplitHostPort(p1.Addr()); host != "127.0.0.1" {
		t.Fatalf("proxy bound %s, want the requested address only", p1.Addr())
	}
	p2, _ := pool.get(newEgressAllow([]string{"pypi.org"}), "127.0.0.1")
	if p1 != p2 {
		t.Fatal("pool must reuse one proxy per identical allowlist")
	}
	p3, _ := pool.get(newEgressAllow([]string{"npmjs.org"}), "127.0.0.1")
	if p3 == p1 {
		t.Fatal("distinct allowlists must get distinct proxies")
	}
}

// --- docker arg construction (fake docker CLI, no daemon needed) ---

// fakeDocker installs a stand-in docker CLI that logs every invocation and
// answers the few queries the sandbox makes. gateway is what `network inspect`
// reports; extra is shell run before the default per-subcommand answers.
type fakeDocker struct {
	bin string
	log string
}

func newFakeDocker(t *testing.T, gateway, extra string) fakeDocker {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake docker CLI is a POSIX shell script")
	}
	dir := t.TempDir()
	f := fakeDocker{bin: filepath.Join(dir, "docker"), log: filepath.Join(dir, "calls.log")}
	script := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '" + f.log + "'\n" + extra + `
case "$1 $2" in
  "network create") echo netid ;;
  "network inspect") echo "` + gateway + ` " ;;
  "inspect -f") echo true ;;
esac
exit 0
`
	if err := os.WriteFile(f.bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return f
}

func (f fakeDocker) calls(t *testing.T) []string {
	t.Helper()
	data, err := os.ReadFile(f.log)
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimSpace(string(data)), "\n")
}

func (f fakeDocker) called(t *testing.T, prefix string) int {
	n := 0
	for _, c := range f.calls(t) {
		if strings.HasPrefix(c, prefix) {
			n++
		}
	}
	return n
}

// run_shell never gets network, so egress args must always be --network none with
// no proxy env — offline regardless of any allowlist config.
func TestEgressArgsNoNetworkIsOffline(t *testing.T) {
	s := NewDockerSandbox()
	args, env, err := s.egressNetworkArgs(context.Background(), agentcore.SandboxLimits{Network: false, NetworkAllow: []string{"pypi.org"}})
	if err != nil || strings.Join(args, " ") != "--network none" || len(env) != 0 {
		t.Fatalf("no-network = args %v env %v err %v, want [--network none]", args, env, err)
	}
}

// Network with an empty allowlist used to be silently open; it is now refused,
// and open egress needs the explicit AnyHost entry.
func TestEgressArgsEmptyAllowlistIsRefused(t *testing.T) {
	s := NewDockerSandbox()
	if _, _, err := s.egressNetworkArgs(context.Background(), agentcore.SandboxLimits{Network: true}); !errors.Is(err, errNoEgressAllowlist) {
		t.Fatalf("empty allowlist err = %v, want errNoEgressAllowlist", err)
	}
	args, env, err := s.egressNetworkArgs(context.Background(), agentcore.SandboxLimits{Network: true, NetworkAllow: []string{AnyHost}})
	if err != nil || len(args) != 0 || len(env) != 0 {
		t.Fatalf("AnyHost = args %v env %v err %v, want the default bridge", args, env, err)
	}
	// The refusal reaches Exec before any container is started.
	fake := newFakeDocker(t, "127.0.0.1", "")
	s = NewDockerSandbox(WithDockerBinary(fake.bin))
	if _, err := s.Exec(context.Background(), agentcore.SandboxExec{Argv: []string{"true"}, Constraints: agentcore.SandboxLimits{Network: true}}); !errors.Is(err, errNoEgressAllowlist) {
		t.Fatalf("Exec err = %v, want errNoEgressAllowlist", err)
	}
	if n := fake.called(t, "run"); n != 0 {
		t.Fatalf("docker run invoked %d times for a refused request", n)
	}
}

// An allowlist confines the container to the internal egress network and
// points it at a proxy listening only on that network's gateway.
func TestEgressArgsAllowlistUsesInternalNetwork(t *testing.T) {
	fake := newFakeDocker(t, "127.0.0.1", "")
	s := NewDockerSandbox(WithDockerBinary(fake.bin))
	lim := agentcore.SandboxLimits{Network: true, NetworkAllow: []string{"pypi.org"}}
	args, env, err := s.egressNetworkArgs(context.Background(), lim)
	if err != nil {
		t.Fatalf("egressNetworkArgs: %v", err)
	}
	if len(args) != 2 || args[0] != "--network" || !strings.HasPrefix(args[1], "agentray-egress-") {
		t.Fatalf("args = %v, want --network <internal egress network>", args)
	}
	for _, key := range []string{"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"} {
		if !strings.HasPrefix(env[key], "http://127.0.0.1:") {
			t.Fatalf("env[%s] = %q, want the gateway-bound proxy", key, env[key])
		}
	}
	create := ""
	for _, c := range fake.calls(t) {
		if strings.HasPrefix(c, "network create") {
			create = c
		}
	}
	for _, want := range []string{"--internal", "com.docker.network.bridge.enable_icc=false", args[1]} {
		if !strings.Contains(create, want) {
			t.Fatalf("network create %q missing %q", create, want)
		}
	}
	// A second allowlisted run reuses the network; StopEgress removes it.
	if _, _, err := s.egressNetworkArgs(context.Background(), lim); err != nil {
		t.Fatal(err)
	}
	if n := fake.called(t, "network create"); n != 1 {
		t.Fatalf("network created %d times, want 1", n)
	}
	s.StopEgress()
	if n := fake.called(t, "network rm "+args[1]); n != 1 {
		t.Fatalf("StopEgress removed the network %d times, want 1", n)
	}
}

// When the proxy cannot listen where the container can reach it (Docker
// Desktop: the bridge is inside a VM), the request fails — it never falls back
// to an open or proxy-env-only network.
func TestEgressArgsFailClosedWhenGatewayUnbindable(t *testing.T) {
	fake := newFakeDocker(t, "203.0.113.1", "")
	s := NewDockerSandbox(WithDockerBinary(fake.bin))
	defer s.StopEgress()
	_, _, err := s.egressNetworkArgs(context.Background(), agentcore.SandboxLimits{Network: true, NetworkAllow: []string{"pypi.org"}})
	if err == nil || !strings.Contains(err.Error(), "gateway 203.0.113.1") {
		t.Fatalf("err = %v, want a gateway listen failure", err)
	}

	failing := newFakeDocker(t, "127.0.0.1", `[ "$1 $2" = "network create" ] && { echo "permission denied" >&2; exit 1; }`)
	s = NewDockerSandbox(WithDockerBinary(failing.bin))
	_, _, err = s.egressNetworkArgs(context.Background(), agentcore.SandboxLimits{Network: true, NetworkAllow: []string{"pypi.org"}})
	if err == nil || !strings.Contains(err.Error(), "permission denied") {
		t.Fatalf("err = %v, want the network create failure", err)
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
