package sandbox

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/2found/2ai/core/agentcore"
)

// Every network tool says where a call can connect, and refuses to name a
// host list when it can reach anywhere; every tool returning outside text is
// framed as external content.
func TestNetworkToolsDeclareHostsAndExternalContent(t *testing.T) {
	sb := NewHostSandbox() // any non-nil sandbox: only the limits matter here
	pinned := NewShellTool(sb, agentcore.SandboxLimits{Network: true, NetworkAllow: []string{"api.example.com:8443", "docs.example.com"}}, nil)
	if hosts, err := pinned.NetworkHosts(`{}`); err != nil || !slices.Equal(hosts, []string{"api.example.com", "docs.example.com"}) || !pinned.ExternalContent() {
		t.Fatalf("pinned shell hosts=%v err=%v", hosts, err)
	}
	offline := NewShellTool(sb, agentcore.SandboxLimits{}, nil)
	if _, err := offline.NetworkHosts(`{}`); err == nil || offline.ExternalContent() {
		t.Fatal("an offline shell claimed hosts or external output")
	}
	for name, tool := range map[string]agentcore.NetworkTool{
		"open shell":   NewShellTool(sb, agentcore.SandboxLimits{Network: true, NetworkAllow: []string{AnyHost}}, nil),
		"host shell":   NewShellTool(nil, agentcore.SandboxLimits{}, nil),
		"host browser": NewBrowserTool(nil, nil, BrowserUseLimits("example.com"), ""),
		"open browser": NewBrowserTool(sb, nil, BrowserUseLimits(AnyHost), ""),
		"web_search":   NewWebSearchTool(searchStub{}, 0),
	} {
		if hosts, err := tool.NetworkHosts(`{"query":"q"}`); err == nil {
			t.Fatalf("%s can reach any host but declared %v", name, hosts)
		}
	}
	browser := NewBrowserTool(sb, nil, BrowserUseLimits("example.com"), "")
	if hosts, err := browser.NetworkHosts(`{}`); err != nil || !slices.Equal(hosts, []string{"example.com"}) || !browser.ExternalContent() {
		t.Fatalf("pinned browser hosts=%v err=%v", hosts, err)
	}
	search := NewWebSearchTool(NewDuckDuckGoSearch(nil), 0)
	if hosts, err := search.NetworkHosts(`{"query":"q"}`); err != nil || !slices.Equal(hosts, []string{"html.duckduckgo.com"}) || !search.ExternalContent() {
		t.Fatalf("web_search hosts=%v err=%v", hosts, err)
	}
	fetch := NewWebFetchTool(nil, nil)
	if hosts, err := fetch.NetworkHosts(`{"url":"https://Docs.Example.com/a"}`); err != nil || !slices.Equal(hosts, []string{"docs.example.com"}) || !fetch.ExternalContent() {
		t.Fatalf("web_fetch hosts=%v err=%v", hosts, err)
	}
	if _, err := fetch.NetworkHosts(`{"url":"https://{{cred:HOST}}/a"}`); err == nil {
		t.Fatal("web_fetch accepted a placeholder host")
	}
	var _ agentcore.ExternalContentTool = NewHTTPRequestTool(nil)
}

type searchStub struct{}

func (searchStub) Search(context.Context, string, int) ([]SearchResult, error) { return nil, nil }

// web_fetch follows a redirect within the requested host and its subdomains,
// without a Referer, and refuses one to another host, so NetworkHosts is exact.
func TestWebFetchKeepsRedirectsOnTheRequestedHost(t *testing.T) {
	var referer string
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("elsewhere"))
	}))
	defer other.Close()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/same":
			http.Redirect(w, r, "/final", http.StatusFound)
		case "/away":
			http.Redirect(w, r, strings.Replace(other.URL, "127.0.0.1", "localhost", 1)+"/x", http.StatusFound)
		default:
			referer = r.Header.Get("Referer")
			_, _ = w.Write([]byte("landed"))
		}
	}))
	defer srv.Close()
	tool := NewWebFetchTool(nil, nil)
	tool.AllowAllIPsForTest()
	out, err := tool.Run(context.Background(), `{"url":"`+srv.URL+`/same?key=secret"}`)
	if err != nil || !strings.Contains(out, "landed") || referer != "" {
		t.Fatalf("same-host redirect: out=%q err=%v referer=%q", out, err, referer)
	}
	if _, err := tool.Run(context.Background(), `{"url":"`+srv.URL+`/away"}`); err == nil || !strings.Contains(err.Error(), "redirected off") {
		t.Fatalf("off-host redirect was followed: %v", err)
	}
}
