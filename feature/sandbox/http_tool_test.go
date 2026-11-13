package sandbox

import "testing"

func TestValidateURL(t *testing.T) {
	tool := NewHTTPRequestTool(nil, WithHTTPAllowHosts([]string{"api.example.com"}))

	if err := tool.validateURL("https://api.example.com/v1/thing"); err != nil {
		t.Errorf("allowlisted https host should pass: %v", err)
	}
	if err := tool.validateURL("https://evil.com/x"); err == nil {
		t.Error("non-allowlisted host should be refused")
	}
	if err := tool.validateURL("http://api.example.com/x"); err == nil {
		t.Error("plain http should be refused when https-only")
	}
	if err := tool.validateURL("ftp://api.example.com/x"); err == nil {
		t.Error("non-http scheme should be refused")
	}

	httpOK := NewHTTPRequestTool(nil, WithHTTPAllowHosts([]string{"api.example.com"}), WithHTTPAllowPlain(true))
	if err := httpOK.validateURL("http://api.example.com/x"); err != nil {
		t.Errorf("plain http should pass when explicitly allowed: %v", err)
	}
}

func TestHTTPToolNetworkHosts(t *testing.T) {
	tool := NewHTTPRequestTool(nil)
	hosts, err := tool.NetworkHosts(`{"url":"https://API.Example.com:8443/v1?q={{cred:K}}"}`)
	if err != nil || len(hosts) != 1 || hosts[0] != "api.example.com" {
		t.Fatalf("NetworkHosts = %v, %v", hosts, err)
	}
	for _, args := range []string{
		`not json`,
		`{"url":"/relative"}`,
		`{"url":"https://{{cred:HOST}}/x"}`, // where a secret goes must not depend on a secret
		`{"url":"https://{{cred:HOST}}.example.com/x"}`,
	} {
		if hosts, err := tool.NetworkHosts(args); err == nil {
			t.Errorf("NetworkHosts(%s) = %v, want an error", args, hosts)
		}
	}
}
