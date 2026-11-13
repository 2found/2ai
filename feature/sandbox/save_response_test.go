package sandbox

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// save_as is what connects the network tools to the rest of the agent. These
// tests are about that connection, not about writing files: the assertion that
// matters is that what web_fetch downloaded is afterwards a file read_file and
// run_shell can open, at the path the agent was told.

func TestWebFetchSavesTheBodyIntoTheWorkspace(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/csv")
		_, _ = w.Write([]byte("id,name\n1,alpha\n2,beta\n"))
	}))
	defer srv.Close()

	ws := newTestWorkspace(t)
	tool := NewWebFetchTool(nil, ws)
	tool.AllowAllIPsForTest()

	out, err := tool.Run(context.Background(), `{"url":"`+srv.URL+`","save_as":"data/rows.csv"}`)
	if err != nil {
		t.Fatalf("web_fetch Run: %v", err)
	}
	if !strings.Contains(out, "data/rows.csv") {
		t.Fatalf("the receipt must name the path the agent reads back: %q", out)
	}
	if strings.Contains(out, "alpha") {
		t.Fatalf("a saved body should not also be returned inline — that defeats the point: %q", out)
	}

	got, err := os.ReadFile(filepath.Join(ws.Root(), "data", "rows.csv"))
	if err != nil {
		t.Fatalf("saved file is not in the workspace: %v", err)
	}
	if string(got) != "id,name\n1,alpha\n2,beta\n" {
		t.Fatalf("saved bytes = %q", got)
	}
}

func TestHTTPRequestSavesTheBodyIntoTheWorkspace(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	ws := newTestWorkspace(t)
	tool := NewHTTPRequestTool(nil,
		WithHTTPAllowHosts([]string{hostOf(t, srv.URL)}),
		WithHTTPAllowPlain(true),
		WithHTTPWorkspace(ws),
	)
	tool.AllowAllIPsForTest()

	out, err := tool.Run(context.Background(), `{"url":"`+srv.URL+`","save_as":"api/resp.json"}`)
	if err != nil {
		t.Fatalf("http_request Run: %v", err)
	}
	// The status still comes back: a 404 saved to disk is an error page in a
	// file, and an agent that only saw "saved 9 bytes" would parse it as data.
	if !strings.Contains(out, "status:") || !strings.Contains(out, "api/resp.json") {
		t.Fatalf("expected status + receipt, got %q", out)
	}
	got, err := os.ReadFile(filepath.Join(ws.Root(), "api", "resp.json"))
	if err != nil {
		t.Fatalf("saved file is not in the workspace: %v", err)
	}
	if string(got) != `{"ok":true}` {
		t.Fatalf("saved bytes = %q", got)
	}
}

func TestSaveAsCannotEscapeTheWorkspace(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("pwned"))
	}))
	defer srv.Close()

	ws := newTestWorkspace(t)
	tool := NewWebFetchTool(nil, ws)
	tool.AllowAllIPsForTest()

	for _, rel := range []string{"../escape.txt", "/etc/passwd", "a/../../escape.txt"} {
		out, err := tool.Run(context.Background(), `{"url":"`+srv.URL+`","save_as":"`+rel+`"}`)
		if err != nil {
			t.Fatalf("save_as %q: unexpected transport error: %v", rel, err)
		}
		// The write is refused — that is the security property. What the model gets
		// back is the fetched body plus a loud failure, because the request has
		// already been sent and hiding its result invites a retry of it.
		if !strings.Contains(out, "save_as FAILED") {
			t.Errorf("save_as %q was accepted; it must be refused: %q", rel, out)
		}
	}
	// Nothing may have landed next to the workspace.
	for _, abs := range []string{
		filepath.Join(filepath.Dir(ws.Root()), "escape.txt"),
		"/etc/passwd_agentray_probe",
	} {
		if _, err := os.Stat(abs); err == nil {
			t.Errorf("%s was written outside the workspace", abs)
		}
	}
}

func TestSaveAsWithoutAWorkspaceFailsLoudly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("body"))
	}))
	defer srv.Close()

	tool := NewWebFetchTool(nil, nil)
	tool.AllowAllIPsForTest()

	// Without a workspace the parameter is not offered at all, so the model has no
	// reason to send it.
	if props, ok := tool.Schema().Parameters["properties"].(map[string]any); !ok {
		t.Fatal("schema has no properties map")
	} else if _, offered := props["save_as"]; offered {
		t.Error("save_as is advertised by a tool that has no workspace to save into")
	}

	// And if one sends it anyway, the answer says so. Quietly returning the body
	// as though it had been saved would break the agent's next step, which is a
	// read of a file that was never written.
	out, err := tool.Run(context.Background(), `{"url":"`+srv.URL+`","save_as":"x.txt"}`)
	if err != nil {
		t.Fatalf("web_fetch Run: %v", err)
	}
	if !strings.Contains(out, "save_as FAILED") {
		t.Fatalf("save_as with no workspace must say so, got %q", out)
	}
}

func newTestWorkspace(t *testing.T) *Workspace {
	t.Helper()
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	return ws
}

// hostOf pulls the bare host out of a test server URL for the allowlist.
func hostOf(t *testing.T, raw string) string {
	t.Helper()
	u, err := parseAbsoluteURL(raw)
	if err != nil {
		t.Fatalf("parse %q: %v", raw, err)
	}
	return u.Hostname()
}

// A body that stopped at the read limit is a prefix of the document, and the
// receipt is the only place the agent can learn that. Without it, "saved 262144
// bytes" reads as the whole file — the agent parses it and reports a count that
// is confidently short.
func TestSaveReceiptAdmitsTruncation(t *testing.T) {
	sink := responseSink{fs: newWorkspaceFS(nil, newTestWorkspace(t))}

	full, err := sink.save(context.Background(), ToolWebFetch, "data/report.csv", []byte("a,b\n1,2\n"), false)
	if err != nil {
		t.Fatalf("save: %v", err)
	}
	if strings.Contains(full, "TRUNCATED") {
		t.Errorf("a complete body was reported as truncated: %s", full)
	}

	cut, err := sink.save(context.Background(), ToolWebFetch, "data/big.csv", []byte("a,b\n1,2\n"), true)
	if err != nil {
		t.Fatalf("save: %v", err)
	}
	if !strings.Contains(cut, "TRUNCATED") {
		t.Errorf("a body cut off at the read limit was reported as complete: %s", cut)
	}
}
