package sandbox

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

func TestLSPToolAgainstProtocolFixture(t *testing.T) {
	ws := mustWorkspace(t)
	content := "😀 const target = target;\n"
	mustWrite(t, ws, "main.ts", content)
	counter := filepath.Join(t.TempDir(), "initialize.log")
	registry := NewLSPSessionRegistry(4, time.Minute)
	t.Cleanup(registry.Close)
	cfg := LSPConfig{
		TimeoutSeconds: 5,
		Servers: []LSPServerConfig{{
			Name: "fixture", Command: os.Args[0],
			Args:       []string{"-test.run=^TestLSPHelperProcess$", "--", "lsp-helper", counter},
			Extensions: []string{".ts"}, LanguageID: "typescript",
		}},
	}
	tool, err := NewLSPToolWithRegistry(nil, ws, registry, "tenant-project-agent", cfg)
	if err != nil {
		t.Fatalf("NewLSPTool: %v", err)
	}

	cases := []struct {
		name string
		args string
		want []string
	}{
		{"status", `{"action":"status"}`, []string{"configured_servers: fixture", "read-only"}},
		{"diagnostics", `{"action":"diagnostics","file":"main.ts"}`, []string{"main.ts:1:4", "warning", "fixture diagnostic"}},
		{"symbols", `{"action":"document_symbols","file":"main.ts"}`, []string{"target", "line 1"}},
		{"hover utf16", `{"action":"hover","file":"main.ts","line":1,"symbol":"target#2"}`, []string{"position=0:18"}},
		{"definition", `{"action":"definition","file":"main.ts","line":1,"symbol":"target#2"}`, []string{"main.ts:1:10", "const target"}},
		{"references", `{"action":"references","file":"main.ts","line":1,"symbol":"target#2"}`, []string{"main.ts:1:10", "const target"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := tool.Run(context.Background(), tc.args)
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			for _, want := range tc.want {
				if !strings.Contains(out, want) {
					t.Errorf("output %q missing %q", out, want)
				}
			}
		})
	}

	// The process survives action boundaries, but the document overlay does not:
	// a later action must observe the newly read workspace bytes.
	mustWrite(t, ws, "main.ts", "const fresh = fresh;\n")
	out, err := tool.Run(context.Background(), `{"action":"hover","file":"main.ts","line":1,"symbol":"fresh#2"}`)
	if err != nil {
		t.Fatalf("fresh hover: %v", err)
	}
	if !strings.Contains(out, "text=const fresh = fresh;") {
		t.Fatalf("fresh hover output = %q", out)
	}
	if got := countLines(t, counter); got != 1 {
		t.Fatalf("initialize count = %d, want one warm client", got)
	}
}

func TestLSPToolIsolatesConversationAndServerConfig(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "main.ts", "const target = target;\n")
	counter := filepath.Join(t.TempDir(), "initialize.log")
	registry := NewLSPSessionRegistry(4, time.Minute)
	t.Cleanup(registry.Close)
	server := LSPServerConfig{
		Name: "fixture", Command: os.Args[0],
		Args:       []string{"-test.run=^TestLSPHelperProcess$", "--", "lsp-helper", counter},
		Extensions: []string{".ts"}, LanguageID: "typescript",
	}
	tool, err := NewLSPToolWithRegistry(nil, ws, registry, "tenant-project-agent", LSPConfig{TimeoutSeconds: 5, Servers: []LSPServerConfig{server}})
	if err != nil {
		t.Fatal(err)
	}
	call := func(session string, tool *LSPTool) {
		t.Helper()
		ctx := agentcore.WithSandboxSession(context.Background(), session)
		if _, err := tool.Run(ctx, `{"action":"hover","file":"main.ts","line":1,"symbol":"target#2"}`); err != nil {
			t.Fatalf("session %s: %v", session, err)
		}
	}
	call("conversation-a", tool)
	call("conversation-a", tool)
	call("conversation-b", tool)

	changed := server
	changed.Settings = map[string]any{"fixture": map[string]any{"mode": "strict"}}
	changedTool, err := NewLSPToolWithRegistry(nil, ws, registry, "tenant-project-agent", LSPConfig{TimeoutSeconds: 5, Servers: []LSPServerConfig{changed}})
	if err != nil {
		t.Fatal(err)
	}
	call("conversation-a", changedTool)
	if got := countLines(t, counter); got != 3 {
		t.Fatalf("initialize count = %d, want conversation/config isolation", got)
	}
}

func TestLSPToolSerializesConcurrentCalls(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "main.ts", "const target = target;\n")
	registry := NewLSPSessionRegistry(2, time.Minute)
	t.Cleanup(registry.Close)
	cfg := LSPConfig{TimeoutSeconds: 5, Servers: []LSPServerConfig{{
		Name: "fixture", Command: os.Args[0],
		Args:       []string{"-test.run=^TestLSPHelperProcess$", "--", "lsp-helper"},
		Extensions: []string{".ts"}, LanguageID: "typescript",
	}}}
	tool, err := NewLSPToolWithRegistry(nil, ws, registry, "tenant-project-agent", cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx := agentcore.WithSandboxSession(context.Background(), "conversation")
	var wg sync.WaitGroup
	errs := make(chan error, 12)
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := tool.Run(ctx, `{"action":"hover","file":"main.ts","line":1,"symbol":"target#2"}`)
			if err == nil && !strings.Contains(out, "position=0:15") {
				err = fmt.Errorf("unexpected output %q", out)
			}
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestLSPToolCancellationDiscardsClientWithoutReplay(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "main.ts", "const SLOW = SLOW;\n")
	counter := filepath.Join(t.TempDir(), "initialize.log")
	registry := NewLSPSessionRegistry(2, time.Minute)
	t.Cleanup(registry.Close)
	cfg := LSPConfig{TimeoutSeconds: 5, Servers: []LSPServerConfig{{
		Name: "fixture", Command: os.Args[0],
		Args:       []string{"-test.run=^TestLSPHelperProcess$", "--", "lsp-helper", counter},
		Extensions: []string{".ts"}, LanguageID: "typescript",
	}}}
	tool, err := NewLSPToolWithRegistry(nil, ws, registry, "tenant-project-agent", cfg)
	if err != nil {
		t.Fatal(err)
	}
	base := agentcore.WithSandboxSession(context.Background(), "conversation")
	ctx, cancel := context.WithTimeout(base, 20*time.Millisecond)
	_, err = tool.Run(ctx, `{"action":"hover","file":"main.ts","line":1,"symbol":"SLOW#2"}`)
	cancel()
	if err == nil || !strings.Contains(err.Error(), "deadline exceeded") {
		t.Fatalf("timeout error = %v", err)
	}
	mustWrite(t, ws, "main.ts", "const target = target;\n")
	if _, err := tool.Run(base, `{"action":"hover","file":"main.ts","line":1,"symbol":"target#2"}`); err != nil {
		t.Fatalf("replacement client: %v", err)
	}
	if got := countLines(t, counter); got != 2 {
		t.Fatalf("initialize count = %d, want timed-out client replaced", got)
	}
}

func TestLSPToolTreatsInjectedHostSandboxAsHostMode(t *testing.T) {
	ws := mustWorkspace(t)
	cfg := LSPConfig{Servers: []LSPServerConfig{{
		Name: "fixture", Command: os.Args[0], Extensions: []string{".go"},
	}}}
	tool, err := NewLSPTool(NewHostSandbox(), ws, cfg)
	if err != nil {
		t.Fatalf("NewLSPTool: %v", err)
	}
	if !tool.hosted {
		t.Fatal("explicit HostSandbox must use host path semantics")
	}
	if tool.serverRoot != ws.Root() {
		t.Fatalf("serverRoot = %q, want %q", tool.serverRoot, ws.Root())
	}
	if _, ok := tool.fs.(hostFS); !ok {
		t.Fatalf("filesystem = %T, want hostFS", tool.fs)
	}
}

func TestLSPToolRejectsWorkspaceEscapeAndAmbiguousSymbol(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "main.ts", "target + target\n")
	cfg, err := ParseLSPConfig(fmt.Sprintf(`{"servers":[{"name":"fixture","command":%q,"args":["-test.run=^TestLSPHelperProcess$","--","lsp-helper"],"extensions":[".ts"]}]}`, os.Args[0]))
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	tool, err := NewLSPTool(nil, ws, cfg)
	if err != nil {
		t.Fatalf("tool: %v", err)
	}
	if _, err := tool.Run(context.Background(), `{"action":"hover","file":"../outside.ts","line":1,"symbol":"x"}`); err == nil || !strings.Contains(err.Error(), "escapes workspace") {
		t.Fatalf("escape error = %v", err)
	}
	if _, err := tool.Run(context.Background(), `{"action":"hover","file":"main.ts","line":1,"symbol":"target"}`); err == nil || !strings.Contains(err.Error(), "occurs 2") {
		t.Fatalf("ambiguity error = %v", err)
	}
}

func TestLSPFileURIConversionIsPortable(t *testing.T) {
	tests := []struct {
		name string
		path string
		goos string
		want string
	}{
		{name: "unix", path: "/workspace/My File.go", goos: "linux", want: "file:///workspace/My%20File.go"},
		{name: "windows drive", path: `C:\Users\dev\My File.go`, goos: "windows", want: "file:///C:/Users/dev/My%20File.go"},
		{name: "windows UNC", path: `\\server\share\main.go`, goos: "windows", want: "file://server/share/main.go"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := pathToFileURIForOS(tc.path, tc.goos); got != tc.want {
				t.Fatalf("URI = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestLSPFileURIPathValidatesAuthorityAndWindowsPaths(t *testing.T) {
	tests := []struct {
		raw  string
		goos string
		want string
		ok   bool
	}{
		{raw: "file:///workspace/a%20b.go", goos: "linux", want: "/workspace/a b.go", ok: true},
		{raw: "file://localhost/workspace/a.go", goos: "linux", want: "/workspace/a.go", ok: true},
		{raw: "file://server/share/a.go", goos: "linux", ok: false},
		{raw: "file:///C:/Users/dev/a.go", goos: "windows", want: `C:\Users\dev\a.go`, ok: true},
		{raw: "file://server/share/a.go", goos: "windows", want: `\\server\share\a.go`, ok: true},
		{raw: "file://user@server/share/a.go", goos: "windows", ok: false},
		{raw: "file://server:99/share/a.go", goos: "windows", ok: false},
		// A literal "%2F" filename is encoded as "%252F". URL.Parse decodes it
		// once; the path conversion must not decode it again into a separator.
		{raw: "file:///workspace/name%252Fpart.go", goos: "linux", want: "/workspace/name%2Fpart.go", ok: true},
	}
	for _, tc := range tests {
		t.Run(tc.raw+"/"+tc.goos, func(t *testing.T) {
			u, err := url.Parse(tc.raw)
			if err != nil {
				t.Fatal(err)
			}
			got, ok := fileURIPathForOS(u, tc.goos)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("path = %q, ok=%v; want %q, %v", got, ok, tc.want, tc.ok)
			}
		})
	}
}

func TestDecodeLSPPullDiagnosticsRequiresFullReport(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		known     bool
		wantCount int
	}{
		{name: "full empty is authoritative", raw: `{"kind":"full","items":[]}`, known: true},
		{name: "full items", raw: `{"kind":"full","items":[{"message":"problem","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}]}`, known: true, wantCount: 1},
		{name: "unchanged needs cached result", raw: `{"kind":"unchanged","resultId":"same"}`},
		{name: "missing kind is unknown", raw: `{"items":[]}`},
		{name: "non-array items is unknown", raw: `{"kind":"full","items":null}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			diagnostics, known, err := decodeLSPPullDiagnostics(json.RawMessage(tc.raw))
			if err != nil {
				t.Fatal(err)
			}
			if known != tc.known || len(diagnostics) != tc.wantCount {
				t.Fatalf("known=%v count=%d, want known=%v count=%d", known, len(diagnostics), tc.known, tc.wantCount)
			}
		})
	}
}

func TestLSPToolDoesNotTreatUnchangedPullReportAsClean(t *testing.T) {
	serverReads, clientWrites := io.Pipe()
	clientReads, serverWrites := io.Pipe()
	t.Cleanup(func() {
		_ = clientWrites.Close()
		_ = serverWrites.Close()
		_ = serverReads.Close()
		_ = clientReads.Close()
	})
	conn := newLSPConn(clientWrites, clientReads, nil)
	go func() {
		payload, err := readLSPFrame(bufio.NewReader(serverReads))
		if err != nil {
			return
		}
		var request lspWireMessage
		if json.Unmarshal(payload, &request) != nil {
			return
		}
		writeLSPFrame(serverWrites, map[string]any{
			"jsonrpc": "2.0", "id": json.RawMessage(request.ID),
			"result": map[string]any{"kind": "unchanged", "resultId": "prior"},
		})
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	tool := &LSPTool{}
	output, err := tool.executeLSPAction(
		ctx, conn, json.RawMessage(`true`), "file:///workspace/main.go", 1,
		lspActionRequest{action: "diagnostics", rel: "main.go"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(output, "OK: no diagnostics") || !strings.Contains(output, "unknown") {
		t.Fatalf("diagnostics output = %q", output)
	}
}

func TestLSPToolFallsBackToPublishedDiagnosticsAfterPullError(t *testing.T) {
	serverReads, clientWrites := io.Pipe()
	clientReads, serverWrites := io.Pipe()
	t.Cleanup(func() {
		_ = clientWrites.Close()
		_ = serverWrites.Close()
		_ = serverReads.Close()
		_ = clientReads.Close()
	})
	conn := newLSPConn(clientWrites, clientReads, nil)
	go func() {
		payload, err := readLSPFrame(bufio.NewReader(serverReads))
		if err != nil {
			return
		}
		var request lspWireMessage
		if json.Unmarshal(payload, &request) != nil {
			return
		}
		writeLSPFrame(serverWrites, map[string]any{
			"jsonrpc": "2.0", "id": json.RawMessage(request.ID),
			"error": map[string]any{"code": -32603, "message": "pull unavailable"},
		})
		writeLSPFrame(serverWrites, map[string]any{
			"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
			"params": map[string]any{
				"uri": "file:///workspace/main.go", "version": 1,
				"diagnostics": []any{map[string]any{
					"message": "published result", "range": rangeJSON(0, 0, 0, 1),
				}},
			},
		})
	}()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	output, err := (&LSPTool{}).executeLSPAction(
		ctx, conn, json.RawMessage(`true`), "file:///workspace/main.go", 1,
		lspActionRequest{action: "diagnostics", rel: "main.go"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output, "published result") {
		t.Fatalf("diagnostics output = %q", output)
	}
}

func TestLSPToolObservesLateDynamicDiagnosticRegistration(t *testing.T) {
	serverReads, clientWrites := io.Pipe()
	clientReads, serverWrites := io.Pipe()
	t.Cleanup(func() {
		_ = clientWrites.Close()
		_ = serverWrites.Close()
		_ = serverReads.Close()
		_ = clientReads.Close()
	})
	conn := newLSPConn(clientWrites, clientReads, nil)
	type result struct {
		output string
		err    error
	}
	resultC := make(chan result, 1)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	go func() {
		output, err := (&LSPTool{}).executeLSPAction(
			ctx, conn, nil, "file:///workspace/main.go", 1,
			lspActionRequest{action: "diagnostics", rel: "main.go"},
		)
		resultC <- result{output: output, err: err}
	}()

	writeLSPFrame(serverWrites, map[string]any{
		"jsonrpc": "2.0", "id": 99, "method": "client/registerCapability",
		"params": map[string]any{"registrations": []any{
			map[string]any{"id": "diagnostics", "method": "textDocument/diagnostic"},
		}},
	})
	serverReader := bufio.NewReader(serverReads)
	if _, err := readLSPFrame(serverReader); err != nil { // registration response
		t.Fatal(err)
	}
	payload, err := readLSPFrame(serverReader)
	if err != nil {
		t.Fatal(err)
	}
	var request lspWireMessage
	if err := json.Unmarshal(payload, &request); err != nil {
		t.Fatal(err)
	}
	if request.Method != "textDocument/diagnostic" {
		t.Fatalf("request method = %q", request.Method)
	}
	writeLSPFrame(serverWrites, map[string]any{
		"jsonrpc": "2.0", "id": json.RawMessage(request.ID),
		"result": map[string]any{"kind": "full", "items": []any{map[string]any{
			"message": "dynamic result", "range": rangeJSON(0, 0, 0, 1),
		}}},
	})
	got := <-resultC
	if got.err != nil || !strings.Contains(got.output, "dynamic result") {
		t.Fatalf("dynamic diagnostics output=%q err=%v", got.output, got.err)
	}
}

// TestLSPHelperProcess doubles as a deterministic language server. Child test
// binaries run only this function; ordinary test runs return immediately.
func TestLSPHelperProcess(t *testing.T) {
	if !containsArg(os.Args, "lsp-helper") {
		return
	}
	r := bufio.NewReader(os.Stdin)
	var openedText string
	var openedVersion int
	for {
		payload, err := readLSPFrame(r)
		if err != nil {
			return
		}
		var msg lspWireMessage
		if err := json.Unmarshal(payload, &msg); err != nil {
			return
		}
		switch msg.Method {
		case "initialize":
			if counter := helperArgAfter("lsp-helper"); counter != "" {
				f, err := os.OpenFile(counter, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
				if err == nil {
					_, _ = io.WriteString(f, "initialize\n")
					_ = f.Close()
				}
			}
			writeHelperLSP(map[string]any{
				"jsonrpc": "2.0", "id": json.RawMessage(msg.ID),
				"result": map[string]any{"capabilities": map[string]any{
					"diagnosticProvider": true, "hoverProvider": true, "definitionProvider": true,
					"referencesProvider": true, "documentSymbolProvider": true,
				}},
			})
		case "initialized", "textDocument/didClose":
		case "textDocument/didOpen":
			var params struct {
				TextDocument struct {
					URI     string `json:"uri"`
					Text    string `json:"text"`
					Version int    `json:"version"`
				} `json:"textDocument"`
			}
			_ = json.Unmarshal(msg.Params, &params)
			openedText = strings.TrimSpace(params.TextDocument.Text)
			openedVersion = params.TextDocument.Version
			writeHelperLSP(map[string]any{
				"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
				"params": map[string]any{"uri": params.TextDocument.URI, "version": openedVersion, "diagnostics": []any{
					map[string]any{
						"range": rangeJSON(0, 3, 0, 9), "severity": 2,
						"source": "fixture", "code": "W1", "message": "fixture diagnostic: " + openedText,
					},
				}},
			})
		case "textDocument/diagnostic":
			writeHelperLSP(map[string]any{
				"jsonrpc": "2.0", "id": json.RawMessage(msg.ID),
				"result": map[string]any{"kind": "full", "items": []any{
					map[string]any{
						"range": rangeJSON(0, 3, 0, 9), "severity": 2,
						"source": "fixture", "code": "W1", "message": "fixture diagnostic: " + openedText,
					},
				}},
			})
		case "textDocument/hover":
			var params struct {
				Position lspPosition `json:"position"`
			}
			_ = json.Unmarshal(msg.Params, &params)
			if strings.Contains(openedText, "SLOW") {
				time.Sleep(300 * time.Millisecond)
			}
			writeHelperLSP(map[string]any{
				"jsonrpc": "2.0", "id": json.RawMessage(msg.ID),
				"result": map[string]any{"contents": map[string]any{"kind": "plaintext", "value": fmt.Sprintf("position=%d:%d text=%s", params.Position.Line, params.Position.Character, openedText)}},
			})
		case "textDocument/definition", "textDocument/references":
			var params struct {
				TextDocument struct {
					URI string `json:"uri"`
				} `json:"textDocument"`
			}
			_ = json.Unmarshal(msg.Params, &params)
			location := map[string]any{"uri": params.TextDocument.URI, "range": rangeJSON(0, 9, 0, 15)}
			var result any = location
			if msg.Method == "textDocument/references" {
				result = []any{location, location}
			}
			writeHelperLSP(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID), "result": result})
		case "textDocument/documentSymbol":
			writeHelperLSP(map[string]any{
				"jsonrpc": "2.0", "id": json.RawMessage(msg.ID),
				"result": []any{map[string]any{
					"name": "target", "kind": 13, "range": rangeJSON(0, 9, 0, 15), "selectionRange": rangeJSON(0, 9, 0, 15),
				}},
			})
		case "shutdown":
			writeHelperLSP(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID), "result": nil})
		case "exit":
			return
		}
	}
}

func helperArgAfter(marker string) string {
	for i, arg := range os.Args {
		if arg == marker && i+1 < len(os.Args) {
			return os.Args[i+1]
		}
	}
	return ""
}

func countLines(t *testing.T, path string) int {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read counter: %v", err)
	}
	return strings.Count(string(data), "\n")
}

func writeHelperLSP(value any) {
	payload, _ := json.Marshal(value)
	_, _ = fmt.Fprintf(os.Stdout, "Content-Length: %d\r\n\r\n", len(payload))
	_, _ = os.Stdout.Write(payload)
}

func rangeJSON(sl, sc, el, ec int) map[string]any {
	return map[string]any{
		"start": map[string]any{"line": sl, "character": sc},
		"end":   map[string]any{"line": el, "character": ec},
	}
}

func containsArg(args []string, wanted string) bool {
	for _, arg := range args {
		if arg == wanted {
			return true
		}
	}
	return false
}
