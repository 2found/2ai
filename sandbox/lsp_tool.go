package sandbox

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
)

const maxLSPResults = 200

var lspToolSequence atomic.Uint64

// LSPTool provides bounded read-only language intelligence. A process-local
// registry retains initialized servers by conversation and complete server
// identity; every action still synchronizes fresh workspace bytes.
type LSPTool struct {
	processSB  agentcore.ProcessSandbox
	workspace  *Workspace
	fs         workspaceFS
	config     LSPConfig
	hosted     bool
	serverRoot string
	registry   *LSPSessionRegistry
	namespace  string
	instance   string
}

func NewLSPTool(sb agentcore.Sandbox, workspace *Workspace, config LSPConfig) (*LSPTool, error) {
	return NewLSPToolWithRegistry(sb, workspace, nil, "", config)
}

func NewLSPToolWithRegistry(sb agentcore.Sandbox, workspace *Workspace, registry *LSPSessionRegistry, namespace string, config LSPConfig) (*LSPTool, error) {
	if workspace == nil {
		return nil, fmt.Errorf("lsp requires an agent workspace")
	}
	if err := config.validate(); err != nil {
		return nil, err
	}
	hosted := sb == nil
	if _, ok := sb.(*HostSandbox); ok {
		hosted = true
	}
	var fs workspaceFS
	if hosted {
		// An explicitly injected HostSandbox is still host mode. Besides keeping
		// LSP file URIs on the real workspace path, hostFS retains its symlink
		// escape checks instead of routing reads through shell helpers.
		fs = hostFS{ws: workspace}
	} else {
		fs = newWorkspaceFS(sb, workspace)
	}
	if sb == nil {
		sb = NewHostSandbox()
	}
	processSB, ok := sb.(agentcore.ProcessSandbox)
	if !ok {
		return nil, fmt.Errorf("lsp requires a sandbox backend with interactive process support")
	}
	if registry == nil {
		registry = NewLSPSessionRegistry(0, 0)
	}
	root := shellWorkdir
	if hosted {
		root = workspace.Root()
	}
	return &LSPTool{
		processSB: processSB, workspace: workspace, fs: fs,
		config: config, hosted: hosted, serverRoot: root, registry: registry,
		namespace: strings.TrimSpace(namespace), instance: fmt.Sprintf("lsp-%d", lspToolSequence.Add(1)),
	}, nil
}

func (t *LSPTool) Name() string { return ToolLSP }

func (t *LSPTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolLSP,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Query a configured Language Server Protocol server for read-only, symbol-aware code intelligence. " +
			"Actions: status; diagnostics for a file; document_symbols; or hover/definition/references at a symbol. " +
			"For position actions pass a 1-based line and the exact symbol substring on that line. If it occurs " +
			"more than once, suffix it with #N (for example value#2). Results are bounded and paths cannot escape " +
			"the workspace. Language-server binaries and optional sandbox images are provisioned by the operator.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{"status", "diagnostics", "document_symbols", "hover", "definition", "references"},
				},
				"file":   map[string]any{"type": "string", "description": "Workspace-relative source file. Required except for status."},
				"line":   map[string]any{"type": "integer", "description": "1-based source line for hover/definition/references."},
				"symbol": map[string]any{"type": "string", "description": "Exact substring on line; append #N to select the Nth occurrence."},
			},
			"required": []string{"action"},
		},
	}
}

func (t *LSPTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Action string `json:"action"`
		File   string `json:"file"`
		Line   int    `json:"line"`
		Symbol string `json:"symbol"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("lsp: invalid arguments: %w", err)
	}
	in.Action = strings.ToLower(strings.TrimSpace(in.Action))
	if in.Action == "status" {
		return fmt.Sprintf("configured_servers: %s\nstatus: configured, binaries not probed until use\nmode: read-only\nprocess: conversation-scoped pool (capacity %d, idle %s)", strings.Join(t.config.serverNames(), ", "), t.registry.capacity, t.registry.idle), nil
	}
	switch in.Action {
	case "diagnostics", "document_symbols", "hover", "definition", "references":
	default:
		return "", fmt.Errorf("lsp: unsupported action %q", in.Action)
	}
	if strings.TrimSpace(in.File) == "" {
		return "", fmt.Errorf("lsp: %s requires file", in.Action)
	}
	abs, rel, err := t.workspace.Resolve(in.File)
	if err != nil {
		return "", fmt.Errorf("lsp: %w", err)
	}
	info, err := t.fs.Stat(ctx, rel)
	if err != nil {
		return "", fmt.Errorf("lsp: %w", err)
	}
	if info.IsDir {
		return "", fmt.Errorf("lsp: %s is a directory", rel)
	}
	data, err := t.fs.ReadFile(ctx, rel)
	if err != nil {
		return "", fmt.Errorf("lsp: %w", err)
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("lsp: %s is not valid UTF-8", rel)
	}

	position := lspPosition{}
	if in.Action == "hover" || in.Action == "definition" || in.Action == "references" {
		position, err = resolveLSPSymbolPosition(string(data), in.Line, in.Symbol)
		if err != nil {
			return "", fmt.Errorf("lsp: %w", err)
		}
	}
	navigation := in.Action != "diagnostics"
	servers := t.config.matchingServers(rel, navigation)
	if len(servers) == 0 {
		return "", fmt.Errorf("lsp: no configured %s server handles %s", map[bool]string{true: "navigation", false: "diagnostic"}[navigation], rel)
	}
	if navigation {
		servers = servers[:1]
	}

	processPath := abs
	if !t.hosted {
		processPath = filepath.Join(t.serverRoot, filepath.FromSlash(rel))
	}
	outputs := make([]string, 0, len(servers))
	var failures []string
	for _, server := range servers {
		out, err := t.runServer(ctx, server, lspActionRequest{
			action: in.Action, rel: rel, processPath: processPath, content: string(data), position: position,
		})
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", server.Name, err))
			continue
		}
		if len(servers) > 1 {
			out = "server: " + server.Name + "\n" + out
		}
		outputs = append(outputs, out)
	}
	if len(outputs) == 0 {
		return "", fmt.Errorf("lsp: all matching servers failed: %s", strings.Join(failures, "; "))
	}
	if len(failures) > 0 {
		outputs = append(outputs, "warnings:\n- "+strings.Join(failures, "\n- "))
	}
	return strings.Join(outputs, "\n\n"), nil
}

type lspActionRequest struct {
	action      string
	rel         string
	processPath string
	content     string
	position    lspPosition
}

func (t *LSPTool) runServer(parent context.Context, server LSPServerConfig, req lspActionRequest) (string, error) {
	ctx, cancel := context.WithTimeout(parent, time.Duration(t.config.TimeoutSeconds)*time.Second)
	defer cancel()
	key := t.sessionKey(ctx, server)
	client, err := t.registry.acquire(key, func() (*lspClient, error) { return t.startClient(ctx, server) })
	if err != nil {
		return "", err
	}
	released := false
	release := func() {
		if !released {
			t.registry.release(key, client)
			released = true
		}
	}
	defer release()

	uri := pathToFileURI(req.processPath)
	output, err := client.execute(ctx, t, server, uri, req)
	if err != nil {
		// A cancelled request may leave a late JSON-RPC response in the stream;
		// any protocol error leaves similar uncertainty. Forget the client and do
		// not replay an action that might already have completed server-side.
		t.registry.invalidate(key, client)
		released = true
		return "", client.errorWithStderr(err)
	}
	return output, nil
}

func (t *LSPTool) sessionKey(ctx context.Context, server LSPServerConfig) string {
	session := strings.TrimSpace(agentcore.SandboxSessionFrom(ctx))
	if session == "" {
		session = t.instance
	}
	namespace := t.namespace
	if namespace == "" {
		namespace = t.instance
	}
	encoded, _ := json.Marshal(server)
	sum := sha256.Sum256(encoded)
	mode := "sandbox"
	if t.hosted {
		mode = "host"
	}
	return strings.Join([]string{namespace, session, t.workspace.Root(), mode, hex.EncodeToString(sum[:8])}, "\x00")
}

func (t *LSPTool) startClient(ctx context.Context, server LSPServerConfig) (*lspClient, error) {

	env := map[string]string{"HOME": sandboxWorkdir, "TMPDIR": sandboxWorkdir}
	cleanupHome := func() {}
	if t.hosted {
		home, err := os.MkdirTemp("", "agentray-lsp-home-")
		if err != nil {
			return nil, fmt.Errorf("create temporary server home: %w", err)
		}
		cleanupHome = func() { _ = os.RemoveAll(home) }
		env["HOME"], env["TMPDIR"] = home, home
		for _, key := range []string{"PATH", "LANG", "LC_ALL", "GOROOT", "GOPATH", "GOMODCACHE", "VIRTUAL_ENV"} {
			if value := os.Getenv(key); value != "" {
				env[key] = value
			}
		}
	}
	proc, err := t.processSB.Start(context.Background(), agentcore.SandboxExec{
		Argv: append([]string{server.Command}, server.Args...),
		Env:  env,
		Mounts: []agentcore.SandboxMount{{
			Source: t.workspace.Root(), Target: shellWorkdir, ReadOnly: true,
		}},
		Workdir: shellWorkdir,
		Image:   server.Image,
		Constraints: agentcore.SandboxLimits{
			MemoryMB: 512, CPUs: 1, PidsLimit: 128, TimeoutSeconds: lspProcessLifetime.Seconds(),
		},
	})
	if err != nil {
		cleanupHome()
		return nil, fmt.Errorf("start %s: %w", server.Command, err)
	}
	client := &lspClient{process: proc, cleanup: cleanupHome, stderrDone: make(chan struct{})}
	client.stderr.limit = 8 * 1024
	go func() {
		_, _ = io.Copy(&client.stderr, proc.Stderr())
		close(client.stderrDone)
	}()
	rootURI := pathToFileURI(t.serverRoot)
	client.conn = newLSPConn(proc.Stdin(), proc.Stdout(), server.Settings, lspWorkspaceFolder{
		URI: rootURI, Name: filepath.Base(t.serverRoot),
	})
	ready := false
	defer func() {
		if !ready {
			client.close()
		}
	}()

	initRaw, err := client.conn.request(ctx, "initialize", map[string]any{
		"processId": nil, "rootUri": rootURI,
		"workspaceFolders": []map[string]any{{"uri": rootURI, "name": filepath.Base(t.serverRoot)}},
		"capabilities": map[string]any{
			"textDocument": map[string]any{
				"publishDiagnostics": map[string]any{"relatedInformation": true},
				"diagnostic":         map[string]any{"dynamicRegistration": true},
				"definition":         map[string]any{"linkSupport": true},
			},
			"workspace": map[string]any{"configuration": true, "workspaceFolders": true},
		},
		"initializationOptions": server.InitializationOptions,
	})
	if err != nil {
		return nil, client.errorWithStderr(err)
	}
	var initialized struct {
		Capabilities struct {
			DiagnosticProvider json.RawMessage `json:"diagnosticProvider"`
		} `json:"capabilities"`
	}
	if err := json.Unmarshal(initRaw, &initialized); err != nil {
		return nil, fmt.Errorf("decode initialize result: %w", err)
	}
	if err := client.conn.notifyContext(ctx, "initialized", map[string]any{}); err != nil {
		return nil, err
	}
	if server.Settings != nil {
		if err := client.conn.notifyContext(ctx, "workspace/didChangeConfiguration", map[string]any{"settings": server.Settings}); err != nil {
			return nil, err
		}
	}
	client.diagnosticProvider = initialized.Capabilities.DiagnosticProvider
	ready = true
	return client, nil
}

func (t *LSPTool) executeLSPAction(ctx context.Context, conn *lspConn, diagnosticProvider json.RawMessage, uri string, documentVersion int, req lspActionRequest) (string, error) {
	doc := map[string]any{"uri": uri}
	switch req.action {
	case "diagnostics":
		var diagnostics []lspDiagnostic
		known := false
		pullAttempted := false
		var pullErr error
		waitCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		for !known {
			pullSupported := rawCapabilityEnabled(diagnosticProvider) || conn.supportsDynamicCapability("textDocument/diagnostic")
			if pullSupported && !pullAttempted {
				pullAttempted = true
				raw, err := conn.request(waitCtx, "textDocument/diagnostic", map[string]any{"textDocument": doc})
				if err != nil {
					pullErr = err
				} else {
					diagnostics, known, pullErr = decodeLSPPullDiagnostics(raw)
				}
				if known {
					break
				}
			}
			watchCapability := ""
			if !pullAttempted {
				watchCapability = "textDocument/diagnostic"
			}
			var capabilityAvailable bool
			var err error
			diagnostics, known, capabilityAvailable, err = conn.waitForDiagnosticsOrCapability(
				waitCtx, uri, documentVersion, watchCapability,
			)
			if err != nil {
				return "", err
			}
			if capabilityAvailable {
				continue
			}
			break
		}
		if !known {
			if pullErr != nil {
				return "", pullErr
			}
			return "diagnostics: unknown (server published no result within 3s; do not assume the file is clean)", nil
		}
		return formatLSPDiagnostics(req.rel, diagnostics), nil
	case "document_symbols":
		raw, err := conn.request(ctx, "textDocument/documentSymbol", map[string]any{"textDocument": doc})
		if err != nil {
			return "", err
		}
		return formatLSPSymbols(raw), nil
	case "hover":
		raw, err := conn.request(ctx, "textDocument/hover", map[string]any{"textDocument": doc, "position": req.position})
		if err != nil {
			return "", err
		}
		return formatLSPHover(raw), nil
	case "definition", "references":
		method := "textDocument/definition"
		params := map[string]any{"textDocument": doc, "position": req.position}
		if req.action == "references" {
			method = "textDocument/references"
			params["context"] = map[string]any{"includeDeclaration": true}
		}
		raw, err := conn.request(ctx, method, params)
		if err != nil {
			return "", err
		}
		locations, err := decodeLSPLocations(raw)
		if err != nil {
			return "", err
		}
		return t.formatLSPLocations(ctx, locations), nil
	default:
		return "", fmt.Errorf("unsupported action %s", req.action)
	}
}

func decodeLSPPullDiagnostics(raw json.RawMessage) ([]lspDiagnostic, bool, error) {
	var report struct {
		Kind  string          `json:"kind"`
		Items json.RawMessage `json:"items"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return nil, false, fmt.Errorf("decode pull diagnostics: %w", err)
	}
	items := bytes.TrimSpace(report.Items)
	if report.Kind != "full" || len(items) == 0 || items[0] != '[' {
		return nil, false, nil
	}
	var diagnostics []lspDiagnostic
	if err := json.Unmarshal(items, &diagnostics); err != nil {
		return nil, false, fmt.Errorf("decode pull diagnostic items: %w", err)
	}
	return diagnostics, true, nil
}

func rawCapabilityEnabled(raw json.RawMessage) bool {
	s := strings.TrimSpace(string(raw))
	return s != "" && s != "null" && s != "false"
}

func resolveLSPSymbolPosition(content string, line int, rawSymbol string) (lspPosition, error) {
	if line < 1 {
		return lspPosition{}, fmt.Errorf("line must be at least 1")
	}
	symbol, occurrence, explicit := parseSymbolOccurrence(strings.TrimSpace(rawSymbol))
	if symbol == "" {
		return lspPosition{}, fmt.Errorf("symbol is required")
	}
	lines := strings.Split(normalizeToLF(content), "\n")
	if line > len(lines) {
		return lspPosition{}, fmt.Errorf("line %d is beyond end of file (%d lines)", line, len(lines))
	}
	sourceLine := lines[line-1]
	var offsets []int
	for searchFrom := 0; searchFrom <= len(sourceLine)-len(symbol); {
		i := strings.Index(sourceLine[searchFrom:], symbol)
		if i < 0 {
			break
		}
		at := searchFrom + i
		offsets = append(offsets, at)
		searchFrom = at + len(symbol)
	}
	if len(offsets) == 0 {
		return lspPosition{}, fmt.Errorf("symbol %q not found on line %d", symbol, line)
	}
	if !explicit && len(offsets) > 1 {
		return lspPosition{}, fmt.Errorf("symbol %q occurs %d times on line %d; use %s#N", symbol, len(offsets), line, symbol)
	}
	if occurrence < 1 || occurrence > len(offsets) {
		return lspPosition{}, fmt.Errorf("symbol %q has %d occurrences on line %d, not #%d", symbol, len(offsets), line, occurrence)
	}
	prefix := sourceLine[:offsets[occurrence-1]]
	return lspPosition{Line: line - 1, Character: len(utf16.Encode([]rune(prefix)))}, nil
}

func parseSymbolOccurrence(raw string) (string, int, bool) {
	if before, after, ok := strings.Cut(raw, "#"); ok && before != "" && after != "" && !strings.Contains(after, "#") {
		if n, err := strconv.Atoi(after); err == nil && n > 0 {
			return before, n, true
		}
	}
	return raw, 1, false
}

func pathToFileURI(path string) string {
	return pathToFileURIForOS(path, runtime.GOOS)
}

func pathToFileURIForOS(path, goos string) string {
	slashPath := filepath.ToSlash(path)
	if goos == "windows" {
		slashPath = strings.ReplaceAll(path, `\`, "/")
	}
	u := &url.URL{Scheme: "file"}
	if goos == "windows" && strings.HasPrefix(slashPath, "//") {
		unc := strings.TrimPrefix(slashPath, "//")
		if host, rest, ok := strings.Cut(unc, "/"); ok && host != "" {
			u.Host = host
			u.Path = "/" + rest
			return u.String()
		}
	}
	if goos == "windows" && isWindowsDrivePath(slashPath) {
		slashPath = "/" + slashPath
	}
	u.Path = slashPath
	return u.String()
}

func isWindowsDrivePath(path string) bool {
	return len(path) >= 2 && path[1] == ':' && ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z'))
}

func fileURIPathForOS(u *url.URL, goos string) (string, bool) {
	if u == nil || u.Scheme != "file" || u.User != nil || u.Port() != "" {
		return "", false
	}
	host := u.Hostname()
	path := u.Path
	if goos == "windows" {
		if host != "" && !strings.EqualFold(host, "localhost") {
			path = "//" + host + "/" + strings.TrimPrefix(path, "/")
		} else if strings.HasPrefix(path, "/") && isWindowsDrivePath(path[1:]) {
			path = path[1:]
		}
		return strings.ReplaceAll(path, "/", `\`), path != ""
	}
	if host != "" && !strings.EqualFold(host, "localhost") {
		return "", false
	}
	return filepath.FromSlash(path), path != ""
}

func formatLSPDiagnostics(rel string, diagnostics []lspDiagnostic) string {
	if len(diagnostics) == 0 {
		return "OK: no diagnostics"
	}
	sort.Slice(diagnostics, func(i, j int) bool {
		a, b := diagnostics[i].Range.Start, diagnostics[j].Range.Start
		return a.Line < b.Line || (a.Line == b.Line && a.Character < b.Character)
	})
	if len(diagnostics) > maxLSPResults {
		diagnostics = diagnostics[:maxLSPResults]
	}
	lines := make([]string, 0, len(diagnostics)+1)
	for _, d := range diagnostics {
		severity := map[int]string{1: "error", 2: "warning", 3: "info", 4: "hint"}[d.Severity]
		if severity == "" {
			severity = "diagnostic"
		}
		originParts := make([]string, 0, 2)
		if d.Source != "" {
			originParts = append(originParts, d.Source)
		}
		if d.Code != nil {
			originParts = append(originParts, fmt.Sprint(d.Code))
		}
		origin := strings.Join(originParts, "/")
		if origin != "" {
			origin = " [" + origin + "]"
		}
		lines = append(lines, fmt.Sprintf("%s:%d:%d: %s%s: %s", rel, d.Range.Start.Line+1, d.Range.Start.Character+1, severity, origin, strings.TrimSpace(d.Message)))
	}
	if len(lines) == maxLSPResults {
		lines = append(lines, fmt.Sprintf("[results capped at %d]", maxLSPResults))
	}
	return strings.Join(lines, "\n")
}

func decodeLSPLocations(raw json.RawMessage) ([]lspLocation, error) {
	if string(bytesTrimSpace(raw)) == "null" || len(bytesTrimSpace(raw)) == 0 {
		return nil, nil
	}
	var items []json.RawMessage
	if len(bytesTrimSpace(raw)) > 0 && bytesTrimSpace(raw)[0] == '[' {
		if err := json.Unmarshal(raw, &items); err != nil {
			return nil, fmt.Errorf("decode locations: %w", err)
		}
	} else {
		items = []json.RawMessage{raw}
	}
	locations := make([]lspLocation, 0, len(items))
	for _, item := range items {
		var probe struct {
			URI       string `json:"uri"`
			TargetURI string `json:"targetUri"`
		}
		if err := json.Unmarshal(item, &probe); err != nil {
			return nil, fmt.Errorf("decode location: %w", err)
		}
		if probe.TargetURI != "" {
			var link lspLocationLink
			if err := json.Unmarshal(item, &link); err != nil {
				return nil, err
			}
			locations = append(locations, lspLocation{URI: link.TargetURI, Range: link.TargetSelectionRange})
			continue
		}
		var location lspLocation
		if err := json.Unmarshal(item, &location); err != nil {
			return nil, err
		}
		locations = append(locations, location)
	}
	return locations, nil
}

func bytesTrimSpace(raw []byte) []byte {
	return []byte(strings.TrimSpace(string(raw)))
}

func (t *LSPTool) formatLSPLocations(ctx context.Context, locations []lspLocation) string {
	if len(locations) == 0 {
		return "No locations found"
	}
	seen := map[string]bool{}
	lines := make([]string, 0, len(locations))
	for _, location := range locations {
		rel, ok := t.relativeServerURI(location.URI)
		if !ok {
			continue
		}
		key := fmt.Sprintf("%s:%d:%d", rel, location.Range.Start.Line, location.Range.Start.Character)
		if seen[key] {
			continue
		}
		seen[key] = true
		contextLine := ""
		if data, err := t.fs.ReadFile(ctx, rel); err == nil {
			fileLines := strings.Split(normalizeToLF(string(data)), "\n")
			if location.Range.Start.Line >= 0 && location.Range.Start.Line < len(fileLines) {
				contextLine = ": " + strings.TrimSpace(fileLines[location.Range.Start.Line])
			}
		}
		lines = append(lines, fmt.Sprintf("%s:%d:%d%s", rel, location.Range.Start.Line+1, location.Range.Start.Character+1, contextLine))
		if len(lines) == maxLSPResults {
			lines = append(lines, fmt.Sprintf("[results capped at %d]", maxLSPResults))
			break
		}
	}
	if len(lines) == 0 {
		return "No workspace locations found"
	}
	return strings.Join(lines, "\n")
}

func (t *LSPTool) relativeServerURI(raw string) (string, bool) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "file" {
		return "", false
	}
	path, ok := fileURIPathForOS(u, runtime.GOOS)
	if !ok {
		return "", false
	}
	rel, err := filepath.Rel(t.serverRoot, path)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", false
	}
	_, guarded, err := t.workspace.Resolve(filepath.ToSlash(rel))
	if err != nil {
		return "", false
	}
	return guarded, true
}

func formatLSPHover(raw json.RawMessage) string {
	if string(bytesTrimSpace(raw)) == "null" {
		return "No hover information"
	}
	var hover struct {
		Contents any `json:"contents"`
	}
	if err := json.Unmarshal(raw, &hover); err != nil {
		return "Invalid hover response"
	}
	text := strings.TrimSpace(flattenLSPMarkup(hover.Contents))
	if text == "" {
		return "No hover information"
	}
	return text
}

func flattenLSPMarkup(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			if text := strings.TrimSpace(flattenLSPMarkup(item)); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n\n")
	case map[string]any:
		if text, ok := v["value"].(string); ok {
			return text
		}
	}
	return ""
}

func formatLSPSymbols(raw json.RawMessage) string {
	if string(bytesTrimSpace(raw)) == "null" {
		return "No symbols found"
	}
	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		return "Invalid document-symbol response"
	}
	var lines []string
	var walk func([]map[string]any, int)
	walk = func(symbols []map[string]any, depth int) {
		for _, symbol := range symbols {
			if len(lines) >= maxLSPResults {
				return
			}
			name, _ := symbol["name"].(string)
			kind := intFromJSON(symbol["kind"])
			line := symbolLine(symbol)
			lines = append(lines, fmt.Sprintf("%s%s (kind %d, line %d)", strings.Repeat("  ", depth), name, kind, line))
			if children, ok := symbol["children"].([]any); ok {
				mapped := make([]map[string]any, 0, len(children))
				for _, child := range children {
					if m, ok := child.(map[string]any); ok {
						mapped = append(mapped, m)
					}
				}
				walk(mapped, depth+1)
			}
		}
	}
	walk(items, 0)
	if len(lines) == 0 {
		return "No symbols found"
	}
	if len(lines) == maxLSPResults {
		lines = append(lines, fmt.Sprintf("[results capped at %d]", maxLSPResults))
	}
	return strings.Join(lines, "\n")
}

func intFromJSON(value any) int {
	if n, ok := value.(float64); ok {
		return int(n)
	}
	return 0
}

func symbolLine(symbol map[string]any) int {
	rangeValue, _ := symbol["selectionRange"].(map[string]any)
	if rangeValue == nil {
		rangeValue, _ = symbol["range"].(map[string]any)
	}
	if rangeValue == nil {
		if location, ok := symbol["location"].(map[string]any); ok {
			rangeValue, _ = location["range"].(map[string]any)
		}
	}
	start, _ := rangeValue["start"].(map[string]any)
	return intFromJSON(start["line"]) + 1
}

func withLSPStderr(err error, stderr string) error {
	stderr = strings.TrimSpace(stderr)
	if stderr == "" {
		return err
	}
	return fmt.Errorf("%w; server stderr: %s", err, stderr)
}

type cappedStringWriter struct {
	mu    sync.Mutex
	b     strings.Builder
	limit int
}

func (w *cappedStringWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	n := len(p)
	remaining := w.limit - w.b.Len()
	if remaining > 0 {
		if len(p) > remaining {
			p = p[:remaining]
		}
		_, _ = w.b.Write(p)
	}
	return n, nil
}

func (w *cappedStringWriter) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.b.String()
}
