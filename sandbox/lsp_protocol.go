package sandbox

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"time"
)

const maxLSPMessageBytes = 8 * 1024 * 1024

const lspWriteTimeout = time.Second

// Servers that omit diagnostic document versions can publish more than once
// for a single edit. Waiting briefly for the stream to go quiet avoids
// accepting an in-flight publication for the previous document contents.
const lspDiagnosticSettleDelay = 50 * time.Millisecond

type lspPosition struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

type lspRange struct {
	Start lspPosition `json:"start"`
	End   lspPosition `json:"end"`
}

type lspDiagnostic struct {
	Range    lspRange       `json:"range"`
	Severity int            `json:"severity,omitempty"`
	Code     any            `json:"code,omitempty"`
	Source   string         `json:"source,omitempty"`
	Message  string         `json:"message"`
	Tags     []int          `json:"tags,omitempty"`
	Data     map[string]any `json:"data,omitempty"`
}

type lspLocation struct {
	URI   string   `json:"uri"`
	Range lspRange `json:"range"`
}

type lspLocationLink struct {
	TargetURI            string   `json:"targetUri"`
	TargetRange          lspRange `json:"targetRange"`
	TargetSelectionRange lspRange `json:"targetSelectionRange"`
}

type lspResponseError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

type lspWireMessage struct {
	JSONRPC string            `json:"jsonrpc,omitempty"`
	ID      json.RawMessage   `json:"id,omitempty"`
	Method  string            `json:"method,omitempty"`
	Params  json.RawMessage   `json:"params,omitempty"`
	Result  json.RawMessage   `json:"result,omitempty"`
	Error   *lspResponseError `json:"error,omitempty"`
}

type lspConn struct {
	in  io.Writer
	out *bufio.Reader

	initOnce sync.Once
	writes   chan lspWriteRequest
	stateMu  sync.Mutex
	nextID   int
	pending  map[string]chan lspWireMessage
	terminal error
	closed   chan struct{}

	workspaceFolders    []lspWorkspaceFolder
	dynamicCapabilities map[string]string
	capabilityEvents    chan struct{}

	diagnosticsMu    sync.Mutex
	diagnostics      map[string]lspDiagnosticPublication
	diagnosticEvents chan struct{}
	settings         map[string]any
}

type lspWriteRequest struct {
	ctx     context.Context
	payload []byte
	result  chan error
}

type lspWorkspaceFolder struct {
	URI  string `json:"uri"`
	Name string `json:"name"`
}

type lspDiagnosticPublication struct {
	items      []lspDiagnostic
	version    *int
	receivedAt time.Time
}

func newLSPConn(in io.Writer, out io.Reader, settings map[string]any, workspaceFolders ...lspWorkspaceFolder) *lspConn {
	c := &lspConn{
		in: in, out: bufio.NewReader(out), nextID: 1,
		diagnostics: make(map[string]lspDiagnosticPublication), diagnosticEvents: make(chan struct{}, 1),
		settings: settings, workspaceFolders: append([]lspWorkspaceFolder(nil), workspaceFolders...),
		dynamicCapabilities: make(map[string]string), capabilityEvents: make(chan struct{}, 1),
	}
	c.initialize()
	go c.readLoop()
	return c
}

func (c *lspConn) initialize() {
	c.initOnce.Do(func() {
		c.writes = make(chan lspWriteRequest)
		c.pending = make(map[string]chan lspWireMessage)
		if c.dynamicCapabilities == nil {
			c.dynamicCapabilities = make(map[string]string)
		}
		if c.capabilityEvents == nil {
			c.capabilityEvents = make(chan struct{}, 1)
		}
		c.closed = make(chan struct{})
		if c.nextID == 0 {
			c.nextID = 1
		}
		go c.writeLoop()
	})
}

func (c *lspConn) readLoop() {
	c.initialize()
	for {
		payload, err := readLSPFrame(c.out)
		if err != nil {
			c.fail(err)
			return
		}
		var msg lspWireMessage
		if err := json.Unmarshal(payload, &msg); err != nil {
			c.fail(fmt.Errorf("decode LSP message: %w", err))
			return
		}
		if msg.Method != "" {
			if err := c.handleIncoming(msg); err != nil {
				c.fail(err)
				return
			}
			continue
		}
		c.dispatchResponse(msg)
	}
}

func (c *lspConn) fail(err error) {
	if err == nil {
		err = io.EOF
	}
	c.initialize()
	c.stateMu.Lock()
	if c.terminal == nil {
		c.terminal = err
		close(c.closed)
	}
	c.stateMu.Unlock()
}

func (c *lspConn) terminalError() error {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	if c.terminal != nil {
		return c.terminal
	}
	return io.EOF
}

func (c *lspConn) dispatchResponse(msg lspWireMessage) {
	key := strings.TrimSpace(string(msg.ID))
	if key == "" {
		return
	}
	c.stateMu.Lock()
	response := c.pending[key]
	if response != nil {
		delete(c.pending, key)
	}
	c.stateMu.Unlock()
	if response != nil {
		response <- msg
	}
}

func readLSPFrame(r *bufio.Reader) ([]byte, error) {
	contentLength := -1
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		name, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(name), "Content-Length") {
			n, err := strconv.Atoi(strings.TrimSpace(value))
			if err != nil || n < 0 || n > maxLSPMessageBytes {
				return nil, fmt.Errorf("invalid LSP Content-Length %q", strings.TrimSpace(value))
			}
			contentLength = n
		}
	}
	if contentLength < 0 {
		return nil, fmt.Errorf("LSP frame missing Content-Length")
	}
	payload := make([]byte, contentLength)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, err
	}
	return payload, nil
}

func (c *lspConn) writeFrame(payload []byte) error {
	if _, err := fmt.Fprintf(c.in, "Content-Length: %d\r\n\r\n", len(payload)); err != nil {
		return err
	}
	_, err := c.in.Write(payload)
	return err
}

func (c *lspConn) writeLoop() {
	for {
		select {
		case <-c.closed:
			return
		case req := <-c.writes:
			if err := req.ctx.Err(); err != nil {
				req.result <- err
				continue
			}
			err := c.writeFrame(req.payload)
			req.result <- err
			if err != nil {
				c.fail(err)
				return
			}
		}
	}
}

func (c *lspConn) notifyContext(ctx context.Context, method string, params any) error {
	return c.sendContext(ctx, map[string]any{"jsonrpc": "2.0", "method": method, "params": params})
}

// sendContext bounds pipe writes. A server that stops reading stdin must not
// pin a cancelled tool call; invalidation kills the process and lets any
// abandoned writer unwind.
func (c *lspConn) sendContext(ctx context.Context, value any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	payload, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(payload) > maxLSPMessageBytes {
		return fmt.Errorf("LSP message is %d bytes, over the %dMB cap", len(payload), maxLSPMessageBytes/(1024*1024))
	}
	c.initialize()
	req := lspWriteRequest{ctx: ctx, payload: payload, result: make(chan error, 1)}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-c.closed:
		return c.terminalError()
	case c.writes <- req:
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-c.closed:
		return c.terminalError()
	case err := <-req.result:
		return err
	}
}

func (c *lspConn) request(ctx context.Context, method string, params any) (json.RawMessage, error) {
	c.initialize()
	c.stateMu.Lock()
	if c.terminal != nil {
		err := c.terminal
		c.stateMu.Unlock()
		return nil, err
	}
	id := c.nextID
	c.nextID++
	key := strconv.Itoa(id)
	response := make(chan lspWireMessage, 1)
	c.pending[key] = response
	c.stateMu.Unlock()
	defer func() {
		c.stateMu.Lock()
		if c.pending[key] == response {
			delete(c.pending, key)
		}
		c.stateMu.Unlock()
	}()
	if err := c.sendContext(ctx, map[string]any{"jsonrpc": "2.0", "id": id, "method": method, "params": params}); err != nil {
		return nil, err
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.closed:
		return nil, c.terminalError()
	case msg := <-response:
		if msg.Error != nil {
			return nil, fmt.Errorf("LSP %s failed (%d): %s", method, msg.Error.Code, msg.Error.Message)
		}
		return msg.Result, nil
	}
}

func wireIDEqual(raw json.RawMessage, id int) bool {
	return string(bytes.TrimSpace(raw)) == strconv.Itoa(id)
}

func (c *lspConn) handleIncoming(msg lspWireMessage) error {
	if msg.Method == "textDocument/publishDiagnostics" {
		var params struct {
			URI         string          `json:"uri"`
			Version     *int            `json:"version,omitempty"`
			Diagnostics json.RawMessage `json:"diagnostics"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			return fmt.Errorf("decode published diagnostics: %w", err)
		}
		rawDiagnostics := bytes.TrimSpace(params.Diagnostics)
		if params.URI == "" || len(rawDiagnostics) == 0 || rawDiagnostics[0] != '[' {
			return fmt.Errorf("decode published diagnostics: uri and diagnostics array are required")
		}
		var diagnostics []lspDiagnostic
		if err := json.Unmarshal(rawDiagnostics, &diagnostics); err != nil {
			return fmt.Errorf("decode published diagnostic items: %w", err)
		}
		c.diagnosticsMu.Lock()
		existing, exists := c.diagnostics[params.URI]
		stale := exists && existing.version != nil && (params.Version == nil || *params.Version < *existing.version)
		if !stale {
			c.diagnostics[params.URI] = lspDiagnosticPublication{
				items: diagnostics, version: params.Version, receivedAt: time.Now(),
			}
		}
		c.diagnosticsMu.Unlock()
		if !stale {
			select {
			case c.diagnosticEvents <- struct{}{}:
			default:
			}
		}
	}
	if len(msg.ID) == 0 {
		return nil
	}
	// Language servers may ask the client to register capabilities, create
	// progress tokens, or fetch configuration at any time. The read loop handles
	// these immediately, including while the retained client is idle.
	var result any
	var responseError *lspResponseError
	switch msg.Method {
	case "workspace/configuration":
		var params struct {
			Items []struct {
				Section string `json:"section"`
			} `json:"items"`
		}
		_ = json.Unmarshal(msg.Params, &params)
		values := make([]any, len(params.Items))
		for i, item := range params.Items {
			values[i] = lspSettingSection(c.settings, item.Section)
		}
		result = values
	case "workspace/workspaceFolders":
		result = c.workspaceFolders
	case "workspace/applyEdit":
		result = map[string]any{"applied": false, "failureReason": "AgentRay LSP tool is read-only"}
	case "client/registerCapability":
		c.registerDynamicCapabilities(msg.Params)
	case "client/unregisterCapability":
		c.unregisterDynamicCapabilities(msg.Params)
	case "window/workDoneProgress/create", "window/showMessageRequest",
		"workspace/semanticTokens/refresh", "workspace/inlayHint/refresh",
		"workspace/codeLens/refresh", "workspace/codeAction/refresh",
		"workspace/inlineValue/refresh", "workspace/foldingRange/refresh",
		"workspace/diagnostic/refresh":
		// These requests either acknowledge registration/refresh or represent a
		// headless prompt for which null means no action selected.
	case "window/showDocument":
		result = map[string]any{"success": false}
	default:
		responseError = &lspResponseError{Code: -32601, Message: "Method not found: " + msg.Method}
	}
	ctx, cancel := context.WithTimeout(context.Background(), lspWriteTimeout)
	defer cancel()
	response := map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID)}
	if responseError != nil {
		response["error"] = responseError
	} else {
		response["result"] = result
	}
	return c.sendContext(ctx, response)
}

func (c *lspConn) registerDynamicCapabilities(raw json.RawMessage) {
	var params struct {
		Registrations []struct {
			ID     string `json:"id"`
			Method string `json:"method"`
		} `json:"registrations"`
	}
	if json.Unmarshal(raw, &params) != nil {
		return
	}
	c.stateMu.Lock()
	changed := false
	for _, registration := range params.Registrations {
		if registration.ID != "" && registration.Method != "" {
			changed = changed || c.dynamicCapabilities[registration.ID] != registration.Method
			c.dynamicCapabilities[registration.ID] = registration.Method
		}
	}
	c.stateMu.Unlock()
	if changed {
		select {
		case c.capabilityEvents <- struct{}{}:
		default:
		}
	}
}

func (c *lspConn) unregisterDynamicCapabilities(raw json.RawMessage) {
	var params struct {
		Unregistrations []struct {
			ID string `json:"id"`
		} `json:"unregistrations"`
		Unregisterations []struct {
			ID string `json:"id"`
		} `json:"unregisterations"`
	}
	if json.Unmarshal(raw, &params) != nil {
		return
	}
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	for _, registration := range append(params.Unregistrations, params.Unregisterations...) {
		delete(c.dynamicCapabilities, registration.ID)
	}
}

func (c *lspConn) supportsDynamicCapability(method string) bool {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	for _, registeredMethod := range c.dynamicCapabilities {
		if registeredMethod == method {
			return true
		}
	}
	return false
}

func lspSettingSection(settings map[string]any, section string) any {
	if section == "" {
		return settings
	}
	var current any = settings
	for _, part := range strings.Split(section, ".") {
		object, ok := current.(map[string]any)
		if !ok {
			return nil
		}
		current, ok = object[part]
		if !ok {
			return nil
		}
	}
	return current
}

func (c *lspConn) waitForDiagnostics(ctx context.Context, uri string, version int) ([]lspDiagnostic, bool, error) {
	diagnostics, known, _, err := c.waitForDiagnosticsOrCapability(ctx, uri, version, "")
	return diagnostics, known, err
}

func (c *lspConn) waitForDiagnosticsOrCapability(ctx context.Context, uri string, version int, capability string) ([]lspDiagnostic, bool, bool, error) {
	c.initialize()
	for {
		diagnostics, known, settleAfter := c.diagnosticsStatus(uri, version)
		if known {
			return diagnostics, true, false, nil
		}
		if capability != "" && c.supportsDynamicCapability(capability) {
			return nil, false, true, nil
		}
		var settle <-chan time.Time
		var timer *time.Timer
		if settleAfter > 0 {
			timer = time.NewTimer(settleAfter)
			settle = timer.C
		}
		select {
		case <-ctx.Done():
			stopTimer(timer)
			return nil, false, false, nil
		case <-c.closed:
			stopTimer(timer)
			return nil, false, false, c.terminalError()
		case <-c.diagnosticEvents:
			stopTimer(timer)
		case <-c.capabilityEvents:
			stopTimer(timer)
		case <-settle:
		}
	}
}

func (c *lspConn) currentDiagnostics(uri string, version int) ([]lspDiagnostic, bool) {
	diagnostics, known, _ := c.diagnosticsStatus(uri, version)
	return diagnostics, known
}

func (c *lspConn) diagnosticsStatus(uri string, version int) ([]lspDiagnostic, bool, time.Duration) {
	c.diagnosticsMu.Lock()
	defer c.diagnosticsMu.Unlock()
	publication, ok := c.diagnostics[uri]
	if !ok {
		return nil, false, 0
	}
	if publication.version != nil && *publication.version != version {
		if *publication.version < version {
			delete(c.diagnostics, uri)
		}
		return nil, false, 0
	}
	if publication.version == nil {
		remaining := lspDiagnosticSettleDelay - time.Since(publication.receivedAt)
		if remaining > 0 {
			return nil, false, remaining
		}
	}
	return publication.items, true, 0
}

func stopTimer(timer *time.Timer) {
	if timer == nil || timer.Stop() {
		return
	}
	select {
	case <-timer.C:
	default:
	}
}

// drainPending detects a terminal reader error and drops any response that has
// no live request. Server notifications and requests are dispatched directly by
// the read loop and diagnostics are cleared separately by document generation.
func (c *lspConn) drainPending() error {
	c.initialize()
	select {
	case <-c.closed:
		return c.terminalError()
	default:
		return nil
	}
}

// clearDiagnostics prevents a retained client from treating the previous open's
// publication as the result for newly read workspace bytes. Calls are
// serialized by lspClient.executionMu. The diagnostics mutex coordinates with
// the always-on read loop.
func (c *lspConn) clearDiagnostics(uri string) {
	c.diagnosticsMu.Lock()
	defer c.diagnosticsMu.Unlock()
	delete(c.diagnostics, uri)
}
