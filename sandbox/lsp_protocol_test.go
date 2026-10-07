package sandbox

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestLSPFramingHandlesSplitHeadersAndExactPayload(t *testing.T) {
	payload := []byte(`{"jsonrpc":"2.0","id":1,"result":{"text":"line1\\nline2"}}`)
	stream := "Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n" +
		"Content-Length: " + stringInt(len(payload)) + "\r\n\r\n" + string(payload) +
		"Content-Length: 2\r\n\r\n{}"
	r := bufio.NewReader(strings.NewReader(stream))
	first, err := readLSPFrame(r)
	if err != nil {
		t.Fatalf("first frame: %v", err)
	}
	if !bytes.Equal(first, payload) {
		t.Fatalf("payload = %q, want %q", first, payload)
	}
	second, err := readLSPFrame(r)
	if err != nil || string(second) != "{}" {
		t.Fatalf("second = %q, err=%v", second, err)
	}
}

func TestResolveLSPSymbolPositionUsesUTF16AndOccurrence(t *testing.T) {
	pos, err := resolveLSPSymbolPosition("😀 const target = target;\n", 1, "target#2")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if pos.Line != 0 || pos.Character != 18 {
		t.Fatalf("position = %+v, want line=0 character=18", pos)
	}
	if _, err := resolveLSPSymbolPosition("target + target\n", 1, "target"); err == nil || !strings.Contains(err.Error(), "occurs 2") {
		t.Fatalf("ambiguous error = %v", err)
	}
}

func stringInt(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

func TestLSPConnServicesIdleServerRequestsAndNotificationBursts(t *testing.T) {
	serverReads, clientWrites := io.Pipe()
	clientReads, serverWrites := io.Pipe()
	t.Cleanup(func() {
		_ = clientWrites.Close()
		_ = serverWrites.Close()
		_ = serverReads.Close()
		_ = clientReads.Close()
	})
	_ = newLSPConn(clientWrites, clientReads, map[string]any{
		"python": map[string]any{"analysis": map[string]any{"mode": "strict"}},
	})

	response := make(chan lspWireMessage, 1)
	go func() {
		// The old action-driven dispatcher stalled after its 64-message buffer
		// filled. The retained connection must consume notifications while idle.
		for i := 0; i < 100; i++ {
			writeLSPFrame(serverWrites, map[string]any{
				"jsonrpc": "2.0", "method": "window/logMessage",
				"params": map[string]any{"type": 4, "message": fmt.Sprintf("message-%d", i)},
			})
		}
		writeLSPFrame(serverWrites, map[string]any{
			"jsonrpc": "2.0", "id": 41, "method": "workspace/configuration",
			"params": map[string]any{"items": []any{map[string]any{"section": "python.analysis"}}},
		})
		payload, err := readLSPFrame(bufio.NewReader(serverReads))
		if err != nil {
			return
		}
		var msg lspWireMessage
		_ = json.Unmarshal(payload, &msg)
		response <- msg
	}()

	select {
	case msg := <-response:
		if !wireIDEqual(msg.ID, 41) || !strings.Contains(string(msg.Result), `"mode":"strict"`) {
			t.Fatalf("configuration response = %+v", msg)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("idle server request was not serviced")
	}
}

func TestLSPConnHandlesWorkspaceAndCapabilityRequests(t *testing.T) {
	serverReads, clientWrites := io.Pipe()
	clientReads, serverWrites := io.Pipe()
	t.Cleanup(func() {
		_ = clientWrites.Close()
		_ = serverWrites.Close()
		_ = serverReads.Close()
		_ = clientReads.Close()
	})
	c := newLSPConn(clientWrites, clientReads, nil, lspWorkspaceFolder{
		URI: "file:///workspace", Name: "workspace",
	})
	serverReader := bufio.NewReader(serverReads)
	roundTrip := func(id int, method string, params any) lspWireMessage {
		t.Helper()
		writeLSPFrame(serverWrites, map[string]any{
			"jsonrpc": "2.0", "id": id, "method": method, "params": params,
		})
		payload, err := readLSPFrame(serverReader)
		if err != nil {
			t.Fatal(err)
		}
		var response lspWireMessage
		if err := json.Unmarshal(payload, &response); err != nil {
			t.Fatal(err)
		}
		return response
	}

	folders := roundTrip(1, "workspace/workspaceFolders", nil)
	if folders.Error != nil || !strings.Contains(string(folders.Result), `"uri":"file:///workspace"`) {
		t.Fatalf("workspace folders response = %+v", folders)
	}
	roundTrip(2, "client/registerCapability", map[string]any{"registrations": []any{
		map[string]any{"id": "diagnostics", "method": "textDocument/diagnostic"},
	}})
	if !c.supportsDynamicCapability("textDocument/diagnostic") {
		t.Fatal("dynamic diagnostic capability was not registered")
	}
	roundTrip(3, "client/unregisterCapability", map[string]any{"unregistrations": []any{
		map[string]any{"id": "diagnostics", "method": "textDocument/diagnostic"},
	}})
	if c.supportsDynamicCapability("textDocument/diagnostic") {
		t.Fatal("dynamic diagnostic capability was not unregistered")
	}
	unknown := roundTrip(4, "agentray/unknown", nil)
	if unknown.Error == nil || unknown.Error.Code != -32601 {
		t.Fatalf("unknown method response = %+v", unknown)
	}
}

type blockingLSPWriter struct {
	started chan struct{}
	unblock chan struct{}
	writes  atomic.Int32
}

func (w *blockingLSPWriter) Write(p []byte) (int, error) {
	w.writes.Add(1)
	select {
	case <-w.started:
	default:
		close(w.started)
	}
	<-w.unblock
	return len(p), nil
}

func TestLSPConnWriteHonorsCancellation(t *testing.T) {
	w := &blockingLSPWriter{started: make(chan struct{}), unblock: make(chan struct{})}
	c := &lspConn{in: w}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	started := time.Now()
	err := c.sendContext(ctx, map[string]any{"jsonrpc": "2.0", "method": "blocked"})
	if err != context.DeadlineExceeded {
		t.Fatalf("send error = %v", err)
	}
	if elapsed := time.Since(started); elapsed > 250*time.Millisecond {
		t.Fatalf("cancelled send took %s", elapsed)
	}
	close(w.unblock)
}

func TestLSPConnCanceledWritesDoNotQueueGoroutines(t *testing.T) {
	w := &blockingLSPWriter{started: make(chan struct{}), unblock: make(chan struct{})}
	c := &lspConn{in: w}

	firstCtx, firstCancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer firstCancel()
	if err := c.sendContext(firstCtx, map[string]any{"first": true}); err != context.DeadlineExceeded {
		t.Fatalf("first send error = %v", err)
	}
	for i := 0; i < 9; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
		err := c.sendContext(ctx, map[string]any{"late": i})
		cancel()
		if err != context.DeadlineExceeded {
			t.Fatalf("late send %d error = %v", i, err)
		}
	}
	close(w.unblock)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) && w.writes.Load() < 2 {
		time.Sleep(time.Millisecond)
	}
	// One frame uses one write for its header and one for its payload. Give the
	// writer pump a scheduling turn to prove that no canceled frame follows it.
	time.Sleep(10 * time.Millisecond)
	if got := w.writes.Load(); got != 2 {
		t.Fatalf("physical writes = %d, want only one header/payload frame", got)
	}
}

func TestLSPConnDropsUnknownResponsesWithoutBackpressure(t *testing.T) {
	serverReads, clientWrites := io.Pipe()
	clientReads, serverWrites := io.Pipe()
	t.Cleanup(func() {
		_ = clientWrites.Close()
		_ = serverWrites.Close()
		_ = serverReads.Close()
		_ = clientReads.Close()
	})
	c := newLSPConn(clientWrites, clientReads, nil)
	serverReader := bufio.NewReader(serverReads)
	serverDone := make(chan struct{})
	go func() {
		defer close(serverDone)
		for i := 0; i < 100; i++ {
			writeLSPFrame(serverWrites, map[string]any{"jsonrpc": "2.0", "id": 1000 + i, "result": nil})
		}
		payload, err := readLSPFrame(serverReader)
		if err != nil {
			return
		}
		var request lspWireMessage
		if json.Unmarshal(payload, &request) != nil {
			return
		}
		writeLSPFrame(serverWrites, map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(request.ID), "result": "ok"})
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	result, err := c.request(ctx, "test/afterUnknown", nil)
	if err != nil || string(result) != `"ok"` {
		t.Fatalf("request after unknown burst: result=%s err=%v", result, err)
	}
	<-serverDone
}

func TestLSPConnCorrelatesConcurrentResponsesByID(t *testing.T) {
	serverReads, clientWrites := io.Pipe()
	clientReads, serverWrites := io.Pipe()
	t.Cleanup(func() {
		_ = clientWrites.Close()
		_ = serverWrites.Close()
		_ = serverReads.Close()
		_ = clientReads.Close()
	})
	c := newLSPConn(clientWrites, clientReads, nil)
	serverReader := bufio.NewReader(serverReads)
	go func() {
		requests := make([]lspWireMessage, 2)
		for i := range requests {
			payload, err := readLSPFrame(serverReader)
			if err != nil {
				return
			}
			_ = json.Unmarshal(payload, &requests[i])
		}
		for i := len(requests) - 1; i >= 0; i-- {
			writeLSPFrame(serverWrites, map[string]any{
				"jsonrpc": "2.0", "id": json.RawMessage(requests[i].ID), "result": requests[i].Method,
			})
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for _, method := range []string{"test/first", "test/second"} {
		method := method
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, err := c.request(ctx, method, nil)
			if err != nil {
				errs <- err
				return
			}
			var got string
			if err := json.Unmarshal(result, &got); err != nil || got != method {
				errs <- fmt.Errorf("%s result=%s decoded=%q err=%v", method, result, got, err)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestLSPConnRejectsDiagnosticsFromPriorDocumentVersion(t *testing.T) {
	old := 3
	c := &lspConn{diagnostics: map[string]lspDiagnosticPublication{
		"file:///workspace/main.go": {items: []lspDiagnostic{{Message: "stale"}}, version: &old},
	}}
	if diagnostics, ok := c.currentDiagnostics("file:///workspace/main.go", 4); ok || diagnostics != nil {
		t.Fatalf("accepted stale diagnostics: %+v", diagnostics)
	}
}

func TestLSPConnDoesNotRegressPublishedDiagnosticVersion(t *testing.T) {
	c := &lspConn{
		diagnostics: make(map[string]lspDiagnosticPublication), diagnosticEvents: make(chan struct{}, 1),
	}
	uri := "file:///workspace/main.go"
	if err := c.handleIncoming(publishedDiagnosticsMessage(t, uri, 2, "fresh")); err != nil {
		t.Fatal(err)
	}
	if err := c.handleIncoming(publishedDiagnosticsMessage(t, uri, 1, "stale")); err != nil {
		t.Fatal(err)
	}
	diagnostics, ok := c.currentDiagnostics(uri, 2)
	if !ok || len(diagnostics) != 1 || diagnostics[0].Message != "fresh" {
		t.Fatalf("diagnostics = %+v, known=%v; want version 2 publication", diagnostics, ok)
	}
}

func TestLSPConnRejectsMalformedPublishedDiagnostics(t *testing.T) {
	c := &lspConn{diagnostics: make(map[string]lspDiagnosticPublication), diagnosticEvents: make(chan struct{}, 1)}
	for _, params := range []string{
		`{"uri":"file:///workspace/main.go","version":1}`,
		`{"uri":"file:///workspace/main.go","version":1,"diagnostics":null}`,
		`{"uri":"file:///workspace/main.go","version":1,"diagnostics":{}}`,
	} {
		err := c.handleIncoming(lspWireMessage{
			Method: "textDocument/publishDiagnostics", Params: json.RawMessage(params),
		})
		if err == nil {
			t.Fatalf("accepted malformed diagnostics params %s", params)
		}
	}
	if diagnostics, ok := c.currentDiagnostics("file:///workspace/main.go", 1); ok || diagnostics != nil {
		t.Fatalf("malformed publication became known clean: %+v", diagnostics)
	}
}

func TestLSPConnSettlesOnLatestUnversionedDiagnostics(t *testing.T) {
	c := &lspConn{
		diagnostics: make(map[string]lspDiagnosticPublication), diagnosticEvents: make(chan struct{}, 1),
	}
	uri := "file:///workspace/main.go"
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	type result struct {
		diagnostics []lspDiagnostic
		known       bool
		err         error
	}
	resultC := make(chan result, 1)
	go func() {
		diagnostics, known, err := c.waitForDiagnostics(ctx, uri, 7)
		resultC <- result{diagnostics: diagnostics, known: known, err: err}
	}()

	if err := c.handleIncoming(publishedDiagnosticsMessage(t, uri, 0, "stale")); err != nil {
		t.Fatal(err)
	}
	time.Sleep(lspDiagnosticSettleDelay / 2)
	if err := c.handleIncoming(publishedDiagnosticsMessage(t, uri, 0, "fresh")); err != nil {
		t.Fatal(err)
	}

	got := <-resultC
	if got.err != nil || !got.known || len(got.diagnostics) != 1 || got.diagnostics[0].Message != "fresh" {
		t.Fatalf("wait result = %+v", got)
	}
}

func publishedDiagnosticsMessage(t *testing.T, uri string, version int, message string) lspWireMessage {
	t.Helper()
	params := map[string]any{
		"uri": uri, "diagnostics": []any{map[string]any{"message": message, "range": rangeJSON(0, 0, 0, 1)}},
	}
	if version > 0 {
		params["version"] = version
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	return lspWireMessage{Method: "textDocument/publishDiagnostics", Params: raw}
}

func writeLSPFrame(w io.Writer, value any) {
	payload, _ := json.Marshal(value)
	_, _ = fmt.Fprintf(w, "Content-Length: %d\r\n\r\n", len(payload))
	_, _ = w.Write(payload)
}
