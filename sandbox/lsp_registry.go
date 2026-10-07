package sandbox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore"
)

const (
	defaultLSPRegistryCapacity = 4
	defaultLSPIdleTimeout      = 5 * time.Minute
	lspProcessLifetime         = time.Hour
	lspShutdownTimeout         = time.Second
)

var (
	errLSPClientClosed  = errors.New("LSP client is closed")
	errLSPClientReaping = errors.New("previous LSP process is still shutting down")
)

// LSPSessionRegistry retains initialized language servers across the
// short-lived tool instances built for consecutive turns. It is deliberately
// process-local: another server replica starts a clean client, so correctness
// never depends on sticky routing or a shared daemon.
//
// Capacity is a soft limit. An in-flight client is never killed to make room.
type LSPSessionRegistry struct {
	mu       sync.Mutex
	gates    [64]sync.Mutex
	clients  map[string]*lspClient
	closing  map[string]*lspClient
	capacity int
	idle     time.Duration
	now      func() time.Time
	closed   bool
}

func NewLSPSessionRegistry(capacity int, idle time.Duration) *LSPSessionRegistry {
	if capacity <= 0 {
		capacity = defaultLSPRegistryCapacity
	}
	if idle <= 0 {
		idle = defaultLSPIdleTimeout
	}
	return &LSPSessionRegistry{
		clients: make(map[string]*lspClient), closing: make(map[string]*lspClient),
		capacity: capacity, idle: idle, now: time.Now,
	}
}

type lspRegistryEntry struct {
	refs     int
	lastUsed time.Time
	timer    *time.Timer
}

// lspClient owns one initialized server. executionMu is intentionally held for
// the whole document-sync/query/close sequence so two actions cannot overlap
// didOpen/didClose lifecycles for the same retained server.
type lspClient struct {
	executionMu        sync.Mutex
	entry              lspRegistryEntry
	process            agentcore.SandboxProcess
	conn               *lspConn
	diagnosticProvider json.RawMessage
	stderr             cappedStringWriter
	stderrDone         chan struct{}
	cleanup            func()
	waitOnce           sync.Once
	cleanupOnce        sync.Once
	exitDone           chan struct{}
	shutdownTimeout    time.Duration
	version            int
	broken             bool
	closed             bool
}

func (r *LSPSessionRegistry) acquire(key string, start func() (*lspClient, error)) (*lspClient, error) {
	if r == nil {
		return nil, errors.New("LSP session registry is nil")
	}
	gate := &r.gates[evalRegistryStripe(key)]
	gate.Lock()
	defer gate.Unlock()

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil, errLSPClientClosed
	}
	if closing := r.closing[key]; closing != nil {
		if closing.exitConfirmed() {
			delete(r.closing, key)
		} else {
			r.mu.Unlock()
			return nil, errLSPClientReaping
		}
	}
	if current := r.clients[key]; current != nil {
		if current.entry.timer != nil {
			current.entry.timer.Stop()
			current.entry.timer = nil
		}
		current.entry.refs++
		r.mu.Unlock()
		return current, nil
	}
	victims := r.capacityVictimsLocked()
	r.mu.Unlock()
	for _, victim := range victims {
		r.closeTracked(victim.key, victim.client)
	}

	client, err := start()
	if err != nil {
		return nil, err
	}
	client.ensureProcessWaiter()
	client.entry.refs = 1
	client.entry.lastUsed = r.now()
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		_ = client.close()
		return nil, errLSPClientClosed
	}
	r.clients[key] = client
	r.mu.Unlock()
	return client, nil
}

type lspRegistryVictim struct {
	key    string
	client *lspClient
}

func (r *LSPSessionRegistry) capacityVictimsLocked() []lspRegistryVictim {
	if len(r.clients) < r.capacity {
		return nil
	}
	type candidate struct {
		key  string
		when time.Time
	}
	var candidates []candidate
	for key, client := range r.clients {
		if client.entry.refs == 0 {
			candidates = append(candidates, candidate{key: key, when: client.entry.lastUsed})
		}
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].when.Before(candidates[j].when) })
	need := len(r.clients) - r.capacity + 1
	if need > len(candidates) {
		need = len(candidates)
	}
	victims := make([]lspRegistryVictim, 0, need)
	for _, candidate := range candidates[:need] {
		client := r.clients[candidate.key]
		delete(r.clients, candidate.key)
		r.closing[candidate.key] = client
		if client.entry.timer != nil {
			client.entry.timer.Stop()
			client.entry.timer = nil
		}
		victims = append(victims, lspRegistryVictim{key: candidate.key, client: client})
	}
	return victims
}

func (r *LSPSessionRegistry) release(key string, client *lspClient) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.clients[key] != client || client.entry.refs <= 0 {
		return
	}
	client.entry.refs--
	client.entry.lastUsed = r.now()
	if client.entry.refs != 0 {
		return
	}
	if client.entry.timer != nil {
		client.entry.timer.Stop()
	}
	client.entry.timer = time.AfterFunc(r.idle, func() { r.expire(key, client) })
}

func (r *LSPSessionRegistry) expire(key string, client *lspClient) {
	gate := &r.gates[evalRegistryStripe(key)]
	gate.Lock()
	defer gate.Unlock()
	r.mu.Lock()
	if r.clients[key] != client || client.entry.refs != 0 || r.now().Sub(client.entry.lastUsed) < r.idle {
		r.mu.Unlock()
		return
	}
	delete(r.clients, key)
	r.closing[key] = client
	client.entry.timer = nil
	r.mu.Unlock()
	r.closeTracked(key, client)
}

func (r *LSPSessionRegistry) invalidate(key string, client *lspClient) {
	gate := &r.gates[evalRegistryStripe(key)]
	gate.Lock()
	defer gate.Unlock()
	r.mu.Lock()
	if r.clients[key] == client {
		delete(r.clients, key)
		r.closing[key] = client
		if client.entry.timer != nil {
			client.entry.timer.Stop()
			client.entry.timer = nil
		}
	}
	r.mu.Unlock()
	r.closeTracked(key, client)
}

func (r *LSPSessionRegistry) closeTracked(key string, client *lspClient) {
	if client.close() {
		r.mu.Lock()
		if r.closing[key] == client {
			delete(r.closing, key)
		}
		r.mu.Unlock()
		return
	}
	done := client.exitDone
	if done == nil {
		return
	}
	go func() {
		<-done
		r.mu.Lock()
		if r.closing[key] == client {
			delete(r.closing, key)
		}
		r.mu.Unlock()
	}()
}

// Close releases every retained server. Embedding hosts should call it during
// shutdown; idle and hard-lifetime bounds also make forgotten cleanup finite.
func (r *LSPSessionRegistry) Close() {
	if r == nil {
		return
	}
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	r.closed = true
	clients := make([]*lspClient, 0, len(r.clients))
	for key, client := range r.clients {
		delete(r.clients, key)
		if client.entry.timer != nil {
			client.entry.timer.Stop()
			client.entry.timer = nil
		}
		clients = append(clients, client)
	}
	r.mu.Unlock()
	for _, client := range clients {
		_ = client.close()
	}
}

func (c *lspClient) execute(ctx context.Context, tool *LSPTool, server LSPServerConfig, uri string, req lspActionRequest) (output string, err error) {
	c.executionMu.Lock()
	defer func() {
		if err != nil {
			c.broken = true
		}
		c.executionMu.Unlock()
	}()
	if c.closed || c.broken || c.process == nil || c.conn == nil {
		return "", errLSPClientClosed
	}
	if err := c.conn.drainPending(); err != nil {
		return "", err
	}
	c.version++
	c.conn.clearDiagnostics(uri)
	if err := c.conn.notifyContext(ctx, "textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{
			"uri": uri, "languageId": server.LanguageID, "version": c.version, "text": req.content,
		},
	}); err != nil {
		return "", err
	}
	output, actionErr := tool.executeLSPAction(ctx, c.conn, c.diagnosticProvider, uri, c.version, req)
	closeCtx, cancel := context.WithTimeout(context.Background(), lspWriteTimeout)
	closeErr := c.conn.notifyContext(closeCtx, "textDocument/didClose", map[string]any{"textDocument": map[string]any{"uri": uri}})
	cancel()
	if actionErr != nil {
		return "", actionErr
	}
	if closeErr != nil {
		return "", closeErr
	}
	return output, nil
}

func (c *lspClient) ensureProcessWaiter() {
	if c.process == nil {
		return
	}
	c.waitOnce.Do(func() {
		c.exitDone = make(chan struct{})
		go func() {
			_, _ = c.process.Wait()
			c.finishCleanup()
			close(c.exitDone)
		}()
	})
}

func (c *lspClient) finishCleanup() {
	c.cleanupOnce.Do(func() {
		if c.cleanup != nil {
			c.cleanup()
		}
	})
}

func (c *lspClient) exitConfirmed() bool {
	if c.process == nil {
		return true
	}
	c.ensureProcessWaiter()
	select {
	case <-c.exitDone:
		return true
	default:
		return false
	}
}

func (c *lspClient) waitForExit(timeout time.Duration) bool {
	if c.exitConfirmed() {
		return true
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-c.exitDone:
		return true
	case <-timer.C:
		return false
	}
}

func (c *lspClient) close() bool {
	c.executionMu.Lock()
	defer c.executionMu.Unlock()
	if c.closed {
		return c.exitConfirmed()
	}
	c.closed = true
	if c.process == nil {
		c.finishCleanup()
		return true
	}
	c.ensureProcessWaiter()
	timeout := c.shutdownTimeout
	if timeout <= 0 {
		timeout = lspShutdownTimeout
	}
	shutdownCompleted := false
	if c.conn != nil {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		if _, err := c.conn.request(ctx, "shutdown", nil); err == nil {
			shutdownCompleted = true
			_ = c.conn.notifyContext(ctx, "exit", nil)
		}
		cancel()
	}
	_ = c.process.Stdin().Close()
	if (shutdownCompleted || c.conn == nil) && c.waitForExit(timeout) {
		return true
	}
	_ = c.process.Kill()
	return c.waitForExit(timeout)
}

func (c *lspClient) errorWithStderr(err error) error {
	if err == nil {
		return nil
	}
	return withLSPStderr(fmt.Errorf("%w", err), c.stderr.String())
}
