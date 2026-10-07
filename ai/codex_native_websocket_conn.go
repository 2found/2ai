package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// A connection owns one reader for its entire lifetime, including cached idle
// periods, so control frames and remote closure cannot leave stale cache state.
type codexWebSocketConn struct {
	conn     *websocket.Conn
	state    atomic.Int32
	mu       sync.Mutex
	writeMu  sync.Mutex
	parser   *codexWebSocketParser
	terminal error
	finished chan struct{}
}

func connectCodexWebSocket(ctx context.Context, url string, headers http.Header, timeout time.Duration, dialer *websocket.Dialer) (*codexSocket, error) {
	if ctx.Err() != nil {
		return nil, errors.New("Request was aborted")
	}
	d := *websocket.DefaultDialer
	if dialer != nil {
		d = *dialer
	}
	// Pi only enables a handshake timeout when explicitly configured.
	d.HandshakeTimeout = 0
	dialCtx := ctx
	cancel := func() {}
	if timeout > 0 {
		dialCtx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()
	// DialContext applies deadlines, but cancellation after TCP connect also
	// needs to interrupt an HTTP upgrade blocked in a read.
	var stops []func() bool
	track := func(dial func(context.Context, string, string) (net.Conn, error)) func(context.Context, string, string) (net.Conn, error) {
		return func(ctx context.Context, network, address string) (net.Conn, error) {
			conn, err := dial(ctx, network, address)
			if err == nil {
				stops = append(stops, context.AfterFunc(dialCtx, func() { _ = conn.Close() }))
			}
			return conn, err
		}
	}
	baseDial := d.NetDialContext
	if baseDial == nil {
		if d.NetDial != nil {
			baseDial = func(_ context.Context, network, address string) (net.Conn, error) { return d.NetDial(network, address) }
		} else {
			baseDial = (&net.Dialer{}).DialContext
		}
	}
	d.NetDialContext = track(baseDial)
	if d.NetDialTLSContext != nil {
		d.NetDialTLSContext = track(d.NetDialTLSContext)
	}
	defer func() {
		for _, stop := range stops {
			stop()
		}
	}()
	conn, response, err := d.DialContext(dialCtx, url, headers.Clone())
	if err != nil {
		if response != nil && response.Body != nil {
			response.Body.Close()
		}
		if ctx.Err() != nil {
			return nil, errors.New("Request was aborted")
		}
		deadline, hasDeadline := dialCtx.Deadline()
		if dialCtx.Err() != nil || (hasDeadline && !time.Now().Before(deadline)) {
			return nil, fmt.Errorf("WebSocket connect timeout after %gms", float64(timeout)/float64(time.Millisecond))
		}
		return nil, err
	}
	if ctx.Err() != nil {
		conn.Close()
		return nil, errors.New("Request was aborted")
	}
	native := &codexWebSocketConn{conn: conn, finished: make(chan struct{})}
	native.state.Store(1)
	socket := &codexSocket{
		ready:  func() *int { state := int(native.state.Load()); return &state },
		close:  func(code int, reason string) { native.close(code, reason) },
		send:   native.send,
		listen: native.listen,
	}
	go native.read()
	return socket, nil
}
func (c *codexWebSocketConn) close(code int, reason string) {
	if !c.state.CompareAndSwap(1, 2) {
		return
	}
	_ = c.conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(code, reason), time.Now().Add(time.Second))
	_ = c.conn.Close()
	c.state.Store(3)
}
func (c *codexWebSocketConn) send(ctx context.Context, payload []byte) error {
	if ctx.Err() != nil {
		return errors.New("Request was aborted")
	}
	stop := context.AfterFunc(ctx, func() { c.close(1000, "aborted") })
	defer stop()
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	err := c.conn.WriteMessage(websocket.TextMessage, payload)
	if ctx.Err() != nil {
		return errors.New("Request was aborted")
	}
	return err
}
func (c *codexWebSocketConn) listen(p *codexWebSocketParser) func() {
	c.mu.Lock()
	c.parser = p
	if c.terminal != nil {
		p.fail(c.terminal)
	}
	c.mu.Unlock()
	return func() {
		c.mu.Lock()
		if c.parser == p {
			c.parser = nil
		}
		c.mu.Unlock()
	}
}
func (c *codexWebSocketConn) read() {
	defer close(c.finished)
	defer c.conn.Close()
	for {
		kind, data, err := c.conn.ReadMessage()
		c.mu.Lock()
		if err != nil {
			c.state.Store(3)
			var closed *websocket.CloseError
			if errors.As(err, &closed) {
				// A received close frame completes the close handshake; synthetic 1006
				// represents an interrupted transport and carries no remote reason.
				reason := closed.Text
				clean := closed.Code != 1006
				if !clean {
					reason = ""
				}
				raw, _ := json.Marshal(map[string]any{"code": closed.Code, "reason": reason, "wasClean": clean})
				c.terminal = codexSocketCloseError(raw)
				if c.parser != nil {
					c.parser.closed(raw)
				}
			} else {
				c.terminal = err
				if c.parser != nil {
					c.parser.fail(err)
				}
			}
			c.mu.Unlock()
			return
		}
		if c.parser != nil {
			c.parser.message(data, kind == websocket.BinaryMessage)
		}
		c.mu.Unlock()
	}
}

func runCodexWebSocketRequest(ctx context.Context, socket *codexSocket, payload []byte, idle time.Duration, consume func(json.RawMessage) (bool, error)) error {
	parser := newCodexWebSocketParser()
	// Install before sending to avoid dropping a synchronous/local peer's reply.
	detach := socket.listen(parser)
	defer detach()
	if err := socket.send(ctx, payload); err != nil {
		return err
	}
	return parser.run(ctx, idle, func() { socket.closeSilently("idle_timeout") }, consume)
}
