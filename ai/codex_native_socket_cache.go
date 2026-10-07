package ai

import (
	"context"
	"sync"
	"time"
)

// Socket operations are concrete callbacks so native transports and the source
// conformance harness share ownership semantics without a provider interface.
type codexSocket struct {
	ready  func() *int
	close  func(int, string)
	send   func(context.Context, []byte) error
	listen func(*codexWebSocketParser) func()
}

func (s *codexSocket) reusable() bool {
	if s.ready == nil {
		return true
	}
	state := s.ready()
	return state == nil || *state == 1
}
func (s *codexSocket) closeSilently(reason string) {
	defer func() { _ = recover() }()
	if s.close != nil {
		s.close(1000, reason)
	}
}

type codexSocketEntry struct {
	socket       *codexSocket
	busy         bool
	created      time.Time
	stopIdle     func()
	continuation *codexWebSocketContinuation
}
type codexSocketKey struct{ session, account string }
type codexSocketLease struct {
	socket  *codexSocket
	entry   *codexSocketEntry
	reused  bool
	release func(bool)
}
type codexSocketCache struct {
	mu      sync.Mutex
	entries map[codexSocketKey]*codexSocketEntry
	order   []codexSocketKey
	now     func() time.Time
	after   func(time.Duration, func()) func()
}

func newCodexSocketCache() *codexSocketCache {
	return &codexSocketCache{entries: map[codexSocketKey]*codexSocketEntry{}, now: time.Now, after: func(delay time.Duration, run func()) func() {
		timer := time.AfterFunc(delay, run)
		return func() { timer.Stop() }
	}}
}
func (c *codexSocketCache) remove(key codexSocketKey, entry *codexSocketEntry) {
	if c.entries[key] == entry {
		delete(c.entries, key)
		for i, k := range c.order {
			if k == key {
				c.order = append(c.order[:i], c.order[i+1:]...)
				break
			}
		}
	}
}
func (c *codexSocketCache) insert(key codexSocketKey, entry *codexSocketEntry) {
	if c.entries[key] == nil {
		// Pi stores nested insertion-ordered Maps: sessions, then accounts.
		at := len(c.order)
		for i, k := range c.order {
			if k.session == key.session {
				at = i + 1
			}
		}
		c.order = append(c.order, codexSocketKey{})
		copy(c.order[at+1:], c.order[at:])
		c.order[at] = key
	}
	c.entries[key] = entry
}
func (c *codexSocketCache) idle(key codexSocketKey, entry *codexSocketEntry) {
	if entry.stopIdle != nil {
		entry.stopIdle()
	}
	entry.stopIdle = c.after(5*time.Minute, func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		if entry.busy {
			return
		}
		entry.socket.closeSilently("idle_timeout")
		c.remove(key, entry)
	})
}
func (c *codexSocketCache) lease(key codexSocketKey, entry *codexSocketEntry, reused bool) codexSocketLease {
	return codexSocketLease{socket: entry.socket, entry: entry, reused: reused, release: func(keep bool) {
		c.mu.Lock()
		defer c.mu.Unlock()
		if !keep || !entry.socket.reusable() {
			entry.socket.closeSilently("done")
			if !reused && entry.stopIdle != nil {
				entry.stopIdle()
				entry.stopIdle = nil
			}
			c.remove(key, entry)
			return
		}
		entry.busy = false
		c.idle(key, entry)
	}}
}
func (c *codexSocketCache) acquire(ctx context.Context, session, account string, connect func(context.Context) (*codexSocket, error)) (codexSocketLease, error) {
	key := codexSocketKey{session, account}
	cache := session != ""
	c.mu.Lock()
	if entry := c.entries[key]; cache && entry != nil {
		if entry.stopIdle != nil {
			entry.stopIdle()
			entry.stopIdle = nil
		}
		if !entry.busy && c.now().Sub(entry.created) >= 55*time.Minute {
			entry.socket.closeSilently("connection_age_limit")
			c.remove(key, entry)
		} else if !entry.busy && entry.socket.reusable() {
			entry.busy = true
			lease := c.lease(key, entry, true)
			c.mu.Unlock()
			return lease, nil
		}
		if entry.busy {
			cache = false
		} else if !entry.socket.reusable() {
			entry.socket.closeSilently("done")
			c.remove(key, entry)
		}
	}
	c.mu.Unlock()
	socket, err := connect(ctx)
	if err != nil {
		return codexSocketLease{}, err
	}
	if !cache {
		return codexSocketLease{socket: socket, release: func(bool) { socket.closeSilently("done") }}, nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	entry := &codexSocketEntry{socket: socket, busy: true, created: c.now()}
	c.insert(key, entry)
	return c.lease(key, entry, false), nil
}
func (c *codexSocketCache) closeSessions(session string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, key := range append([]codexSocketKey(nil), c.order...) {
		entry := c.entries[key]
		if session != "" && key.session != session {
			continue
		}
		if entry.stopIdle != nil {
			entry.stopIdle()
			entry.stopIdle = nil
		}
		entry.socket.closeSilently("debug_close")
		c.remove(key, entry)
	}
}
