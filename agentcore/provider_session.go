package agentcore

import (
	"sync"
	"time"
)

const (
	defaultProviderSessionCapacity = 1024
	defaultProviderSessionIdleTTL  = 30 * time.Minute
)

type providerSessionEntry struct {
	session  *ProviderSession
	refs     int
	lastUsed time.Time
}

// ProviderSessionRegistry retains conversation state across short-lived Agent
// instances. It is bounded and lease-aware: only idle sessions are evicted, so
// pressure may temporarily exceed capacity rather than closing state a live
// request is using. A cache miss merely makes a provider relearn an optimization;
// request correctness never depends on process affinity.
type ProviderSessionRegistry struct {
	mu       sync.Mutex
	entries  map[string]*providerSessionEntry
	capacity int
	idleTTL  time.Duration
	now      func() time.Time
	closed   bool
}

// NewProviderSessionRegistry builds a bounded registry. Non-positive values use
// conservative defaults suitable for both a desktop daemon and a server worker.
func NewProviderSessionRegistry(capacity int, idleTTL time.Duration) *ProviderSessionRegistry {
	if capacity <= 0 {
		capacity = defaultProviderSessionCapacity
	}
	if idleTTL <= 0 {
		idleTTL = defaultProviderSessionIdleTTL
	}
	return &ProviderSessionRegistry{
		entries: make(map[string]*providerSessionEntry), capacity: capacity,
		idleTTL: idleTTL, now: time.Now,
	}
}

// Acquire leases the state for key. The release function is idempotent. An
// empty key gets an ephemeral session that is closed on release, which keeps
// one-off runs isolated without growing the registry.
func (r *ProviderSessionRegistry) Acquire(key string) (*ProviderSession, func()) {
	if r == nil || key == "" {
		session := NewProviderSession()
		var once sync.Once
		return session, func() { once.Do(session.Close) }
	}

	now := time.Now()
	r.mu.Lock()
	if r.now != nil {
		now = r.now()
	}
	if r.closed {
		r.mu.Unlock()
		session := NewProviderSession()
		var once sync.Once
		return session, func() { once.Do(session.Close) }
	}
	// Expire stale entries first, but do not reserve capacity until we know this
	// is a cache miss. Reserving before the lookup would evict the very idle entry
	// being reacquired when capacity is one.
	toClose := r.pruneLocked(now, false)
	entry := r.entries[key]
	if entry == nil {
		toClose = append(toClose, r.pruneLocked(now, true)...)
		entry = &providerSessionEntry{session: NewProviderSession(), lastUsed: now}
		r.entries[key] = entry
	}
	entry.refs++
	entry.lastUsed = now
	r.mu.Unlock()
	closeProviderSessions(toClose)

	var once sync.Once
	return entry.session, func() {
		once.Do(func() { r.release(key, entry) })
	}
}

func (r *ProviderSessionRegistry) release(key string, want *providerSessionEntry) {
	if r == nil {
		return
	}
	r.mu.Lock()
	entry := r.entries[key]
	if entry != want {
		r.mu.Unlock()
		return
	}
	if entry.refs > 0 {
		entry.refs--
	}
	if r.now != nil {
		entry.lastUsed = r.now()
	} else {
		entry.lastUsed = time.Now()
	}
	toClose := r.pruneLocked(entry.lastUsed, false)
	r.mu.Unlock()
	closeProviderSessions(toClose)
}

// pruneLocked removes expired idle entries and, when reserve is true, enough
// least-recently-used idle entries to make room for one new key. Active entries
// are never candidates.
func (r *ProviderSessionRegistry) pruneLocked(now time.Time, reserve bool) []*ProviderSession {
	var removed []*ProviderSession
	for key, entry := range r.entries {
		if entry.refs == 0 && now.Sub(entry.lastUsed) >= r.idleTTL {
			delete(r.entries, key)
			removed = append(removed, entry.session)
		}
	}
	target := r.capacity
	if reserve {
		target--
	}
	for len(r.entries) > target {
		var oldestKey string
		var oldest *providerSessionEntry
		for key, entry := range r.entries {
			if entry.refs != 0 || (oldest != nil && !entry.lastUsed.Before(oldest.lastUsed)) {
				continue
			}
			oldestKey, oldest = key, entry
		}
		if oldest == nil {
			break
		}
		delete(r.entries, oldestKey)
		removed = append(removed, oldest.session)
	}
	return removed
}

// Close releases every retained idle or active session. Hosts should call it
// only after their requests have stopped; it is idempotent.
func (r *ProviderSessionRegistry) Close() {
	if r == nil {
		return
	}
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	r.closed = true
	sessions := make([]*ProviderSession, 0, len(r.entries))
	for _, entry := range r.entries {
		sessions = append(sessions, entry.session)
	}
	r.entries = nil
	r.mu.Unlock()
	closeProviderSessions(sessions)
}

func closeProviderSessions(sessions []*ProviderSession) {
	for _, session := range sessions {
		session.Close()
	}
}
