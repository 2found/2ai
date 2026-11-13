package protocol

import "sync"

// ProviderSessionState is one provider-owned mutable record retained for the
// lifetime of a logical agent conversation. Implementations must make Close
// idempotent: a registry eviction and an explicit shutdown may race at process
// teardown, and neither may leak a transport or panic the host.
//
// The interface intentionally says nothing about serialization. Provider state
// commonly contains live sockets, server-side response handles, or learned
// endpoint quirks; keeping it an optimization rather than durable truth lets the
// same provider run in a laptop process and across stateless server replicas.
type ProviderSessionState interface {
	Close()
}

// AccountScopedProviderState is the optional, narrow reset seam for state that
// belongs to the credential which created it (for example a server-side response
// id). Endpoint-scoped lessons should not implement this interface, so an OAuth
// account rotation preserves them.
type AccountScopedProviderState interface {
	ProviderSessionState
	ResetAccountScoped()
}

// ProviderSession is a concurrency-safe bag of provider-private state for one
// logical conversation. Providers namespace their own keys and own the concrete
// value types; agentcore owns only lifecycle and account-rotation semantics.
type ProviderSession struct {
	mu     sync.RWMutex
	states map[string]ProviderSessionState
	closed bool
}

// NewProviderSession creates an unregistered session. Embedded consumers that
// already own conversation lifetime can use this directly; server runtimes
// normally lease one from ProviderSessionRegistry instead.
func NewProviderSession() *ProviderSession {
	return &ProviderSession{states: make(map[string]ProviderSessionState)}
}

// State returns the value at key, creating it exactly once. A nil factory, an
// empty key, a nil factory result, or a closed session returns nil. The factory
// runs under the session lock so two first requests cannot create parallel live
// transports and discard one without closing it.
func (s *ProviderSession) State(key string, create func() ProviderSessionState) ProviderSessionState {
	if s == nil || key == "" || create == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if state := s.states[key]; state != nil {
		return state
	}
	state := create()
	if state != nil {
		s.states[key] = state
	}
	return state
}

// ResetAccountScoped resets only records that explicitly declare themselves
// credential-bound. Endpoint and conversation lessons remain warm.
func (s *ProviderSession) ResetAccountScoped() {
	if s == nil {
		return
	}
	s.mu.RLock()
	states := make([]AccountScopedProviderState, 0, len(s.states))
	if !s.closed {
		for _, state := range s.states {
			if scoped, ok := state.(AccountScopedProviderState); ok {
				states = append(states, scoped)
			}
		}
	}
	s.mu.RUnlock()
	for _, state := range states {
		state.ResetAccountScoped()
	}
}

// Close releases every provider-owned record exactly once and makes the session
// reject future state creation. Calls into provider code happen outside the map
// lock so a provider's Close implementation cannot deadlock by re-entering its
// own bookkeeping.
func (s *ProviderSession) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	states := make([]ProviderSessionState, 0, len(s.states))
	for _, state := range s.states {
		states = append(states, state)
	}
	s.states = nil
	s.mu.Unlock()
	for _, state := range states {
		state.Close()
	}
}
