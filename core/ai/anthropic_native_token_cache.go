package ai

import "sync"

type anthropicAccessToken struct {
	token     string
	expiresAt *float64
}

type anthropicTokenRefresh struct {
	done  chan struct{}
	token anthropicAccessToken
	err   error
}

// Mirrors the SDK token cache: advisory refresh does not block a usable token;
// mandatory callers join a pending refresh, while invalidation forces a new one.
type anthropicTokenCache struct {
	mu                sync.Mutex
	provider          func(bool) (anthropicAccessToken, error)
	now               func() float64
	onAdvisoryError   func(error)
	cached            *anthropicAccessToken
	pending           *anthropicTokenRefresh
	nextForce         bool
	lastAdvisoryError float64
}

func (c *anthropicTokenCache) getToken() (string, error) {
	c.mu.Lock()
	force := c.nextForce
	c.nextForce = false
	if !force && c.cached != nil {
		cached := *c.cached
		remaining := 0.0
		if cached.expiresAt != nil {
			remaining = *cached.expiresAt - c.now()
		}
		if cached.expiresAt == nil || remaining > 120 {
			c.mu.Unlock()
			return cached.token, nil
		}
		if remaining > 30 {
			if c.pending == nil && c.now()-c.lastAdvisoryError >= 5 {
				c.startRefreshLocked(false, true)
			}
			c.mu.Unlock()
			return cached.token, nil
		}
	}
	pending := c.pending
	if pending == nil || force {
		pending = c.startRefreshLocked(force, false)
	}
	c.mu.Unlock()
	<-pending.done
	return pending.token.token, pending.err
}

func (c *anthropicTokenCache) startRefreshLocked(force, advisory bool) *anthropicTokenRefresh {
	pending := &anthropicTokenRefresh{done: make(chan struct{})}
	c.pending = pending
	go func() {
		token, err := c.provider(force)
		c.mu.Lock()
		if err == nil {
			c.cached = &token
		}
		// The source clears the slot when either overlapping forced refresh
		// settles, even if another refresh was started more recently.
		c.pending = nil
		if advisory && err != nil {
			c.lastAdvisoryError = c.now()
		}
		pending.token, pending.err = token, err
		close(pending.done)
		c.mu.Unlock()
		if advisory && err != nil && c.onAdvisoryError != nil {
			c.onAdvisoryError(err)
		}
	}()
	return pending
}

func (c *anthropicTokenCache) invalidate() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cached = nil
	c.nextForce = true
}
