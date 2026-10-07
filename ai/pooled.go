package ai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/2found/2ai/ai/protocol"
)

// oauthTokenApplier is the private seam between pooledProvider and the wire
// clients that can take a per-request OAuth credential. Each subscription
// provider (claude-code, openai-codex, google-antigravity) implements it; the
// pooled wrapper refuses to wrap anything else so a token can never be dropped
// silently and the request sent with a stale static key.
//
// It returns a per-call CLONE of the wire client with the token installed,
// never mutating the shared instance: agentcore forks subagents that share the
// parent provider and dispatches parallel tool calls concurrently, so a token
// written onto the shared client could be overwritten by a sibling's acquire
// before the request is built — the request would then go out under another
// account's credential and Report would blame the wrong account.
type oauthTokenApplier interface {
	applyOAuthToken(tok OAuthToken) protocol.LLMProvider
}

// pooledProvider is an protocol.LLMProvider that draws a live OAuth access
// token from a TokenSource for every request instead of holding a static key.
// It exists because the subscription vendors authenticate a pool of accounts,
// not one credential: Acquire picks the next usable account (refreshing it when
// near expiry) and Report feeds the outcome back so a rate-limited or dead
// account rotates out.
//
// It deliberately does NOT implement protocol.KeyUpdater: the run loop's
// per-turn key refresh would overwrite the freshly acquired token with the
// provider row's sentinel key (OAuthPoolKey), which is never a real credential.
type pooledProvider struct {
	vendor       string
	sessionScope string
	inner        protocol.LLMProvider
	src          TokenSource
}

// maxOAuthAuthAttempts is a hard safety ceiling for one logical provider
// operation. A healthy pool normally succeeds in one or two attempts; the
// larger ceiling lets a deliberately large server-side account pool rotate
// past revoked siblings without creating an unbounded request loop.
const maxOAuthAuthAttempts = 64

type oauthAttemptState struct {
	attempts int
	seen     map[string]struct{}
	lastAuth error
}

func newOAuthAttemptState() *oauthAttemptState {
	return &oauthAttemptState{seen: make(map[string]struct{})}
}

func (s *oauthAttemptState) accept(tok OAuthToken) bool {
	if s.attempts >= maxOAuthAuthAttempts || strings.TrimSpace(tok.AccessToken) == "" {
		return false
	}
	// A refreshed bearer for the same account is a valid next attempt, while an
	// exact account+bearer cycle proves the source has no new credential to offer.
	identity := tok.AccountID + "\x00" + tok.AccessToken
	if _, duplicate := s.seen[identity]; duplicate {
		return false
	}
	s.seen[identity] = struct{}{}
	s.attempts++
	return true
}

func isOAuthAuthFailure(err error) bool {
	var providerErr *protocol.ProviderError
	if !errors.As(err, &providerErr) {
		return false
	}
	switch providerErr.Status {
	case http.StatusUnauthorized:
		return true
	case http.StatusForbidden:
		// Some vendors encode a transient concurrency cap as 403. Let the normal
		// backoff layer handle that instead of burning a healthy sibling account.
		return !isOAuthConcurrencyCap(err)
	default:
		return false
	}
}

func isOAuthConcurrencyCap(err error) bool {
	var providerErr *protocol.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Status != http.StatusForbidden {
		return false
	}
	return strings.Contains(strings.ToLower(providerErr.Message), "concurren")
}

func (p *pooledProvider) report(ctx context.Context, tok OAuthToken, err error) {
	if isOAuthConcurrencyCap(err) {
		return
	}
	p.src.Report(ctx, tok, err)
}

// newPooledProvider wraps inner so each Chat/Stream call acquires an account
// token from src. Both arguments are required: a nil source has nothing to
// draw from, and an inner client that cannot apply a token would send the
// request unauthenticated.
func newPooledProvider(vendor string, inner protocol.LLMProvider, src TokenSource, sessionScope ...string) (*pooledProvider, error) {
	if src == nil {
		return nil, fmt.Errorf("ai: provider %q requires a TokenSource (OAuth account pool)", vendor)
	}
	if _, ok := inner.(oauthTokenApplier); !ok {
		return nil, fmt.Errorf("ai: provider %q wire client cannot apply OAuth tokens", vendor)
	}
	scope := vendor
	if len(sessionScope) > 0 && sessionScope[0] != "" {
		scope = sessionScope[0]
	}
	return &pooledProvider{vendor: vendor, sessionScope: scope, inner: inner, src: src}, nil
}

// acquire pulls the next usable account token and returns a per-call clone of
// the inner wire client with that token installed. An empty pool or an
// exhausted pool surfaces as a provider error so the loop can escalate rather
// than retrying a request that has no credential at all.
func (p *pooledProvider) acquire(ctx context.Context) (protocol.LLMProvider, OAuthToken, error) {
	tok, err := p.src.Acquire(ctx)
	if err != nil {
		return nil, OAuthToken{}, protocol.NewProviderError(p.vendor, nil, err.Error())
	}
	return p.inner.(oauthTokenApplier).applyOAuthToken(tok), tok, nil
}

func (p *pooledProvider) Name() string        { return p.vendor }
func (p *pooledProvider) SupportsTools() bool { return p.inner.SupportsTools() }
func (p *pooledProvider) ModelCapabilities(model string) protocol.ModelCapabilities {
	return protocol.CapabilitiesOf(p.inner, model)
}

func (p *pooledProvider) Chat(ctx context.Context, req protocol.ChatRequest) (protocol.ChatResponse, error) {
	state := newOAuthAttemptState()
	for state.attempts < maxOAuthAuthAttempts {
		inner, tok, err := p.acquire(ctx)
		if err != nil {
			if state.lastAuth != nil {
				return protocol.ChatResponse{}, state.lastAuth
			}
			return protocol.ChatResponse{}, err
		}
		if !state.accept(tok) {
			if state.lastAuth != nil {
				return protocol.ChatResponse{}, state.lastAuth
			}
			return protocol.ChatResponse{}, protocol.NewProviderError(p.vendor, nil, "OAuth token source returned an empty or repeated credential")
		}
		bindOAuthProviderSession(req.ProviderSession, p.vendor, p.sessionScope, tok)
		resp, callErr := inner.Chat(ctx, req)
		p.report(ctx, tok, callErr)
		if callErr == nil || ctx.Err() != nil || !isOAuthAuthFailure(callErr) {
			return resp, callErr
		}
		state.lastAuth = callErr
	}
	return protocol.ChatResponse{}, state.lastAuth
}

func (p *pooledProvider) Stream(ctx context.Context, req protocol.ChatRequest) (<-chan protocol.ChatDelta, error) {
	state := newOAuthAttemptState()
	ch, tok, cancel, err := p.startStreamAttempt(ctx, req, state)
	if err != nil {
		return nil, err
	}
	out := make(chan protocol.ChatDelta, 16)
	go p.runAuthStream(ctx, req, state, tok, ch, cancel, out)
	return out, nil
}

// startStreamAttempt handles failures raised before a stream channel exists.
// Those attempts have emitted nothing and are always replay-safe; only typed
// 401/403 failures consume another credential.
func (p *pooledProvider) startStreamAttempt(ctx context.Context, req protocol.ChatRequest, state *oauthAttemptState) (<-chan protocol.ChatDelta, OAuthToken, context.CancelFunc, error) {
	for state.attempts < maxOAuthAuthAttempts {
		inner, tok, err := p.acquire(ctx)
		if err != nil {
			if state.lastAuth != nil {
				return nil, OAuthToken{}, nil, state.lastAuth
			}
			return nil, OAuthToken{}, nil, err
		}
		if !state.accept(tok) {
			if state.lastAuth != nil {
				return nil, OAuthToken{}, nil, state.lastAuth
			}
			return nil, OAuthToken{}, nil, protocol.NewProviderError(p.vendor, nil, "OAuth token source returned an empty or repeated credential")
		}
		bindOAuthProviderSession(req.ProviderSession, p.vendor, p.sessionScope, tok)
		attemptCtx, cancel := context.WithCancel(ctx)
		ch, callErr := inner.Stream(attemptCtx, req)
		if callErr == nil {
			return ch, tok, cancel, nil
		}
		cancel()
		p.report(ctx, tok, callErr)
		if ctx.Err() != nil || !isOAuthAuthFailure(callErr) {
			return nil, OAuthToken{}, nil, callErr
		}
		state.lastAuth = callErr
	}
	return nil, OAuthToken{}, nil, state.lastAuth
}

func drainChatDeltas(ch <-chan protocol.ChatDelta) {
	go func() {
		for range ch {
		}
	}()
}

// runAuthStream buffers replay-safe metadata until an attempt succeeds. A
// 401/403 before content rotates credentials and discards that attempt. The
// first content delta is the commit boundary: after it, errors are forwarded
// and never replayed, matching agentcore's visible-stream contract.
func (p *pooledProvider) runAuthStream(ctx context.Context, req protocol.ChatRequest, state *oauthAttemptState, tok OAuthToken, ch <-chan protocol.ChatDelta, cancel context.CancelFunc, out chan<- protocol.ChatDelta) {
	defer close(out)
	defer func() { cancel() }()
	committed := false
	var buffered []protocol.ChatDelta

	send := func(delta protocol.ChatDelta) bool {
		select {
		case out <- delta:
			return true
		case <-ctx.Done():
			return false
		}
	}
	flush := func() bool {
		for _, delta := range buffered {
			if !send(delta) {
				return false
			}
		}
		buffered = buffered[:0]
		return true
	}

	for {
		select {
		case <-ctx.Done():
			// Keep cancellation visible to direct Provider.Stream consumers. The
			// non-blocking send avoids leaking this goroutine if the caller has
			// abandoned a full output buffer; agentcore also checks ctx after close.
			select {
			case out <- protocol.ChatDelta{Done: true, Err: ctx.Err()}:
			default:
			}
			return
		case delta, open := <-ch:
			if !open {
				cancel()
				if ctx.Err() == nil {
					p.report(ctx, tok, nil)
				}
				flush()
				return
			}
			if delta.Err != nil {
				// An error delta is terminal for this attempt. Stop and drain it before
				// Report performs any refresh I/O, so a chatty/broken provider cannot
				// block behind its own full channel while credentials rotate.
				stale := ch
				cancel()
				drainChatDeltas(stale)
				p.report(ctx, tok, delta.Err)
				if !committed && isOAuthAuthFailure(delta.Err) {
					state.lastAuth = delta.Err
					next, nextTok, nextCancel, err := p.startStreamAttempt(ctx, req, state)
					if err == nil {
						buffered = buffered[:0]
						ch, tok, cancel = next, nextTok, nextCancel
						continue
					}
				}
				if !flush() {
					return
				}
				send(delta)
				return
			}
			if !committed && delta.ContentDelta == "" {
				buffered = append(buffered, delta)
				continue
			}
			if !committed {
				if !flush() {
					return
				}
				committed = true
			}
			if !send(delta) {
				return
			}
		}
	}
}
