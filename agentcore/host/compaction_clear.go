package host

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
)

// ToolResultClearing clears old tool results from the request view before a
// summary is paid for, the analogue of the Messages API's clear_tool_uses
// context edit. Tool results are most of a long run's context and stop
// mattering long before the reasoning around them does.
//
// The clearing boundary is persisted (Compactor.Cleared), so a cleared result
// is rewritten to the same bytes on every request and the provider's prompt
// cache still reads the prefix. Moving the boundary rewrites every request
// from the first newly cleared result, so it moves only when that saves at
// least ClearAtLeast, and only once the request is past Trigger or the cache
// has already gone cold.
type ToolResultClearing struct {
	// Disabled turns clearing off; a persisted boundary is then ignored.
	Disabled bool
	// Trigger is the estimated request size past which the boundary may move
	// while the prompt cache is warm. Zero selects half the Budget.
	Trigger int
	// Keep is how many of the most recent tool results stay verbatim. Zero
	// selects DefaultKeepToolResults.
	Keep int
	// ClearAtLeast is the smallest estimated saving worth a boundary move.
	// Zero selects a twentieth of the Budget, and never less than 1024.
	ClearAtLeast int
	// ExcludeTools are never cleared, such as a tool whose result is an
	// instruction the model keeps following.
	ExcludeTools []string
	// CacheTTL is how long the provider keeps a prompt-cache entry. Once the
	// last response is older than this the cache is cold, and the boundary
	// moves whenever that saves ClearAtLeast. Zero selects five minutes.
	CacheTTL time.Duration
}

// DefaultKeepToolResults is how many recent tool results clearing keeps.
const DefaultKeepToolResults = 5

// A result smaller than this is not worth clearing: its placeholder costs
// nearly as much.
const minClearTokens = 128

// A cleared result at least this large is archived, so its placeholder can
// name a locator instead of asking for the call to be repeated.
const minArchiveTokens = 512

func (p ToolResultClearing) resolved(budget int) ToolResultClearing {
	if p.Trigger <= 0 || (budget > 0 && p.Trigger > budget) {
		p.Trigger = budget / 2
	}
	if p.Keep <= 0 {
		p.Keep = DefaultKeepToolResults
	}
	if p.ClearAtLeast <= 0 {
		p.ClearAtLeast = max(1024, budget/20)
	}
	if p.CacheTTL <= 0 {
		p.CacheTTL = 5 * time.Minute
	}
	return p
}

// ClearState is the persisted clearing boundary. Tool results of the
// authoritative transcript before Boundary are cleared in every request
// view. At is when the boundary last moved, in Unix milliseconds: provider
// usage reported before it measured the uncleared view. Locators holds the
// archive locator of a cleared result by its transcript index.
type ClearState struct {
	Boundary int            `json:"boundary"`
	At       int64          `json:"at"`
	Locators map[int]string `json:"locators,omitempty"`
}

// Cleared returns the persisted clearing boundary, or nil when nothing has
// been cleared. Persist it with Checkpoint.
func (c *Compactor) Cleared() json.RawMessage {
	if c.cleared.Boundary <= 0 {
		return nil
	}
	raw, _ := json.Marshal(c.cleared)
	return raw
}

// RestoreCleared reinstates a boundary returned by Cleared.
func (c *Compactor) RestoreCleared(raw json.RawMessage) error {
	var state ClearState
	if err := json.Unmarshal(raw, &state); err != nil || state.Boundary < 0 || state.At < 0 {
		return errors.New("invalid native clearing boundary")
	}
	c.cleared, c.rewrites = state, nil
	return nil
}

// clear applies the clearing boundary to the tail of view, whose messages
// view[head:] are authoritative messages base onward, and moves the boundary
// when the policy allows. cold says the prompt cache cannot be read anyway:
// it has expired, or this request already rewrites the prefix. Unchanged
// messages are shared with the input.
func (c *Compactor) clear(ctx context.Context, view []json.RawMessage, head, base int, policy CompactionPolicy, cold bool) []json.RawMessage {
	settings := policy.Clearing.resolved(policy.Budget)
	if settings.Disabled || policy.Budget <= 0 {
		return view
	}
	facts := c.memo.of(view)
	var results []int // clearable tail results, oldest first
	for i := head; i < len(view); i++ {
		if facts[i].role == "toolResult" && !slices.Contains(settings.ExcludeTools, facts[i].toolName) {
			results = append(results, i)
		}
	}
	auth := func(i int) int { return base + i - head }
	apply := func() []json.RawMessage {
		out, cloned := view, false
		for _, i := range results {
			if auth(i) >= c.cleared.Boundary || facts[i].tokens < minClearTokens {
				continue
			}
			if !cloned {
				out, cloned = slices.Clone(view), true
			}
			out[i] = c.clearedResult(view, i, facts, c.cleared.Locators[auth(i)])
		}
		return out
	}
	out := apply()
	if len(results) > settings.Keep {
		boundary := auth(results[len(results)-settings.Keep])
		saving := 0
		for _, i := range results {
			if a := auth(i); a >= c.cleared.Boundary && a < boundary && facts[i].tokens >= minClearTokens {
				saving += facts[i].tokens - minClearTokens/2
			}
		}
		if boundary > c.cleared.Boundary && saving >= settings.ClearAtLeast && (cold || c.cacheCold(facts, settings.CacheTTL) || factTokens(c.memo.of(out)) > settings.Trigger) {
			for _, i := range results {
				if a := auth(i); a >= c.cleared.Boundary && a < boundary && facts[i].tokens >= minArchiveTokens && !facts[i].isError && policy.Archive != nil {
					if locator, err := policy.Archive(ctx, facts[i].toolName, facts[i].toolCallID, resultText(view[i])); err == nil && locator != "" {
						if c.cleared.Locators == nil {
							c.cleared.Locators = map[int]string{}
						}
						c.cleared.Locators[a] = locator
					}
				}
			}
			c.cleared.Boundary, c.cleared.At, c.rewrites = boundary, c.clock().UnixMilli(), nil
			out = apply()
		}
	}
	for a := range c.cleared.Locators {
		if a < base {
			delete(c.cleared.Locators, a)
		}
	}
	return out
}

// cacheCold reports whether the newest response is older than ttl. A
// transcript without timestamps is never assumed cold.
func (c *Compactor) cacheCold(facts []*messageFacts, ttl time.Duration) bool {
	for i := len(facts) - 1; i >= 0; i-- {
		if facts[i].role == "assistant" && facts[i].timestamp > 0 {
			return c.clock().Sub(time.UnixMilli(facts[i].timestamp)) > ttl
		}
	}
	return false
}

// clearedResult is view[i] with its content replaced by a placeholder that
// names the call and how to recover the result. The bytes depend only on the
// message, its call and its locator, so every request repeats them exactly.
func (c *Compactor) clearedResult(view []json.RawMessage, i int, facts []*messageFacts, locator string) json.RawMessage {
	key := rewriteKey{mark: c.memo.mark(view[i]), locator: locator}
	if raw, ok := c.rewrites[key]; ok {
		return raw
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(view[i], &fields) != nil {
		return view[i]
	}
	f := facts[i]
	var args json.RawMessage
	for j := i - 1; j >= 0 && args == nil; j-- {
		args = facts[j].callArgs[f.toolCallID]
	}
	text, images, _ := renderContent(fields["content"])
	call := fmt.Sprintf("%s(%s)", f.toolName, renderArguments(args))
	var placeholder string
	switch {
	case f.isError:
		placeholder = fmt.Sprintf("[Earlier tool error cleared to save context: %s failed: %s]", call, truncateMiddle(strings.TrimSpace(text), 300))
	case locator != "":
		placeholder = fmt.Sprintf("[Earlier tool result cleared to save context: %s returned %d bytes. The full result is saved: read it with read_spill, locator %q, if you need it again.]", call, len(text), locator)
	default:
		placeholder = fmt.Sprintf("[Earlier tool result cleared to save context: %s returned %d bytes. Call it again if you need the result.]", call, len(text))
	}
	if images > 0 {
		placeholder = strings.TrimSuffix(placeholder, "]") + fmt.Sprintf(" %d image(s) were cleared with it.]", images)
	}
	fields["content"], _ = json.Marshal([]map[string]string{{"type": "text", "text": placeholder}})
	fields["agentrayContextReducedAt"], _ = json.Marshal(c.cleared.At)
	raw, _ := json.Marshal(fields)
	if c.rewrites == nil {
		c.rewrites = map[rewriteKey]json.RawMessage{}
	}
	c.rewrites[key] = raw
	return raw
}

type rewriteKey struct {
	mark    uint64
	locator string
}

func resultText(raw json.RawMessage) string {
	var m struct{ Content json.RawMessage }
	_ = json.Unmarshal(raw, &m)
	text, _, _ := renderContent(m.Content)
	return text
}
