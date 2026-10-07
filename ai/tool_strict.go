package ai

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"

	"github.com/2found/2ai/ai/protocol"
)

const strictToolsStatePrefix = "strict-tools-adaptive\x00"
const maxStrictToolModelLessons = 64

// strictToolsState remembers an endpoint/model wire incompatibility only for a
// logical provider session. It owns no transport and is endpoint-scoped rather
// than credential-scoped, so an OAuth account switch need not relearn it.
type strictToolsState struct {
	mu       sync.RWMutex
	disabled map[string]bool
	closed   bool
}

func newStrictToolsState() protocol.ProviderSessionState {
	return &strictToolsState{disabled: make(map[string]bool)}
}

func (s *strictToolsState) Close() {
	s.mu.Lock()
	s.closed = true
	s.disabled = nil
	s.mu.Unlock()
}

func (s *strictToolsState) isDisabled(model string) bool {
	s.mu.RLock()
	disabled := s.disabled[model]
	s.mu.RUnlock()
	return disabled
}

func (s *strictToolsState) disable(model string) {
	s.mu.Lock()
	if !s.closed && (s.disabled[model] || len(s.disabled) < maxStrictToolModelLessons) {
		s.disabled[model] = true
	}
	s.mu.Unlock()
}

func prepareStrictTools(req protocol.ChatRequest, providerName, baseURL string, dialect toolSchemaDialect) (*strictToolsState, protocol.ChatRequest) {
	if req.ProviderSession == nil {
		return nil, req
	}
	key := strictToolsStatePrefix + providerName + "\x00" + strings.TrimRight(baseURL, "/") + "\x00" + string(rune(dialect))
	state, _ := req.ProviderSession.State(key, newStrictToolsState).(*strictToolsState)
	if state != nil && state.isDisabled(req.Model) {
		return state, withoutStrictTools(req)
	}
	return state, req
}

func withoutStrictTools(req protocol.ChatRequest) protocol.ChatRequest {
	changed := false
	tools := make([]protocol.ToolSchema, len(req.Tools))
	copy(tools, req.Tools)
	for i := range tools {
		if tools[i].Strict != protocol.ToolStrictDefault {
			tools[i].Strict = protocol.ToolStrictDefault
			changed = true
		}
	}
	if changed {
		req.Tools = tools
	}
	return req
}

func toolStrictFieldEmitted(req protocol.ChatRequest, dialect toolSchemaDialect) bool {
	for _, tool := range req.Tools {
		_, strict := projectedToolParameters(tool, dialect)
		if strict != nil {
			return true
		}
	}
	return false
}

// shouldRetryWithoutStrictTools recognizes only an HTTP schema/strictness
// rejection. Authentication, rate limits, transport failures, and arbitrary
// validation errors never trigger a weakened replay.
func shouldRetryWithoutStrictTools(req protocol.ChatRequest, dialect toolSchemaDialect, err error) bool {
	if !toolStrictFieldEmitted(req, dialect) {
		return false
	}
	var providerErr *protocol.ProviderError
	if !errors.As(err, &providerErr) || (providerErr.Status != http.StatusBadRequest && providerErr.Status != http.StatusUnprocessableEntity) {
		return false
	}
	return strictToolsRejectionMessage(providerErr.Message)
}

func strictToolsRejectionMessage(value string) bool {
	message := strings.ToLower(value)
	return containsAny(message,
		"wrong_api_format",
		"mixed values for 'strict'",
		"mixed values for \"strict\"",
		"tool parameters schema",
		"tool parameter schema",
		"invalid schema for function",
		"strict tools",
		"strict tool",
	) || (strings.Contains(message, "strict") && containsAny(message,
		"tool", "function", "schema", "unknown", "unsupported", "not supported", "unrecognized", "extra inputs are not permitted",
	)) || (containsAny(message, "structured output", "structured_output", "structured outputs") && containsAny(message,
		"unsupported", "not supported", "not available", "not enabled",
	)) || (strings.Contains(message, "compiled grammar") && strings.Contains(message, "too large")) ||
		(strings.Contains(message, "schema") && strings.Contains(message, "too complex"))
}

func rememberStrictToolsRejected(state *strictToolsState, model string) {
	if state != nil {
		state.disable(model)
	}
}

// preflightStrictStream observes at most the first provider delta. A strict
// rejection reported inside an HTTP-200 SSE envelope can still be retried
// because no output has crossed the provider boundary. Any content, tool call,
// or ordinary error is put back in order and permanently commits the attempt.
func preflightStrictStream(ctx context.Context, req protocol.ChatRequest, dialect toolSchemaDialect, ch <-chan protocol.ChatDelta) (<-chan protocol.ChatDelta, error, bool) {
	if !toolStrictFieldEmitted(req, dialect) {
		return ch, nil, false
	}
	select {
	case first, ok := <-ch:
		if !ok {
			return ch, nil, false
		}
		if first.Err != nil && shouldRetryWithoutStrictTools(req, dialect, first.Err) {
			return nil, first.Err, true
		}
		out := make(chan protocol.ChatDelta, 16)
		out <- first
		go func() {
			defer close(out)
			for delta := range ch {
				out <- delta
			}
		}()
		return out, nil, false
	case <-ctx.Done():
		return nil, ctx.Err(), false
	}
}
