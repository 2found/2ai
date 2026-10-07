package ai

import (
	"errors"
	"net/http"
	"strings"
	"sync"

	"github.com/2found/2ai/ai/protocol"
)

const openAIAdaptiveStatePrefix = "openai-chat-adaptive\x00"

// openAIAdaptiveState holds endpoint-scoped lessons, not account-scoped state:
// a gateway that rejects prompt_cache_key keeps rejecting it after an API key
// or OAuth account changes. It therefore deliberately does not implement
// protocol.AccountScopedProviderState.
type openAIAdaptiveState struct {
	mu             sync.RWMutex
	models         map[string]protocol.ModelCapabilities
	maxTokenFields map[string]string
	closed         bool
}

func newOpenAIAdaptiveState() protocol.ProviderSessionState {
	return &openAIAdaptiveState{
		models: make(map[string]protocol.ModelCapabilities), maxTokenFields: make(map[string]string),
	}
}

func (s *openAIAdaptiveState) Close() {
	s.mu.Lock()
	s.closed = true
	s.models = nil
	s.maxTokenFields = nil
	s.mu.Unlock()
}

func (s *openAIAdaptiveState) apply(req protocol.ChatRequest, maxTokenField string) (protocol.ChatRequest, string) {
	s.mu.RLock()
	caps := s.models[req.Model]
	if learned := s.maxTokenFields[req.Model]; learned != "" {
		maxTokenField = learned
	}
	s.mu.RUnlock()
	if caps.ReasoningEffort == protocol.CapabilityUnsupported {
		req.ReasoningEffort = ""
	}
	if caps.StructuredOutput == protocol.CapabilityUnsupported {
		req.OutputSchema = nil
	}
	if caps.PromptCaching == protocol.CapabilityUnsupported {
		req.CacheKey = ""
		req.CacheRetention = ""
	}
	return req, maxTokenField
}

func (s *openAIAdaptiveState) remember(model string, learned protocol.ModelCapabilities) {
	if learned == (protocol.ModelCapabilities{}) {
		return
	}
	s.mu.Lock()
	if !s.closed {
		s.models[model] = s.models[model].Overlay(learned)
	}
	s.mu.Unlock()
}

func (s *openAIAdaptiveState) rememberMaxTokenField(model, field string) {
	if field != "max_tokens" && field != "max_completion_tokens" {
		return
	}
	s.mu.Lock()
	if !s.closed {
		s.maxTokenFields[model] = field
	}
	s.mu.Unlock()
}

func (p *OpenAIProvider) adaptiveRequest(req protocol.ChatRequest) (*openAIAdaptiveState, protocol.ChatRequest, string) {
	maxTokenField := p.maxTokensField()
	if req.ProviderSession == nil {
		return nil, req, maxTokenField
	}
	key := openAIAdaptiveStatePrefix + p.Name() + "\x00" + strings.TrimRight(p.BaseURL, "/")
	state, _ := req.ProviderSession.State(key, newOpenAIAdaptiveState).(*openAIAdaptiveState)
	if state == nil {
		return nil, req, maxTokenField
	}
	req, maxTokenField = state.apply(req, maxTokenField)
	return state, req, maxTokenField
}

func alternateMaxTokenField(field string, req protocol.ChatRequest, err error) (string, bool) {
	if req.MaxTokens <= 0 {
		return field, false
	}
	var providerErr *protocol.ProviderError
	if !errors.As(err, &providerErr) || (providerErr.Status != http.StatusBadRequest && providerErr.Status != http.StatusUnprocessableEntity) {
		return field, false
	}
	message := strings.ToLower(providerErr.Message)
	if !containsAny(message, "unsupported", "not supported", "unknown parameter", "unknown field", "unrecognized", "unexpected keyword", "extra inputs are not permitted") {
		return field, false
	}
	switch field {
	case "max_completion_tokens":
		if strings.Contains(message, "max_completion_tokens") {
			return "max_tokens", true
		}
	default:
		if strings.Contains(message, "max_tokens") {
			return "max_completion_tokens", true
		}
	}
	return field, false
}

// withoutRejectedOpenAIHint recognizes only an explicit unsupported-parameter
// response. It never treats an arbitrary 400 as capability discovery: in
// particular, an invalid JSON schema stays a caller-visible error instead of
// silently weakening the request contract.
func withoutRejectedOpenAIHint(req protocol.ChatRequest, err error) (protocol.ChatRequest, protocol.ModelCapabilities, bool) {
	var pe *protocol.ProviderError
	if !errors.As(err, &pe) || (pe.Status != http.StatusBadRequest && pe.Status != http.StatusUnprocessableEntity) {
		return req, protocol.ModelCapabilities{}, false
	}
	message := strings.ToLower(pe.Message)
	unsupported := strings.Contains(message, "unsupported") ||
		strings.Contains(message, "not supported") ||
		strings.Contains(message, "unknown parameter") ||
		strings.Contains(message, "unknown field") ||
		strings.Contains(message, "unrecognized") ||
		strings.Contains(message, "unexpected keyword") ||
		strings.Contains(message, "extra inputs are not permitted")
	if !unsupported {
		return req, protocol.ModelCapabilities{}, false
	}
	if req.ReasoningEffort != "" && containsAny(message, "reasoning_effort", "reasoning effort") {
		req.ReasoningEffort = ""
		return req, protocol.ModelCapabilities{ReasoningEffort: protocol.CapabilityUnsupported}, true
	}
	if req.CacheKey != "" && containsAny(message, "prompt_cache_key", "prompt cache", "prompt caching") {
		req.CacheKey, req.CacheRetention = "", ""
		return req, protocol.ModelCapabilities{PromptCaching: protocol.CapabilityUnsupported}, true
	}
	if req.OutputSchema != nil && containsAny(message, "response_format", "structured output", "json_schema") {
		req.OutputSchema = nil
		return req, protocol.ModelCapabilities{StructuredOutput: protocol.CapabilityUnsupported}, true
	}
	return req, protocol.ModelCapabilities{}, false
}

func containsAny(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

// oauthAccountBinding is stored in the provider session so a newly-built
// pooledProvider can still detect that this logical conversation moved to a
// sibling account between runs. It owns no external resource.
type oauthAccountBinding struct {
	mu      sync.Mutex
	account string
	closed  bool
}

func (s *oauthAccountBinding) Close() {
	s.mu.Lock()
	s.closed = true
	s.account = ""
	s.mu.Unlock()
}

func (s *oauthAccountBinding) switched(account string) bool {
	if account == "" {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return false
	}
	changed := s.account != "" && s.account != account
	s.account = account
	return changed
}

func bindOAuthProviderSession(session *protocol.ProviderSession, vendor, scope string, tok OAuthToken) {
	if session == nil {
		return
	}
	state, _ := session.State("oauth-account\x00"+vendor+"\x00"+scope, func() protocol.ProviderSessionState {
		return &oauthAccountBinding{}
	}).(*oauthAccountBinding)
	if state != nil && state.switched(tok.AccountID) {
		session.ResetAccountScoped()
	}
}
