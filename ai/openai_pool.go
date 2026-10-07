package ai

import "github.com/2found/2ai/ai/protocol"

// Only the subscription wrapper accepts pooled tokens; ordinary API clients keep their static-key contract.
type xaiOAuthProvider struct{ *OpenAIProvider }

func (p *xaiOAuthProvider) applyOAuthToken(tok OAuthToken) protocol.LLMProvider {
	return &OpenAIProvider{APIKey: tok.AccessToken, BaseURL: p.BaseURL, Compat: p.Compat, HTTP: p.HTTP, StreamHTTP: p.StreamHTTP, Vendor: p.Vendor}
}
