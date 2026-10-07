package ai

import (
	"context"
	"strings"
)

// OAuth vendor ids: subscription providers whose credential is a pool of
// OAuth accounts (many logins per provider row) rather than a single API key.
// The vendor id is the provider row's `vendor` column and the wire client's
// identity — it is NOT the API-key vendor it resembles on the wire
// (claude-code speaks the Anthropic Messages API but authenticates with a
// Bearer grant, not x-api-key).
const (
	VendorClaudeCode        = "claude-code"
	VendorOpenAICodex       = "openai-codex"
	VendorGoogleAntigravity = "google-antigravity"
	VendorDevin             = "devin"
	VendorXaiOAuth          = "xai-oauth"
)

// OAuthPoolKey is the sentinel ResolveWorkspaceRun puts in the per-tier key
// map for an OAuth vendor. It is never sent on the wire — pooled providers
// ignore their static key and pull a live access token from their TokenSource
// per request — but it must be non-empty so the run path's "flash key
// configured" gate treats a pooled provider as configured.
const OAuthPoolKey = "oauth-pool"

// IsOAuthVendor reports whether vendor is one of the subscription/OAuth pool
// vendors. Such a provider row has no api_key; it owns a pool of accounts in
// workspace_provider_accounts.
func IsOAuthVendor(vendor string) bool {
	return NormalizeOAuthVendor(vendor) != ""
}

// OAuthToken is one acquired account credential, handed to a wire client for a
// single request. The extra identity fields ride along because the wires need
// them: Codex sends AccountID as the chatgpt-account-id header, Antigravity
// puts ProjectID in the request envelope.
type OAuthToken struct {
	// AccountID is the workspace_provider_accounts row id — the handle Report
	// uses to block/disable the account that served a failed request.
	AccountID string
	// AccessToken is the OAuth access token sent as the Bearer credential.
	AccessToken string
	// ProviderAccountID is the vendor-side account id (Codex chatgpt_account_id).
	ProviderAccountID string
	// ProjectID is the Cloud Code Assist project (Antigravity only).
	ProjectID string
	// Email is the account's login email, for diagnostics.
	Email string
}

// TokenSource is the seam between a pooled (multi-account OAuth) provider and
// the account store. Acquire picks the next usable account — refreshing its
// access token when near expiry — and Report feeds the request's outcome back
// so a rate-limited or dead account is rotated out of the pool.
type TokenSource interface {
	// Acquire returns the next usable account's token. It MUST return an error
	// (never an empty AccessToken) when no account can serve — the caller
	// surfaces that as the provider error.
	Acquire(ctx context.Context) (OAuthToken, error)
	// Report records the outcome of a request made with tok. err nil means the
	// account served fine (clears nothing, updates last_used_at); a rate-limit
	// or auth failure marks the account so the next Acquire skips it.
	Report(ctx context.Context, tok OAuthToken, err error)
}

// NormalizeOAuthVendor folds aliases onto the canonical vendor ids. Exported
// because the oauth package's descriptor lookup must agree with this table —
// a second copy has already drifted once (missing case-fold, wrong zero value).
func NormalizeOAuthVendor(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "claude-code", "claude_code", "claudecode", "anthropic-oauth", "anthropic-claude-code":
		return VendorClaudeCode
	case "openai-codex", "codex", "chatgpt", "openai-oauth":
		return VendorOpenAICodex
	case "google-antigravity", "antigravity":
		return VendorGoogleAntigravity
	case "xai-oauth", "grok-oauth":
		return VendorXaiOAuth
	case "devin", "devin-agent", "codeium-devin", "windsurf-devin":
		return VendorDevin
	}
	return ""
}
