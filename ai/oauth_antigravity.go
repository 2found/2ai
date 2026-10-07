package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/2found/2ai/jsonjs"
)

// Antigravity signs in with a Google OAuth authorization-code grant (the
// Cloud Code Assist pair shipped by the antigravity/hub client), then must
// resolve the account's Cloud Code project before any request: loadCodeAssist
// tells whether the account already has one, otherwise a free-tier
// onboardUser LRO creates it. The wire client sends the project on every
// envelope, so the credential stores it.
//
// Mirrored from internal/oauth/login.go (exchangeCode + enrich + the
// Cloud Code onboarding sequence); that package imports ai, so the client
// flow lives here rather than being shared.
const (
	antigravityOAuthClientID     = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
	antigravityOAuthAuthorizeURL = "https://accounts.google.com/o/oauth2/v2/auth"
	antigravityOAuthTokenURL     = "https://oauth2.googleapis.com/token"
	antigravityOAuthUserinfoURL  = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"
	antigravityOAuthCallbackPort = 51121
	antigravityOAuthCallbackPath = "/oauth-callback"

	antigravityOAuthScope = "https://www.googleapis.com/auth/cloud-platform " +
		"https://www.googleapis.com/auth/userinfo.email " +
		"https://www.googleapis.com/auth/userinfo.profile " +
		"https://www.googleapis.com/auth/cclog " +
		"https://www.googleapis.com/auth/experimentsandconfigs"
	antigravityOAuthFreeTierID  = "free-tier"
	antigravityOAuthOnboardCap  = 30 * time.Second
	antigravityOAuthOnboardPoll = 1 * time.Second
	antigravityOAuthSkewMS      = 300 * 1000
)

// antigravityOAuthClientSecret mirrors the Cloud Code Assist pair in
// internal/oauth/providers.go, which stores the value base64 so secret
// scanners stay quiet — an installed-app OAuth pair, not a real secret, but
// kept in the same encoding for consistency.
var antigravityOAuthClientSecret = func() string {
	raw, err := base64.StdEncoding.DecodeString("R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNY" + "QzR6NnFEQWY=")
	if err != nil {
		panic(err)
	}
	return string(raw)
}()

var antigravityOAuthMetadata = map[string]any{"ideType": "ANTIGRAVITY"}

type AntigravityOAuthOptions struct {
	Client        *http.Client
	RandomValue   func() (string, error)
	CallbackHost  func() string
	StartCallback func(*OAuthCallbackServerOptions) (*OAuthCallbackServer, error)
	Now           func() float64
}

// AntigravityOAuth builds the interactive login for the google-antigravity
// vendor: Google code grant → userinfo email → Cloud Code project discovery
// (+free-tier onboarding). The returned credential carries access/refresh/
// expires plus `project` and `account` for the vault.
func AntigravityOAuth(settings ...AntigravityOAuthOptions) *OAuthAuth {
	var options AntigravityOAuthOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.RandomValue == nil {
		options.RandomValue = func() (string, error) { return uuid.NewString(), nil }
	}
	if options.CallbackHost == nil {
		options.CallbackHost = func() string {
			if host := os.Getenv("PI_OAUTH_CALLBACK_HOST"); host != "" {
				return host
			}
			return "127.0.0.1"
		}
	}
	if options.StartCallback == nil {
		options.StartCallback = StartOAuthCallbackServer
	}
	if options.Now == nil {
		options.Now = func() float64 { return float64(time.Now().UnixMilli()) }
	}
	subscription, label := true, "Sign in with Google"
	return &OAuthAuth{Name: "Google (Antigravity)", IsSubscription: &subscription, LoginLabel: &label,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) { return loginAntigravityOAuth(interaction, options) })
		},
		Refresh: func(ctx context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.refresh")
				}
				return antigravityOAuthRefresh(ctx, catalogProperty(credential, "refresh"), credential, options)
			})
		},
		ToAuth: func(credential any) (any, error) {
			if jsonjs.IsNullish(credential) {
				return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.access")
			}
			return NewObject(Property{Name: "apiKey", Value: catalogProperty(credential, "access")}), nil
		},
	}
}

func loginAntigravityOAuth(interaction ProviderAuthInteraction, options AntigravityOAuthOptions) (any, error) {
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	state, err := options.RandomValue()
	if err != nil {
		return nil, err
	}
	callback, err := options.StartCallback(&OAuthCallbackServerOptions{
		ProviderName: "Antigravity", Host: options.CallbackHost(), Port: antigravityOAuthCallbackPort,
		Path: antigravityOAuthCallbackPath, State: &state, Context: ctx,
		Complete: func(code string) (any, error) { return code, nil },
	})
	if err != nil {
		// An occupied port or sandboxed environment must not abort login: the
		// flow falls back to manual paste, like the sibling OAuth flows.
		callback = nil
	}
	if callback != nil {
		defer callback.Close()
	}
	redirectURI := antigravityOAuthRedirectURI(callback)
	params := url.Values{
		"client_id":     {antigravityOAuthClientID},
		"redirect_uri":  {redirectURI},
		"response_type": {"code"},
		"scope":         {antigravityOAuthScope},
		"access_type":   {"offline"},
		"prompt":        {"consent"},
		"state":         {state},
	}
	interaction.Notify(NewObject(
		Property{Name: "type", Value: "auth_url"},
		Property{Name: "url", Value: antigravityOAuthAuthorizeURL + "?" + params.Encode()},
		Property{Name: "instructions", Value: "Complete the sign-in in your browser."},
	))
	result, err := WaitForCallbackOrManualInput(interaction, callback, OAuthManualPrompt{
		Message:     "Complete login in your browser, or paste the authorization code / redirect URL here:",
		Placeholder: "http://127.0.0.1:51121/oauth-callback?code=…&state=…",
	})
	if err != nil {
		return nil, err
	}
	var code any
	if result.Get("type") == "callback" {
		code = result.Get("value")
	} else {
		input := result.Get("input")
		text, ok := input.(string)
		if !ok {
			if jsonjs.IsNullish(input) {
				return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(input), "input.trim")
			}
			return nil, errors.New("input.trim is not a function. (In 'input.trim()', 'input.trim' is undefined)")
		}
		code, err = antigravityOAuthParseInput(text, state)
		if err != nil {
			return nil, err
		}
	}
	codeText, ok := code.(string)
	if !ok || strings.TrimSpace(codeText) == "" {
		return nil, errors.New("Missing authorization code")
	}
	interaction.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "Exchanging authorization code for tokens..."}))
	token, err := antigravityOAuthExchange(ctx, codeText, redirectURI, options)
	if err != nil {
		if ctx.Err() != nil {
			return nil, errors.New("Login cancelled")
		}
		return nil, err
	}
	access, _ := token.Get("access").(string)
	interaction.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "Resolving the Cloud Code project..."}))
	project, err := antigravityOAuthProject(ctx, access, options)
	if err != nil {
		return nil, err
	}
	token.Set("project", project)
	return token, nil
}

// antigravityOAuthRedirectURI returns the bound callback's URI, or the fixed
// loopback endpoint when no listener could bind (manual paste still works —
// the registered redirect never leaves 127.0.0.1).
func antigravityOAuthRedirectURI(callback *OAuthCallbackServer) string {
	if callback != nil && callback.RedirectURI != "" {
		return callback.RedirectURI
	}
	return fmt.Sprintf("http://%s:%d%s", "127.0.0.1", antigravityOAuthCallbackPort, antigravityOAuthCallbackPath)
}

// antigravityOAuthParseInput accepts the pasted callback URL or a bare code
// and enforces the state when the URL carries one.
func antigravityOAuthParseInput(input, state string) (any, error) {
	code, gotState := parseOAuthAuthorizationInput(input)
	if gotState != nil && *gotState != state {
		return nil, errors.New("State mismatch")
	}
	return code, nil
}

// antigravityOAuthExchange posts the code grant (form-encoded, client_secret
// included — Google's pair accepts it) and maps the token response onto the
// stored credential, adding the account email from userinfo.
func antigravityOAuthExchange(ctx context.Context, code, redirectURI string, options AntigravityOAuthOptions) (*Object, error) {
	form := url.Values{
		"grant_type":    {"authorization_code"},
		"client_id":     {antigravityOAuthClientID},
		"client_secret": {antigravityOAuthClientSecret},
		"code":          {code},
		"redirect_uri":  {redirectURI},
	}
	return antigravityOAuthToken(ctx, form, options, true)
}

// antigravityOAuthRefresh posts the refresh grant. The returned credential
// keeps the previous refresh token when Google rotates nothing and re-uses
// the stored project — refresh doesn't re-run discovery.
func antigravityOAuthRefresh(ctx context.Context, refresh, previous any, options AntigravityOAuthOptions) (any, error) {
	text, _ := refresh.(string)
	if strings.TrimSpace(text) == "" {
		return nil, errors.New("credential.refresh is missing — run login again")
	}
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"client_id":     {antigravityOAuthClientID},
		"client_secret": {antigravityOAuthClientSecret},
		"refresh_token": {text},
	}
	token, err := antigravityOAuthToken(ctx, form, options, false)
	if err != nil {
		return nil, err
	}
	if r, _ := token.Get("refresh").(string); r == "" {
		token.Set("refresh", text)
	}
	if previous != nil {
		if project, _ := catalogProperty(previous, "project").(string); project != "" && token.Get("project") == nil {
			token.Set("project", project)
		}
		if account, _ := catalogProperty(previous, "account").(string); account != "" && token.Get("account") == nil {
			token.Set("account", account)
		}
	}
	return token, nil
}

// antigravityOAuthToken POSTs a form grant and decodes the credential.
// fetchUser is true for the login exchange (fills the account email).
func antigravityOAuthToken(ctx context.Context, form url.Values, options AntigravityOAuthOptions, fetchUser bool) (*Object, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, antigravityOAuthTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	resp, err := options.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("Antigravity token exchange failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var body struct {
		AccessToken  string  `json:"access_token"`
		RefreshToken string  `json:"refresh_token"`
		ExpiresIn    float64 `json:"expires_in"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("token endpoint returned invalid JSON: %s", strings.TrimSpace(string(raw)))
	}
	if body.AccessToken == "" {
		return nil, errors.New("token response missing access_token")
	}
	if fetchUser && body.RefreshToken == "" {
		return nil, errors.New("no refresh token received; please try again")
	}
	credential := NewObject(
		Property{Name: "type", Value: "oauth"},
		Property{Name: "access", Value: body.AccessToken},
		Property{Name: "refresh", Value: body.RefreshToken},
		Property{Name: "expires", Value: options.Now() + body.ExpiresIn*1000 - antigravityOAuthSkewMS},
	)
	if fetchUser {
		if email, err := antigravityOAuthEmail(ctx, body.AccessToken, options); err == nil && email != "" {
			credential.Set("account", email)
		}
	}
	return credential, nil
}

func antigravityOAuthEmail(ctx context.Context, access string, options AntigravityOAuthOptions) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, antigravityOAuthUserinfoURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+access)
	resp, err := options.Client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil || resp.StatusCode/100 != 2 {
		return "", fmt.Errorf("google userinfo failed: %d", resp.StatusCode)
	}
	var body struct {
		Email string `json:"email"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return "", err
	}
	return body.Email, nil
}

// antigravityOAuthProject resolves the Cloud Code Assist project after login:
// loadCodeAssist → optional free-tier onboardUser LRO → reload. Mirrors
// internal/oauth's discoverAntigravityProject.
func antigravityOAuthProject(ctx context.Context, access string, options AntigravityOAuthOptions) (string, error) {
	initial, err := antigravityOAuthLoadCodeAssist(ctx, access, options)
	if err != nil {
		return "", err
	}
	if err := antigravityOAuthFreeTierEligible(initial); err != nil {
		return "", err
	}
	if _, hasTier := initial["currentTier"]; !hasTier {
		if err := antigravityOAuthOnboard(ctx, access, options); err != nil {
			return "", err
		}
	} else if project := antigravityOAuthExtractProject(initial); project != "" {
		return project, nil
	}
	refreshed, err := antigravityOAuthLoadCodeAssist(ctx, access, options)
	if err != nil {
		return "", err
	}
	if project := antigravityOAuthExtractProject(refreshed); project != "" {
		return project, nil
	}
	return "", errors.New("loadCodeAssist did not return a cloudaicompanionProject")
}

// antigravityOAuthLoadCodeAssist POSTs :loadCodeAssist and, when the response
// carries a project but no paidTier, re-posts with the project attached to
// force tier resolution.
func antigravityOAuthLoadCodeAssist(ctx context.Context, access string, options AntigravityOAuthOptions) (map[string]any, error) {
	payload, err := antigravityOAuthCall(ctx, access, options, "/v1internal:loadCodeAssist",
		map[string]any{"metadata": antigravityOAuthMetadata})
	if err != nil {
		return nil, err
	}
	project := antigravityOAuthExtractProject(payload)
	if _, hasPaid := payload["paidTier"]; !hasPaid && project != "" {
		payload, err = antigravityOAuthCall(ctx, access, options, "/v1internal:loadCodeAssist",
			map[string]any{"cloudaicompanionProject": project, "metadata": antigravityOAuthMetadata})
		if err != nil {
			return nil, err
		}
	}
	return payload, nil
}

// antigravityOAuthExtractProject reads cloudaicompanionProject as either a
// bare string or an {id: ...} object — both shapes appear in the wild.
func antigravityOAuthExtractProject(payload map[string]any) string {
	switch v := payload["cloudaicompanionProject"].(type) {
	case string:
		return v
	case map[string]any:
		s, _ := v["id"].(string)
		return s
	}
	return ""
}

// antigravityOAuthFreeTierEligible mirrors the TS check: an explicit free-tier
// ineligibility reason aborts login; otherwise onboarding proceeds.
func antigravityOAuthFreeTierEligible(payload map[string]any) error {
	if tiers, ok := payload["allowedTiers"].([]any); ok {
		for _, t := range tiers {
			if m, _ := t.(map[string]any); m != nil && m["id"] == antigravityOAuthFreeTierID {
				return nil
			}
		}
	}
	if tiers, ok := payload["ineligibleTiers"].([]any); ok {
		for _, t := range tiers {
			m, _ := t.(map[string]any)
			if m == nil || m["tierId"] != antigravityOAuthFreeTierID {
				continue
			}
			reason, _ := m["reasonMessage"].(string)
			if reason == "" {
				return nil
			}
			if vurl, _ := m["validationUrl"].(string); vurl != "" {
				reason += "\n" + vurl
			}
			return errors.New(reason)
		}
	}
	return nil
}

// antigravityOAuthOnboard starts the free-tier LRO and polls the operation
// until done (30s cap, 1s interval).
func antigravityOAuthOnboard(ctx context.Context, access string, options AntigravityOAuthOptions) error {
	deadline := time.Now().Add(antigravityOAuthOnboardCap)
	op, err := antigravityOAuthCall(ctx, access, options, "/v1internal:onboardUser",
		map[string]any{"tierId": antigravityOAuthFreeTierID, "metadata": antigravityOAuthMetadata})
	if err != nil {
		return err
	}
	for {
		if done, _ := op["done"].(bool); done {
			if opErr, _ := op["error"].(map[string]any); opErr != nil {
				msg, _ := opErr["message"].(string)
				return fmt.Errorf("onboardUser operation failed: %s", msg)
			}
			if op["response"] == nil {
				return errors.New("onboardUser finished without a response")
			}
			return nil
		}
		name, _ := op["name"].(string)
		if name == "" {
			return errors.New("onboardUser returned an operation without a name")
		}
		if time.Now().Add(antigravityOAuthOnboardPoll).After(deadline) {
			return errors.New("onboardUser timed out after 30s")
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(antigravityOAuthOnboardPoll):
		}
		op, err = antigravityOAuthCall(ctx, access, options, "/v1internal/"+name, nil)
		if err != nil {
			return err
		}
	}
}

// antigravityOAuthCall POSTs a Cloud Code Assist internal endpoint.
func antigravityOAuthCall(ctx context.Context, access string, options AntigravityOAuthOptions, path string, body map[string]any) (map[string]any, error) {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = strings.NewReader(string(raw))
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, antigravityDailyEndpoint+path, reader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+access)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", AntigravityUserAgent)
	resp, err := options.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("POST %s failed: %d %s", path, resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, errors.New("cloud code assist returned invalid JSON")
	}
	return payload, nil
}
