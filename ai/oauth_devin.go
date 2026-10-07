package ai

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/2found/2ai/jsonjs"
)

// Devin's CLI sign-in is an authorization-code + PKCE grant against
// app.devin.ai, exchanged for a long-lived session token at api.devin.ai. The
// token has no refresh grant (it is itself the refresh material the Cascade
// surface accepts), so expiry comes from the JWT `exp` claim with a
// one-year fallback, matching oh-my-pi's rules/auth/devin.kdl.
const (
	devinOAuthAuthorizeURL = "https://app.devin.ai/auth/cli/continue"
	devinOAuthTokenURL     = "https://api.devin.ai/auth/cli/token"
	devinOAuthCallbackPort = 59653
	devinOAuthCallbackPath = "/callback"
	// devinOAuthFallbackLifetimeMS is the kdl's `fallback-ms` when the token
	// carries no readable `exp`.
	devinOAuthFallbackLifetimeMS = 31536000000
)

type DevinOAuthOptions struct {
	Client        *http.Client
	PKCE          func() (PKCE, error)
	CallbackHost  func() string
	StartCallback func(*OAuthCallbackServerOptions) (*OAuthCallbackServer, error)
	Now           func() float64

	// authorizeURL/tokenURL override the shipped endpoints. Tests point them at
	// httptest servers; empty uses the Devin CLI endpoints.
	authorizeURL string
	tokenURL     string
}

// DevinOAuth builds the interactive login for the Devin vendor. The credential
// it returns is the same shape the other OAuth vendors store: a JS object with
// type/access/refresh/expires.
func DevinOAuth(settings ...DevinOAuthOptions) *OAuthAuth {
	var options DevinOAuthOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.PKCE == nil {
		options.PKCE = GeneratePKCE
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
	if options.authorizeURL == "" {
		options.authorizeURL = devinOAuthAuthorizeURL
	}
	if options.tokenURL == "" {
		options.tokenURL = devinOAuthTokenURL
	}
	subscription := true
	return &OAuthAuth{Name: "Devin", IsSubscription: &subscription,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) { return loginDevinOAuth(interaction, options) })
		},
		Refresh: func(_ context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.refresh")
				}
				// The session token is not refreshed by a grant; report it as-is
				// and let the pool rotate when the backend rejects it.
				return credential, nil
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

func loginDevinOAuth(interaction ProviderAuthInteraction, options DevinOAuthOptions) (any, error) {
	pkce, err := options.PKCE()
	if err != nil {
		return nil, err
	}
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	// Devin's authorize endpoint is a real OAuth flow: it needs the PKCE
	// challenge, a state the callback echoes back, and the exact redirect_uri
	// the browser will land on — without these the page just sits there.
	state := uuid.NewString()
	var callback *OAuthCallbackServer
	_, _ = invokeAuth(func() (any, error) {
		server, err := options.StartCallback(&OAuthCallbackServerOptions{
			ProviderName: "Devin", Host: options.CallbackHost(), Port: devinOAuthCallbackPort,
			Path: devinOAuthCallbackPath, State: &state, Context: ctx,
			Complete: func(code string) (any, error) { return code, nil },
		})
		if err == nil {
			callback = server
		}
		return nil, err
	})
	if callback != nil {
		defer callback.Close()
	}
	params := url.Values{
		"response_type":         {"code"},
		"redirect_uri":          {callbackRedirectURI(callback)},
		"state":                 {state},
		"code_challenge":        {pkce.Challenge},
		"code_challenge_method": {"S256"},
		"prompt":                {"select_account"},
	}
	authURL := options.authorizeURL + "?" + params.Encode()
	interaction.Notify(NewObject(
		Property{Name: "type", Value: "auth_url"},
		Property{Name: "url", Value: authURL},
		Property{Name: "instructions", Value: "Sign in to Devin in your browser."},
	))
	result, err := WaitForCallbackOrManualInput(interaction, callback, OAuthManualPrompt{
		Message:     "Complete login in your browser, or paste the authorization code / redirect URL here:",
		Placeholder: "https://app.devin.ai/auth/cli/continue/callback",
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
		parsed, _ := parseOAuthAuthorizationInput(text)
		code = parsed
	}
	if !catalogEntryTruthy(code) {
		return nil, errors.New("Missing authorization code")
	}
	return devinOAuthExchange(ctx, code, pkce.Verifier, options)
}

// callbackRedirectURI reports the callback's live redirect URI, falling back to
// the preferred localhost endpoint when no listener could bind (manual paste
// still works — the browser never leaves app.devin.ai in that mode anyway).
func callbackRedirectURI(callback *OAuthCallbackServer) string {
	if callback != nil && callback.RedirectURI != "" {
		return callback.RedirectURI
	}
	return fmt.Sprintf("http://%s:%d%s", "127.0.0.1", devinOAuthCallbackPort, devinOAuthCallbackPath)
}

// devinOAuthExchange posts the code + verifier and maps the response onto the
// stored credential shape. Devin answers `{"token": "…"}` (not access_token).
func devinOAuthExchange(ctx context.Context, code, verifier any, options DevinOAuthOptions) (any, error) {
	body, err := devinOAuthTokenBody(code, verifier)
	if err != nil {
		return nil, err
	}
	response, err := oauthFetch(ctx, options.Client, options.tokenURL, http.MethodPost,
		http.Header{"Content-Type": {"application/json"}, "Accept": {"application/json"}}, body)
	if err != nil {
		if ctx.Err() != nil {
			return nil, errors.New("Login cancelled")
		}
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		raw, _ := oauthRead(response)
		text := strings.TrimSpace(string(raw))
		if text == "" {
			text = http.StatusText(response.StatusCode)
		}
		return nil, fmt.Errorf("Devin token exchange failed (%d): %s", response.StatusCode, text)
	}
	value, err := oauthReadJSON(response)
	if err != nil {
		return nil, err
	}
	access := catalogProperty(value, "token")
	if !catalogEntryTruthy(access) {
		raw, _ := jsonjs.MarshalValue(value)
		return nil, fmt.Errorf("Devin token exchange response missing fields: %s", raw)
	}
	token, _ := access.(string)
	expires := devinTokenExpiryMS(token, options)
	return NewObject(
		Property{Name: "type", Value: "oauth"},
		Property{Name: "access", Value: token},
		Property{Name: "refresh", Value: token},
		Property{Name: "expires", Value: expires},
	), nil
}

func devinOAuthTokenBody(code, verifier any) (string, error) {
	text, ok := code.(string)
	if !ok {
		return "", errors.New("Devin token exchange requires an authorization code")
	}
	fields := []Property{{Name: "code", Value: text}}
	if verifierText, ok := verifier.(string); ok && verifierText != "" {
		fields = append(fields, Property{Name: "code_verifier", Value: verifierText})
	}
	value, err := jsonjs.MarshalValue(NewObject(fields...))
	if err != nil {
		return "", err
	}
	return string(value), nil
}

// devinTokenExpiryMS reads the JWT `exp` claim (seconds) and converts it to
// milliseconds; unreadable tokens get the kdl's one-year fallback.
func devinTokenExpiryMS(token string, options DevinOAuthOptions) float64 {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return options.Now() + devinOAuthFallbackLifetimeMS
	}
	payload := strings.Map(func(r rune) rune {
		if strings.ContainsRune(" \t\r\n\f", r) {
			return -1
		}
		return r
	}, parts[1])
	// JWT payloads are base64url — not Std — per RFC 7519.
	raw, err := base64.RawURLEncoding.DecodeString(payload)
	if err != nil {
		// Some issuers pad.
		if raw, err = base64.URLEncoding.DecodeString(payload); err != nil {
			return options.Now() + devinOAuthFallbackLifetimeMS
		}
	}
	latin := make([]rune, len(raw))
	for i, b := range raw {
		latin[i] = rune(b)
	}
	value, err := jsonjs.DecodeValue([]byte(string(latin)))
	if err != nil {
		return options.Now() + devinOAuthFallbackLifetimeMS
	}
	exp, ok := deviceCodeNumber(catalogProperty(value, "exp"))
	if !ok || exp <= 0 {
		return options.Now() + devinOAuthFallbackLifetimeMS
	}
	return exp*1000 - 5*60*1000
}
