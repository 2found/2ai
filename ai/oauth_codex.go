package ai

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/2found/2ai/jsonjs"
)

const codexOAuthClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
const codexOAuthTokenURL = "https://auth.openai.com/oauth/token"
const codexOAuthRedirect = "http://localhost:1455/auth/callback"
const codexOAuthDeviceRedirect = "https://auth.openai.com/deviceauth/callback"

type OpenAICodexOAuthOptions struct {
	Client        *http.Client
	PKCE          func() (PKCE, error)
	State         func() (string, error)
	CallbackHost  func() string
	StartCallback func(*OAuthCallbackServerOptions) (*OAuthCallbackServer, error)
	Now           func() float64
	Sleep         func(float64, context.Context, string) error
}

func OpenAICodexOAuth(settings ...OpenAICodexOAuthOptions) *OAuthAuth {
	var options OpenAICodexOAuthOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.PKCE == nil {
		options.PKCE = GeneratePKCE
	}
	if options.State == nil {
		options.State = func() (string, error) {
			var b [16]byte
			if _, err := rand.Read(b[:]); err != nil {
				return "", err
			}
			return hex.EncodeToString(b[:]), nil
		}
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
	subscription := true
	return &OAuthAuth{Name: "OpenAI (ChatGPT Plus/Pro)", IsSubscription: &subscription,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) {
				method, err := interaction.Prompt(context.Background(), NewObject(Property{Name: "type", Value: "select"}, Property{Name: "message", Value: "Select OpenAI Codex login method:"}, Property{Name: "options", Value: NewArray(NewObject(Property{Name: "id", Value: "browser"}, Property{Name: "label", Value: "Browser login (default)"}), NewObject(Property{Name: "id", Value: "device_code"}, Property{Name: "label", Value: "Device code login (headless)"}))}))
				if err != nil {
					return nil, err
				}
				if method == "device_code" {
					return loginCodexOAuthDevice(interaction, options)
				}
				if method != "browser" {
					value, err := catalogKey(method, make(map[*Array]bool))
					if err != nil {
						return nil, err
					}
					return nil, errors.New("Unknown OpenAI Codex login method: " + value)
				}
				return loginCodexOAuthBrowser(interaction, options)
			})
		},
		Refresh: func(ctx context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.refresh")
				}
				body, err := oauthFormEncode(Property{Name: "grant_type", Value: "refresh_token"}, Property{Name: "refresh_token", Value: catalogProperty(credential, "refresh")}, Property{Name: "client_id", Value: codexOAuthClientID})
				if err != nil {
					return nil, errors.New("OpenAI Codex token refresh error: " + err.Error())
				}
				response, err := oauthFetch(ctx, options.Client, codexOAuthTokenURL, "POST", http.Header{"Content-Type": {"application/x-www-form-urlencoded"}}, body)
				if err != nil {
					return nil, errors.New("OpenAI Codex token refresh error: " + err.Error())
				}
				return codexOAuthToken(response, "refresh", options)
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
func codexOAuthFetch(ctx context.Context, endpoint, contentType, body string, options OpenAICodexOAuthOptions) (*http.Response, error) {
	response, err := oauthFetch(ctx, options.Client, endpoint, "POST", http.Header{"Content-Type": {contentType}}, body)
	if err != nil && ctx.Err() != nil {
		return nil, errors.New("Login cancelled")
	}
	return response, err
}
func codexOAuthToken(response *http.Response, operation string, options OpenAICodexOAuthOptions) (any, error) {
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		raw, _ := oauthRead(response)
		text := string(raw)
		if text == "" {
			text = http.StatusText(response.StatusCode)
			if response.Status != "" {
				_, text, _ = strings.Cut(response.Status, " ")
			}
		}
		return nil, fmt.Errorf("OpenAI Codex token %s failed (%d): %s", operation, response.StatusCode, text)
	}
	value, err := oauthReadJSON(response)
	if err != nil {
		return nil, err
	}
	access, refresh := catalogProperty(value, "access_token"), catalogProperty(value, "refresh_token")
	expires, number := deviceCodeNumber(catalogProperty(value, "expires_in"))
	if !catalogEntryTruthy(access) || !catalogEntryTruthy(refresh) || !number {
		raw, _ := jsonjs.MarshalValue(value)
		return nil, fmt.Errorf("OpenAI Codex token %s response missing fields: %s", operation, raw)
	}
	account, err := codexOAuthAccountID(access)
	if err != nil {
		return nil, err
	}
	return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: access}, Property{Name: "refresh", Value: refresh}, Property{Name: "expires", Value: options.Now() + expires*1000}, Property{Name: "accountId", Value: account}), nil
}
func codexOAuthAccountID(access any) (string, error) {
	invalid := errors.New("Failed to extract accountId from token")
	token, ok := access.(string)
	if !ok {
		return "", invalid
	}
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", invalid
	}
	payload := strings.Map(func(r rune) rune {
		if strings.ContainsRune(" \t\r\n\f", r) {
			return -1
		}
		return r
	}, parts[1])
	var raw []byte
	var err error
	if strings.Contains(payload, "=") {
		raw, err = base64.StdEncoding.DecodeString(payload)
	} else {
		raw, err = base64.RawStdEncoding.DecodeString(payload)
	}
	if err != nil {
		return "", invalid
	}
	latin := make([]rune, len(raw))
	for i, b := range raw {
		latin[i] = rune(b)
	}
	value, err := jsonjs.DecodeValue([]byte(string(latin)))
	if err != nil {
		return "", invalid
	}
	account, ok := catalogProperty(catalogProperty(value, "https://api.openai.com/auth"), "chatgpt_account_id").(string)
	if !ok || account == "" {
		return "", invalid
	}
	return account, nil
}
func codexOAuthExchange(ctx context.Context, code, verifier any, redirect string, options OpenAICodexOAuthOptions) (any, error) {
	body, err := oauthFormEncode(Property{Name: "grant_type", Value: "authorization_code"}, Property{Name: "client_id", Value: codexOAuthClientID}, Property{Name: "code", Value: code}, Property{Name: "code_verifier", Value: verifier}, Property{Name: "redirect_uri", Value: redirect})
	if err != nil {
		return nil, err
	}
	response, err := codexOAuthFetch(ctx, codexOAuthTokenURL, "application/x-www-form-urlencoded", body, options)
	if err != nil {
		return nil, err
	}
	return codexOAuthToken(response, "exchange", options)
}
func loginCodexOAuthBrowser(interaction ProviderAuthInteraction, options OpenAICodexOAuthOptions) (any, error) {
	pkce, err := options.PKCE()
	if err != nil {
		return nil, err
	}
	state, err := options.State()
	if err != nil {
		return nil, err
	}
	params, err := oauthFormEncode(Property{Name: "response_type", Value: "code"}, Property{Name: "client_id", Value: codexOAuthClientID}, Property{Name: "redirect_uri", Value: codexOAuthRedirect}, Property{Name: "scope", Value: "openid profile email offline_access"}, Property{Name: "code_challenge", Value: pkce.Challenge}, Property{Name: "code_challenge_method", Value: "S256"}, Property{Name: "state", Value: state}, Property{Name: "id_token_add_organizations", Value: "true"}, Property{Name: "codex_cli_simplified_flow", Value: "true"}, Property{Name: "originator", Value: "pi"})
	if err != nil {
		return nil, err
	}
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	var callback *OAuthCallbackServer
	_, _ = invokeAuth(func() (any, error) {
		server, err := options.StartCallback(&OAuthCallbackServerOptions{ProviderName: "OpenAI", Host: options.CallbackHost(), Port: 1455, Path: "/auth/callback", State: &state, Context: ctx, Complete: func(code string) (any, error) { return code, nil }})
		if err == nil {
			callback = server
		}
		return nil, err
	})
	// Pi notifies before entering its try/finally. A throwing host notification
	// does not close this listener; hosts can retain it through StartCallback.
	interaction.Notify(NewObject(Property{Name: "type", Value: "auth_url"}, Property{Name: "url", Value: "https://auth.openai.com/oauth/authorize?" + params}, Property{Name: "instructions", Value: "A browser window should open. Complete login to finish."}))
	if callback != nil {
		defer callback.Close()
	}
	result, err := WaitForCallbackOrManualInput(interaction, callback, OAuthManualPrompt{Message: "Complete login in your browser, or paste the authorization code / redirect URL here:", Placeholder: codexOAuthRedirect})
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
		parsed, gotState := parseOAuthAuthorizationInput(text)
		if gotState != nil && *gotState != "" && *gotState != state {
			return nil, errors.New("State mismatch")
		}
		code = parsed
	}
	if !catalogEntryTruthy(code) {
		return nil, errors.New("Missing authorization code")
	}
	return codexOAuthExchange(ctx, code, pkce.Verifier, codexOAuthRedirect, options)
}
func loginCodexOAuthDevice(interaction ProviderAuthInteraction, options OpenAICodexOAuthOptions) (any, error) {
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	response, err := codexOAuthFetch(ctx, "https://auth.openai.com/api/accounts/deviceauth/usercode", "application/json", `{"client_id":"`+codexOAuthClientID+`"}`, options)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		if response.StatusCode == 404 {
			response.Body.Close()
			return nil, errors.New("OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL.")
		}
		raw, _ := oauthRead(response)
		message := fmt.Sprintf("OpenAI Codex device code request failed with status %d", response.StatusCode)
		if len(raw) > 0 {
			message += ": " + string(raw)
		}
		return nil, errors.New(message)
	}
	value, err := oauthReadJSON(response)
	if err != nil {
		return nil, err
	}
	id, user, interval := catalogProperty(value, "device_auth_id"), catalogProperty(value, "user_code"), catalogProperty(value, "interval")
	if text, ok := interval.(string); ok {
		interval = ParseJSNumber(strings.TrimFunc(text, jsWhitespace))
	}
	seconds, number := deviceCodeNumber(interval)
	if !catalogEntryTruthy(id) || !catalogEntryTruthy(user) || !number || math.IsNaN(seconds) || math.IsInf(seconds, 0) || seconds < 0 {
		raw, _ := jsonjs.MarshalValue(value)
		return nil, fmt.Errorf("Invalid OpenAI Codex device code response: %s", raw)
	}
	interaction.Notify(NewObject(Property{Name: "type", Value: "device_code"}, Property{Name: "userCode", Value: user}, Property{Name: "verificationUri", Value: "https://auth.openai.com/codex/device"}, Property{Name: "intervalSeconds", Value: seconds}, Property{Name: "expiresInSeconds", Value: 900}))
	code, err := PollOAuthDeviceCodeFlow(&OAuthDeviceCodePollOptions{Context: ctx, Now: options.Now, Sleep: options.Sleep, IntervalSeconds: seconds, ExpiresInSeconds: 900, Poll: func() (OAuthDeviceCodePollResult, error) {
		body, err := jsonjs.MarshalValue(NewObject(Property{Name: "device_auth_id", Value: id}, Property{Name: "user_code", Value: user}))
		if err != nil {
			return OAuthDeviceCodePollResult{}, err
		}
		response, err := codexOAuthFetch(ctx, "https://auth.openai.com/api/accounts/deviceauth/token", "application/json", string(body), options)
		if err != nil {
			return OAuthDeviceCodePollResult{}, err
		}
		if response.StatusCode >= 200 && response.StatusCode < 300 {
			value, err := oauthReadJSON(response)
			if err != nil {
				return OAuthDeviceCodePollResult{}, err
			}
			code, verifier := catalogProperty(value, "authorization_code"), catalogProperty(value, "code_verifier")
			if !catalogEntryTruthy(code) || !catalogEntryTruthy(verifier) {
				raw, _ := jsonjs.MarshalValue(value)
				return OAuthDeviceCodePollResult{Status: "failed", Message: "Invalid OpenAI Codex device auth token response: " + string(raw)}, nil
			}
			return OAuthDeviceCodePollResult{Status: "complete", Value: NewObject(Property{Name: "code", Value: code}, Property{Name: "verifier", Value: verifier})}, nil
		}
		if response.StatusCode == 403 || response.StatusCode == 404 {
			response.Body.Close()
			return OAuthDeviceCodePollResult{Status: "pending"}, nil
		}
		raw, _ := oauthRead(response)
		var kind any
		if value, err := jsonjs.DecodeValue(raw); err == nil {
			kind = catalogProperty(value, "error")
			switch kind.(type) {
			case *Object, *Array:
				kind = catalogProperty(kind, "code")
			}
		}
		if kind == "deviceauth_authorization_pending" {
			return OAuthDeviceCodePollResult{Status: "pending"}, nil
		}
		if kind == "slow_down" {
			return OAuthDeviceCodePollResult{Status: "slow_down"}, nil
		}
		message := fmt.Sprintf("OpenAI Codex device auth failed with status %d", response.StatusCode)
		if len(raw) > 0 {
			message += ": " + string(raw)
		}
		return OAuthDeviceCodePollResult{Status: "failed", Message: message}, nil
	}})
	if err != nil {
		return nil, err
	}
	return codexOAuthExchange(ctx, catalogProperty(code, "code"), catalogProperty(code, "verifier"), codexOAuthDeviceRedirect, options)
}
