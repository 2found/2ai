package ai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/jsonjs"
	"golang.org/x/text/encoding/unicode"
)

const anthropicOAuthClientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
const anthropicOAuthTokenURL = "https://platform.claude.com/v1/oauth/token"
const anthropicOAuthRedirectURI = "http://localhost:53692/callback"
const anthropicOAuthCopyRedirectURI = "https://platform.claude.com/oauth/code/callback"
const anthropicOAuthScopes = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"

// AnthropicOAuthOptions supplies concrete native dependencies. CallbackHost is
// captured when the flow is constructed, like Pi's loaded module constant.
// ErrorStack controls stacks for errors created by this flow; the default is a
// native Go stack. Supplied OAuthDiagnosticError values retain their own stacks.
type AnthropicOAuthOptions struct {
	Client         *http.Client
	PKCE           func() (PKCE, error)
	StartCallback  func(*OAuthCallbackServerOptions) (*OAuthCallbackServer, error)
	CallbackHost   *string
	Now            func() float64
	RequestTimeout time.Duration
	ErrorStack     func(name, message string) string
}

func AnthropicOAuth(settings ...AnthropicOAuthOptions) *OAuthAuth {
	var options AnthropicOAuthOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.PKCE == nil {
		options.PKCE = GeneratePKCE
	}
	if options.StartCallback == nil {
		options.StartCallback = StartOAuthCallbackServer
	}
	host := os.Getenv("PI_OAUTH_CALLBACK_HOST")
	if host == "" {
		host = "127.0.0.1"
	}
	if options.CallbackHost != nil {
		host = *options.CallbackHost
	}
	options.CallbackHost = &host
	if options.Now == nil {
		options.Now = func() float64 { return float64(time.Now().UnixMilli()) }
	}
	if options.RequestTimeout == 0 {
		options.RequestTimeout = 30 * time.Second
	}
	if options.ErrorStack == nil {
		options.ErrorStack = func(name, message string) string { return name + ": " + message + "\n" + string(debug.Stack()) }
	}
	subscription := true
	return &OAuthAuth{Name: "Anthropic (Claude Pro/Max)", IsSubscription: &subscription,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) {
				method, err := interaction.Prompt(context.Background(), NewObject(Property{Name: "type", Value: "select"}, Property{Name: "message", Value: "Select Anthropic login method:"}, Property{Name: "options", Value: NewArray(
					NewObject(Property{Name: "id", Value: "browser"}, Property{Name: "label", Value: "Browser login (default)"}),
					NewObject(Property{Name: "id", Value: "copy_code"}, Property{Name: "label", Value: "Copy code login (headless)"}),
				)}))
				if err != nil {
					return nil, err
				}
				if method != "browser" && method != "copy_code" {
					text, err := catalogKey(method, make(map[*Array]bool))
					if err != nil {
						return nil, err
					}
					return nil, errors.New("Unknown Anthropic login method: " + text)
				}
				return loginAnthropic(interaction, method == "copy_code", options)
			})
		},
		Refresh: func(ctx context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.refresh")
				}
				body := NewObject(Property{Name: "grant_type", Value: "refresh_token"}, Property{Name: "client_id", Value: anthropicOAuthClientID}, Property{Name: "refresh_token", Value: catalogProperty(credential, "refresh")})
				return anthropicOAuthToken(ctx, body, "", options)
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

func loginAnthropic(interaction ProviderAuthInteraction, copyCode bool, options AnthropicOAuthOptions) (any, error) {
	pkce, err := options.PKCE()
	if err != nil {
		return nil, err
	}
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	redirectURI := anthropicOAuthRedirectURI
	var callback *OAuthCallbackServer
	if copyCode {
		redirectURI = anthropicOAuthCopyRedirectURI
	} else {
		// Pi falls back to manual input for every callback-start rejection,
		// including an occupied port and an already-aborted interaction.
		started, startErr := invokeAuth(func() (any, error) {
			return options.StartCallback(&OAuthCallbackServerOptions{ProviderName: "Anthropic", Host: *options.CallbackHost, Port: 53692, Path: "/callback", State: &pkce.Verifier, Context: ctx, Complete: func(code string) (any, error) { return code, nil }})
		})
		if startErr == nil {
			callback, _ = started.(*OAuthCallbackServer)
		}
		if callback != nil {
			defer callback.Close()
		}
	}
	encode := func(value string) string {
		return strings.NewReplacer("%2A", "*", "~", "%7E").Replace(url.QueryEscape(value))
	}
	authURL := "https://claude.ai/oauth/authorize?code=true&client_id=" + anthropicOAuthClientID + "&response_type=code&redirect_uri=" + encode(redirectURI) + "&scope=" + encode(anthropicOAuthScopes) + "&code_challenge=" + encode(pkce.Challenge) + "&code_challenge_method=S256&state=" + encode(pkce.Verifier)
	instructions := "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."
	if copyCode {
		instructions = "Complete login in your browser, then copy the code Anthropic shows and paste it here."
	}
	interaction.Notify(NewObject(Property{Name: "type", Value: "auth_url"}, Property{Name: "url", Value: authURL}, Property{Name: "instructions", Value: instructions}))
	var input any
	if copyCode {
		input, err = interaction.Prompt(ctx, NewObject(Property{Name: "type", Value: "manual_code"}, Property{Name: "message", Value: "Paste the code Anthropic shows after you sign in:"}, Property{Name: "placeholder", Value: "code#state"}))
	} else {
		var result *Object
		result, err = WaitForCallbackOrManualInput(interaction, callback, OAuthManualPrompt{Message: "Complete login in your browser, or paste the authorization code / redirect URL here:", Placeholder: anthropicOAuthRedirectURI})
		if err == nil && result.Get("type") == "callback" {
			code, _ := result.Get("value").(string)
			if code == "" {
				return nil, errors.New("Missing authorization code")
			}
			return exchangeAnthropicCode(ctx, code, pkce.Verifier, pkce.Verifier, redirectURI, interaction, options)
		}
		if err == nil {
			input = result.Get("input")
		}
	}
	if err != nil {
		return nil, err
	}
	text, ok := input.(string)
	if !ok {
		if jsonjs.IsNullish(input) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(input), "input.trim")
		}
		return nil, errors.New("input.trim is not a function. (In 'input.trim()', 'input.trim' is undefined)")
	}
	code, state := parseOAuthAuthorizationInput(text)
	if state != nil && *state != "" && *state != pkce.Verifier {
		return nil, errors.New("OAuth state mismatch")
	}
	if code == "" {
		return nil, errors.New("Missing authorization code")
	}
	stateValue := pkce.Verifier
	if state != nil {
		stateValue = *state
	}
	return exchangeAnthropicCode(ctx, code, stateValue, pkce.Verifier, redirectURI, interaction, options)
}

func exchangeAnthropicCode(ctx context.Context, code, state, verifier, redirectURI string, interaction ProviderAuthInteraction, options AnthropicOAuthOptions) (any, error) {
	interaction.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "Exchanging authorization code for tokens..."}))
	body := NewObject(Property{Name: "grant_type", Value: "authorization_code"}, Property{Name: "client_id", Value: anthropicOAuthClientID}, Property{Name: "code", Value: code}, Property{Name: "state", Value: state}, Property{Name: "redirect_uri", Value: redirectURI}, Property{Name: "code_verifier", Value: verifier})
	return anthropicOAuthToken(ctx, body, redirectURI, options)
}

func anthropicOAuthDiagnostic(name, message string, options AnthropicOAuthOptions) *OAuthDiagnosticError {
	return &OAuthDiagnosticError{Name: &name, Message: message, stack: sync.OnceValue(func() string { return options.ErrorStack(name, message) })}
}

func anthropicOAuthToken(ctx context.Context, body *Object, redirectURI string, options AnthropicOAuthOptions) (any, error) {
	text, err := postAnthropicOAuthJSON(ctx, body, options)
	if err != nil {
		details, detailErr := formatOAuthErrorDetails(err)
		if detailErr != nil {
			return nil, detailErr
		}
		if redirectURI == "" {
			return nil, errors.New("Anthropic token refresh request failed. url=" + anthropicOAuthTokenURL + "; details=" + details)
		}
		return nil, errors.New("Token exchange request failed. url=" + anthropicOAuthTokenURL + "; redirect_uri=" + redirectURI + "; response_type=authorization_code; details=" + details)
	}
	if err := jsonjs.ValidateJSON([]byte(text)); err != nil {
		details, detailErr := formatOAuthErrorDetails(anthropicOAuthDiagnostic("SyntaxError", err.Error(), options))
		if detailErr != nil {
			return nil, detailErr
		}
		prefix := "Token exchange"
		if redirectURI == "" {
			prefix = "Anthropic token refresh"
		}
		return nil, errors.New(prefix + " returned invalid JSON. url=" + anthropicOAuthTokenURL + "; body=" + text + "; details=" + details)
	}
	data, err := jsonjs.DecodeValue([]byte(text))
	if err != nil {
		return nil, err
	}
	if jsonjs.IsNullish(data) {
		variable := "tokenData"
		if redirectURI == "" {
			variable = "data"
		}
		return nil, jsonjs.PropertyReadError(true, variable+".refresh_token")
	}
	seconds, err := authNumber(catalogProperty(data, "expires_in"))
	if err != nil {
		return nil, err
	}
	return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "refresh", Value: catalogProperty(data, "refresh_token")}, Property{Name: "access", Value: catalogProperty(data, "access_token")}, Property{Name: "expires", Value: options.Now() + seconds*1000 - 5*60*1000}), nil
}

func postAnthropicOAuthJSON(ctx context.Context, body *Object, options AnthropicOAuthOptions) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	timeoutName := "TimeoutError"
	// Pi's AbortSignal.timeout reason is a DOMException (TIMEOUT_ERR = 23),
	// without the Error stack consumed by formatErrorDetails in Bun.
	timeout := &OAuthDiagnosticError{Name: &timeoutName, Message: "The operation timed out.", Code: 23}
	requestContext, cancel := context.WithTimeoutCause(ctx, options.RequestTimeout, timeout)
	// AbortSignal.any/timeout stays live after a successful request. Retire the
	// native timer only when that composite signal actually aborts.
	context.AfterFunc(requestContext, cancel)
	encoded, err := jsonjs.MarshalValue(body)
	if err != nil {
		return "", err
	}
	request, err := http.NewRequestWithContext(requestContext, "POST", anthropicOAuthTokenURL, bytes.NewReader(encoded))
	if err != nil {
		return "", err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	response, err := options.Client.Do(request)
	if err != nil {
		if wrapped, ok := err.(*url.Error); ok {
			err = wrapped.Err
		}
		if (err == context.Canceled || err == context.DeadlineExceeded) && requestContext.Err() != nil {
			return "", anthropicOAuthAbortReason(requestContext, options)
		}
		return "", err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		if (err == context.Canceled || err == context.DeadlineExceeded) && requestContext.Err() != nil {
			return "", anthropicOAuthAbortReason(requestContext, options)
		}
		return "", err
	}
	raw, _ = unicode.UTF8BOM.NewDecoder().Bytes(raw)
	text := string(raw)
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", anthropicOAuthDiagnostic("Error", fmt.Sprintf("HTTP request failed. status=%d; url=%s; body=%s", response.StatusCode, anthropicOAuthTokenURL, text), options)
	}
	return text, nil
}

// Map Go's default cancellation reasons to AbortSignal's DOMException fields.
// Explicit host-provided causes retain their own identity and diagnostics.
func anthropicOAuthAbortReason(ctx context.Context, options AnthropicOAuthOptions) error {
	cause := context.Cause(ctx)
	switch cause {
	case context.Canceled:
		failure := anthropicOAuthDiagnostic("AbortError", "The operation was aborted.", options)
		failure.Code = 20
		return failure
	case context.DeadlineExceeded:
		name := "TimeoutError"
		return &OAuthDiagnosticError{Name: &name, Message: "The operation timed out.", Code: 23}
	default:
		return cause
	}
}
