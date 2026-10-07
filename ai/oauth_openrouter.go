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
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
	"golang.org/x/text/encoding/unicode"
)

const openRouterAuthorizeURL = "https://openrouter.ai/auth"
const openRouterTokenURL = "https://openrouter.ai/api/v1/auth/keys"
const openRouterExchangeTimeoutMessage = "OpenRouter OAuth token exchange timed out"

// OpenRouterOAuthOptions supplies native transport, entropy and listener
// dependencies. Defaults use crypto/rand, a real loopback server, a five-minute
// login deadline and a thirty-second exchange deadline. It does not change the
// provider's authorization or token URLs.
type OpenRouterOAuthOptions struct {
	Client          *http.Client
	PKCE            func() (PKCE, error)
	RandomUUID      func() (string, error)
	StartCallback   func(*OAuthCallbackServerOptions) (*OAuthCallbackServer, error)
	CallbackHost    func() string
	LoginTimeoutMS  *float64
	ExchangeTimeout time.Duration
}

func OpenRouterOAuth(settings ...OpenRouterOAuthOptions) *OAuthAuth {
	var options OpenRouterOAuthOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.PKCE == nil {
		options.PKCE = GeneratePKCE
	}
	if options.RandomUUID == nil {
		options.RandomUUID = func() (string, error) {
			value, err := uuid.NewRandom()
			return value.String(), err
		}
	}
	if options.StartCallback == nil {
		options.StartCallback = StartOAuthCallbackServer
	}
	if options.CallbackHost == nil {
		options.CallbackHost = func() string {
			if host := os.Getenv("PI_OAUTH_CALLBACK_HOST"); host != "" {
				return host
			}
			return "127.0.0.1"
		}
	}
	if options.LoginTimeoutMS == nil {
		ms := float64(5 * 60 * 1000)
		options.LoginTimeoutMS = &ms
	}
	if options.ExchangeTimeout == 0 {
		options.ExchangeTimeout = 30 * time.Second
	}
	label := "Sign in with OpenRouter"
	return &OAuthAuth{Name: "OpenRouter OAuth", LoginLabel: &label,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) { return loginOpenRouter(interaction, options) })
		},
		Refresh: func(_ context.Context, credential any) (any, error) { return credential, nil },
		ToAuth: func(credential any) (any, error) {
			if jsonjs.IsNullish(credential) {
				return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.access")
			}
			return NewObject(Property{Name: "apiKey", Value: catalogProperty(credential, "access")}), nil
		},
	}
}

func loginOpenRouter(interaction ProviderAuthInteraction, options OpenRouterOAuthOptions) (any, error) {
	pkce, err := options.PKCE()
	if err != nil {
		return nil, err
	}
	// Source evaluation reads the callback host before generating the UUID.
	host := options.CallbackHost()
	id, err := options.RandomUUID()
	if err != nil {
		return nil, err
	}
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	callback, err := options.StartCallback(&OAuthCallbackServerOptions{
		ProviderName: "OpenRouter", Host: host, Path: "/oauth/callback/" + id,
		Context: ctx, TimeoutMS: options.LoginTimeoutMS,
		Complete: func(code string) (any, error) {
			return exchangeOpenRouterCode(ctx, code, pkce.Verifier, options)
		},
	})
	if err != nil {
		return nil, err
	}
	defer callback.Close()
	encode := func(value string) string {
		return strings.NewReplacer("%2A", "*", "~", "%7E").Replace(url.QueryEscape(value))
	}
	authorizeURL := openRouterAuthorizeURL + "?callback_url=" + encode(callback.RedirectURI) + "&code_challenge=" + encode(pkce.Challenge) + "&code_challenge_method=S256"
	interaction.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "Listening for OpenRouter OAuth callback on " + callback.RedirectURI}))
	interaction.Notify(NewObject(Property{Name: "type", Value: "auth_url"}, Property{Name: "url", Value: authorizeURL}, Property{Name: "instructions", Value: "Complete sign-in in your browser. If the browser is on another machine, paste the final redirect URL here."}))
	result, err := WaitForCallbackOrManualInput(interaction, callback, OAuthManualPrompt{
		Message: "Complete sign-in in your browser, or paste the authorization code / redirect URL here:", Placeholder: callback.RedirectURI,
	})
	if err != nil {
		return nil, err
	}
	if result.Get("type") == "callback" {
		return result.Get("value"), nil
	}
	input, ok := result.Get("input").(string)
	if !ok {
		return nil, errors.New("input.trim is not a function. (In 'input.trim()', 'input.trim' is undefined)")
	}
	code := parseOpenRouterAuthorizationInput(input)
	if code == "" {
		return nil, errors.New("Missing authorization code")
	}
	interaction.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "Exchanging authorization code for an API key..."}))
	return exchangeOpenRouterCode(ctx, code, pkce.Verifier, options)
}

func parseOpenRouterAuthorizationInput(input string) string {
	value := strings.TrimFunc(input, jsWhitespace)
	if value == "" {
		return ""
	}
	if parsed, err := whatwg.Parse(value); err == nil {
		return urlQueryValues(parsed.Query())["code"]
	}
	if strings.Contains(value, "code=") {
		return urlQueryValues(strings.TrimPrefix(value, "?"))["code"]
	}
	return value
}

func openRouterErrorDetail(body *Object) string {
	for _, field := range []string{"error_description", "message", "error"} {
		if value, ok := body.Get(field).(string); ok {
			return value
		}
	}
	if nested, ok := body.Get("error").(*Object); ok {
		value, _ := nested.Get("message").(string)
		return value
	}
	return ""
}

func exchangeOpenRouterCode(ctx context.Context, code, verifier string, options OpenRouterOAuthOptions) (any, error) {
	if ctx.Err() != nil {
		return nil, errors.New(deviceCodeCancelMessage)
	}
	requestContext, cancel := context.WithCancelCause(context.Background())
	stopAbort := context.AfterFunc(ctx, func() { cancel(context.Cause(ctx)) })
	timer := time.AfterFunc(options.ExchangeTimeout, func() { cancel(errors.New(openRouterExchangeTimeoutMessage)) })
	defer timer.Stop()
	defer stopAbort()
	failure := func(err error) (any, error) {
		if ctx.Err() != nil {
			return nil, errors.New(deviceCodeCancelMessage)
		}
		if requestContext.Err() != nil {
			return nil, errors.New(openRouterExchangeTimeoutMessage)
		}
		return nil, err
	}
	encoded, err := jsonjs.MarshalValue(NewObject(Property{Name: "code", Value: code}, Property{Name: "code_verifier", Value: verifier}, Property{Name: "code_challenge_method", Value: "S256"}))
	if err != nil {
		return failure(err)
	}
	request, err := http.NewRequestWithContext(requestContext, "POST", openRouterTokenURL, bytes.NewReader(encoded))
	if err != nil {
		return failure(err)
	}
	request.Header.Set("accept", "application/json")
	request.Header.Set("content-type", "application/json")
	response, err := options.Client.Do(request)
	if ctx.Err() != nil {
		cancel(context.Cause(ctx))
	}
	if err != nil {
		if transportError, ok := err.(*url.Error); ok {
			err = transportError.Err
		}
		return failure(err)
	}
	defer response.Body.Close()
	ok := response.StatusCode >= 200 && response.StatusCode < 300
	body := NewObject()
	raw, readErr := io.ReadAll(response.Body)
	if ctx.Err() != nil {
		cancel(context.Cause(ctx))
	}
	if readErr == nil {
		raw, _ = unicode.UTF8BOM.NewDecoder().Bytes(raw)
		var parsed any
		parsed, readErr = jsonjs.DecodeValue(raw)
		if object, isObject := parsed.(*Object); isObject && readErr == nil {
			body = object
		}
	}
	if readErr != nil && ok {
		return failure(errors.New("OpenRouter OAuth returned invalid JSON"))
	}
	timer.Stop()
	stopAbort()
	if !ok {
		message := fmt.Sprintf("OpenRouter OAuth key exchange failed (HTTP %d)", response.StatusCode)
		if detail := openRouterErrorDetail(body); detail != "" {
			message += ": " + detail
		}
		return nil, errors.New(message)
	}
	key, isString := body.Get("key").(string)
	if !isString || key == "" {
		return nil, errors.New(`OpenRouter OAuth response carries no "key"`)
	}
	return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: key}, Property{Name: "refresh", Value: ""}, Property{Name: "expires", Value: float64(9007199254740991)}), nil
}
