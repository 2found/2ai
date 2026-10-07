package ai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

const radiusOAuthRedirect = "http://127.0.0.1:1456/oauth/callback"

// Name remains live for prompts; the gateway and advertised name are captured
// at construction. Callers synchronize any later options mutations.
type RadiusOAuthOptions struct{ Name, Gateway string }

type RadiusOAuthRuntimeOptions struct {
	Client        *http.Client
	Now           func() float64
	PollSleep     func(float64, context.Context, string) error
	PKCE          func() (PKCE, error)
	RandomUUID    func() (string, error)
	StartCallback func(*OAuthCallbackServerOptions) (*OAuthCallbackServer, error)
}

// RadiusOAuthResponseError preserves the status and optional OAuth error read
// by the device poller. An empty OAuth error is distinct from an absent one.
type RadiusOAuthResponseError struct {
	Status     int
	OAuthError *string
	Message    string
}

func (e *RadiusOAuthResponseError) Error() string { return e.Message }

func RadiusOAuth(input *RadiusOAuthOptions, settings ...RadiusOAuthRuntimeOptions) *OAuthAuth {
	var options RadiusOAuthRuntimeOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.Now == nil {
		options.Now = func() float64 { return float64(time.Now().UnixMilli()) }
	}
	if options.PKCE == nil {
		options.PKCE = GeneratePKCE
	}
	if options.RandomUUID == nil {
		options.RandomUUID = func() (string, error) { id, err := uuid.NewRandom(); return id.String(), err }
	}
	if options.StartCallback == nil {
		options.StartCallback = StartOAuthCallbackServer
	}
	gateway := NormalizeRadiusGatewayURL(input.Gateway)
	return &OAuthAuth{Name: input.Name,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) {
				ctx := interaction.Context
				if ctx == nil {
					ctx = context.Background()
				}
				// Pi's method-selection prompt does not receive the login signal.
				method, err := interaction.Prompt(context.Background(), NewObject(Property{Name: "type", Value: "select"}, Property{Name: "message", Value: "Sign in to " + input.Name + ":"}, Property{Name: "options", Value: NewArray(NewObject(Property{Name: "id", Value: "browser"}, Property{Name: "label", Value: "Sign in with browser (recommended)"}), NewObject(Property{Name: "id", Value: "device-code"}, Property{Name: "label", Value: "Sign in with device code (when signing in from another device)"}))}))
				if err != nil {
					return nil, err
				}
				if catalogStrictEqual(method, "device-code") {
					return loginRadiusDevice(ctx, gateway, interaction, options)
				}
				if catalogStrictEqual(method, "browser") {
					return loginRadiusBrowser(ctx, gateway, interaction, options)
				}
				text, err := catalogKey(method, make(map[*Array]bool))
				if err != nil {
					return nil, err
				}
				return nil, errors.New("Unknown " + input.Name + " sign-in method: " + text)
			})
		},
		Refresh: func(ctx context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.refresh")
				}
				return requestRadiusToken(ctx, gateway, options, Property{Name: "grant_type", Value: "refresh_token"}, Property{Name: "client_id", Value: "pi-gateway"}, Property{Name: "refresh_token", Value: catalogProperty(credential, "refresh")})
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

func radiusOAuthFailure(response *http.Response, message string) error {
	raw, _ := oauthRead(response)
	var code *string
	description := ""
	if len(raw) > 0 {
		data, err := jsonjs.DecodeValue(raw)
		if jsonjs.ValidateJSON(raw) != nil || err != nil || jsonjs.IsNullish(data) {
			description = string(raw)
		} else {
			if value, ok := catalogProperty(data, "error").(string); ok {
				code = &value
			}
			description, _ = catalogProperty(data, "error_description").(string)
		}
	}
	detail := description
	if code != nil && *code != "" {
		detail = *code
		if description != "" {
			detail += ": " + description
		}
	} else if detail == "" {
		detail = strconv.Itoa(response.StatusCode)
	}
	return &RadiusOAuthResponseError{Status: response.StatusCode, OAuthError: code, Message: message + ": " + detail}
}

func postRadiusOAuth(ctx context.Context, gateway, path string, options RadiusOAuthRuntimeOptions, fields ...Property) (*http.Response, error) {
	body, err := oauthFormEncode(fields...)
	if err != nil {
		return nil, err
	}
	endpoint, err := radiusEndpoint(gateway, path)
	var response *http.Response
	if err == nil {
		response, err = oauthFetch(ctx, options.Client, endpoint, "POST", http.Header{"Accept": {"application/json"}, "Content-Type": {"application/x-www-form-urlencoded"}}, body)
	}
	if err != nil && ctx != nil && ctx.Err() != nil {
		return nil, errors.New("Login cancelled")
	}
	return response, err
}

func requestRadiusToken(ctx context.Context, gateway string, options RadiusOAuthRuntimeOptions, fields ...Property) (any, error) {
	response, err := postRadiusOAuth(ctx, gateway, "/v1/oauth/token", options, fields...)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, radiusOAuthFailure(response, "Radius OAuth token request failed")
	}
	data, err := oauthReadJSON(response)
	if err != nil {
		return nil, err
	}
	if jsonjs.IsNullish(data) {
		return nil, jsonjs.PropertyReadError(true, "data.access_token")
	}
	seconds, err := authNumber(catalogProperty(data, "expires_in"))
	if err != nil {
		return nil, err
	}
	return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: catalogProperty(data, "access_token")}, Property{Name: "refresh", Value: catalogProperty(data, "refresh_token")}, Property{Name: "expires", Value: options.Now() + seconds*1000 - 60000}, Property{Name: "scope", Value: catalogProperty(data, "scope")}), nil
}

func loginRadiusDevice(ctx context.Context, gateway string, interaction ProviderAuthInteraction, options RadiusOAuthRuntimeOptions) (any, error) {
	response, err := postRadiusOAuth(ctx, gateway, "/v1/oauth/device", options, Property{Name: "client_id", Value: "pi-gateway"}, Property{Name: "scope", Value: "gateway offline_access"})
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, radiusOAuthFailure(response, "Radius OAuth device authorization failed")
	}
	data, err := oauthReadJSON(response)
	if err != nil {
		return nil, err
	}
	if jsonjs.IsNullish(data) {
		return nil, jsonjs.PropertyReadError(true, "data.device_code")
	}
	for _, key := range []string{"device_code", "user_code", "verification_uri", "expires_in"} {
		if !catalogEntryTruthy(catalogProperty(data, key)) {
			return nil, errors.New("Radius OAuth device authorization response is missing required fields")
		}
	}
	device, interval, expires := catalogProperty(data, "device_code"), catalogProperty(data, "interval"), catalogProperty(data, "expires_in")
	interaction.Notify(NewObject(Property{Name: "type", Value: "device_code"}, Property{Name: "userCode", Value: catalogProperty(data, "user_code")}, Property{Name: "verificationUri", Value: catalogProperty(data, "verification_uri")}, Property{Name: "intervalSeconds", Value: interval}, Property{Name: "expiresInSeconds", Value: expires}))
	return PollOAuthDeviceCodeFlow(&OAuthDeviceCodePollOptions{Context: ctx, Now: options.Now, Sleep: options.PollSleep, IntervalSeconds: interval, ExpiresInSeconds: expires, Poll: func() (OAuthDeviceCodePollResult, error) {
		value, err := requestRadiusToken(ctx, gateway, options, Property{Name: "grant_type", Value: "urn:ietf:params:oauth:grant-type:device_code"}, Property{Name: "client_id", Value: "pi-gateway"}, Property{Name: "device_code", Value: device})
		if err == nil {
			return OAuthDeviceCodePollResult{Status: "complete", Value: value}, nil
		}
		if failure, ok := err.(*RadiusOAuthResponseError); ok && failure.OAuthError != nil {
			switch *failure.OAuthError {
			case "authorization_pending":
				return OAuthDeviceCodePollResult{Status: "pending"}, nil
			case "slow_down":
				return OAuthDeviceCodePollResult{Status: "slow_down"}, nil
			case "expired_token":
				return OAuthDeviceCodePollResult{Status: "failed", Message: "Device authorization expired."}, nil
			case "access_denied":
				return OAuthDeviceCodePollResult{Status: "failed", Message: "Device authorization was denied."}, nil
			}
		}
		return OAuthDeviceCodePollResult{}, err
	}})
}

func loginRadiusBrowser(ctx context.Context, gateway string, interaction ProviderAuthInteraction, options RadiusOAuthRuntimeOptions) (any, error) {
	endpoint, err := radiusEndpoint(gateway, "/v1/oauth")
	if err != nil {
		return nil, err
	}
	response, err := oauthFetch(ctx, options.Client, endpoint, "GET", http.Header{"Accept": {"application/json"}}, "")
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		raw, err := oauthRead(response)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("Could not load Radius OAuth config from %s: %d %s", gateway, response.StatusCode, raw)
	}
	discovery, err := oauthReadJSON(response)
	if err != nil {
		return nil, err
	}
	if jsonjs.IsNullish(discovery) {
		return nil, jsonjs.PropertyReadError(true, "discovery.authorizationEndpoint")
	}
	authorization, ok := catalogProperty(discovery, "authorizationEndpoint").(string)
	if !ok {
		return nil, errors.New("Invalid Radius OAuth config from " + gateway)
	}
	pkce, err := options.PKCE()
	if err != nil {
		return nil, err
	}
	state, err := options.RandomUUID()
	if err != nil {
		return nil, err
	}
	url, err := whatwg.Parse(string(jsonjs.StringCodePoints(authorization)))
	if err != nil {
		return nil, fmt.Errorf("%q cannot be parsed as a URL.", authorization)
	}
	query, err := oauthFormEncode(Property{Name: "response_type", Value: "code"}, Property{Name: "client_id", Value: "pi-gateway"}, Property{Name: "redirect_uri", Value: radiusOAuthRedirect}, Property{Name: "scope", Value: "gateway offline_access"}, Property{Name: "code_challenge", Value: pkce.Challenge}, Property{Name: "code_challenge_method", Value: "S256"}, Property{Name: "handoff", Value: "url"}, Property{Name: "state", Value: state})
	if err != nil {
		return nil, err
	}
	url.SetSearch(query)
	callback, err := options.StartCallback(&OAuthCallbackServerOptions{ProviderName: "Radius", Host: "127.0.0.1", Port: 1456, Path: "/oauth/callback", State: &state, Context: ctx, Complete: func(code string) (any, error) {
		return requestRadiusToken(ctx, gateway, options, Property{Name: "grant_type", Value: "authorization_code"}, Property{Name: "client_id", Value: "pi-gateway"}, Property{Name: "redirect_uri", Value: radiusOAuthRedirect}, Property{Name: "code", Value: code}, Property{Name: "code_verifier", Value: pkce.Verifier})
	}})
	if err != nil {
		return nil, err
	}
	// Both notifications precede Pi's finally scope. A retaining host owns
	// listener cleanup when its notification callback throws here.
	interaction.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "Listening for OAuth callback on " + radiusOAuthRedirect}))
	interaction.Notify(NewObject(Property{Name: "type", Value: "auth_url"}, Property{Name: "url", Value: url.Href(false)}, Property{Name: "instructions", Value: "Continue in your browser."}))
	defer callback.Close()
	credential, err := callback.Wait()
	if err == nil && !catalogEntryTruthy(credential) {
		return nil, errors.New("OAuth callback did not complete.")
	}
	return credential, err
}
