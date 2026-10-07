package ai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

const metaOAuthClientID = "1031625952748946"
const metaOAuthDeviceURL = "https://auth.meta.com/oidc/device/authorization/"
const metaOAuthTokenURL = "https://auth.meta.com/oidc/device/token/"
const metaOAuthMintURL = "https://api.meta.ai/muse-code/key"

// MetaOAuthOptions supplies concrete HTTP and clock hooks. Each request has
// its own timeout; the derived signal stays live after the request returns.
type MetaOAuthOptions struct {
	Client         *http.Client
	Now            func() float64
	PollSleep      func(float64, context.Context, string) error
	RequestTimeout time.Duration
}

func MetaOAuth(settings ...MetaOAuthOptions) *OAuthAuth {
	var options MetaOAuthOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.Now == nil {
		options.Now = func() float64 { return float64(time.Now().UnixMilli()) }
	}
	if options.RequestTimeout == 0 {
		options.RequestTimeout = 30 * time.Second
	}
	subscription, label := true, "Sign in with Meta"
	return &OAuthAuth{Name: "Meta (Muse subscription)", IsSubscription: &subscription, LoginLabel: &label,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			ctx := interaction.Context
			if ctx == nil {
				ctx = context.Background()
			}
			value, err := invokeAuth(func() (any, error) { return loginMetaOAuth(ctx, interaction, options) })
			if err != nil && ctx.Err() != nil {
				return nil, errors.New("Login cancelled")
			}
			return value, err
		},
		Refresh: func(ctx context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.refresh")
				}
				return mintMetaOAuthKey(ctx, catalogProperty(credential, "refresh"), options)
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

func metaOAuthErrorDetail(body any) string {
	for _, key := range []string{"error_description", "detail", "message", "error"} {
		if text, ok := catalogProperty(body, key).(string); ok {
			if text = strings.TrimFunc(text, jsWhitespace); text != "" {
				return ": " + text
			}
		}
	}
	return ""
}

func metaOAuthTrustedURL(value any) string {
	text, ok := value.(string)
	if !ok || text == "" {
		return ""
	}
	parsed, err := whatwg.Parse(string(jsonjs.StringCodePoints(text)))
	if err != nil || (parsed.Scheme() != "https" && parsed.Scheme() != "http") {
		return ""
	}
	return parsed.Href(false)
}

func requestMetaOAuth(ctx context.Context, endpoint string, headers http.Header, body string, options MetaOAuthOptions) (*http.Response, error) {
	return oauthFetch(oauthRequestTimeout(ctx, options.RequestTimeout), options.Client, endpoint, "POST", headers, body)
}

func postMetaOAuthForm(ctx context.Context, endpoint string, options MetaOAuthOptions, fields ...Property) (*http.Response, error) {
	body, err := oauthFormEncode(fields...)
	if err != nil {
		return nil, err
	}
	return requestMetaOAuth(ctx, endpoint, http.Header{"Accept": {"application/json"}, "Content-Type": {"application/x-www-form-urlencoded"}}, body, options)
}

func mintMetaOAuthKey(ctx context.Context, identity any, options MetaOAuthOptions) (any, error) {
	token, err := catalogKey(identity, make(map[*Array]bool))
	if err != nil {
		return nil, err
	}
	response, err := requestMetaOAuth(ctx, metaOAuthMintURL, http.Header{"Accept": {"application/json"}, "Authorization": {"Bearer " + token}, "Content-Type": {"application/json"}, "X-Api-Version": {"1.0.0"}}, "{}", options)
	if err != nil {
		return nil, err
	}
	body := oauthReadObjectJSON(response)
	if response.StatusCode == 401 || response.StatusCode == 403 {
		return nil, fmt.Errorf("Meta session expired (status %d). Run `/login meta` to sign in again.%s", response.StatusCode, metaOAuthErrorDetail(body))
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("Meta API key mint failed with status %d%s", response.StatusCode, metaOAuthErrorDetail(body))
	}
	key, ok := catalogProperty(body, "api_key").(string)
	if !ok || key == "" {
		message := "Meta did not issue an API key."
		if action := metaOAuthTrustedURL(catalogProperty(body, "action_url")); action != "" {
			message += " Complete setup at " + action
		}
		return nil, errors.New(message)
	}
	return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "refresh", Value: identity}, Property{Name: "access", Value: key}, Property{Name: "expires", Value: options.Now() + 24*60*60*1000}), nil
}

func loginMetaOAuth(ctx context.Context, interaction ProviderAuthInteraction, options MetaOAuthOptions) (any, error) {
	response, err := postMetaOAuthForm(ctx, metaOAuthDeviceURL, options, Property{Name: "client_id", Value: metaOAuthClientID})
	if err != nil {
		return nil, err
	}
	body := oauthReadObjectJSON(response)
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("Meta device authorization failed with status %d%s", response.StatusCode, metaOAuthErrorDetail(body))
	}
	device, deviceOK := catalogProperty(body, "device_code").(string)
	user, userOK := catalogProperty(body, "user_code").(string)
	uri := metaOAuthTrustedURL(catalogProperty(body, "verification_uri_complete"))
	if uri == "" {
		uri = metaOAuthTrustedURL(catalogProperty(body, "verification_uri"))
	}
	if !deviceOK || device == "" || !userOK || user == "" || uri == "" {
		raw, _ := jsonjs.MarshalValue(body)
		return nil, errors.New("Invalid Meta device authorization response: " + string(raw))
	}
	interval, expires := catalogProperty(body, "interval"), catalogProperty(body, "expires_in")
	if !oauthPositiveNumber(interval) {
		interval = Undefined
	}
	if !oauthPositiveNumber(expires) {
		expires = Undefined
	}
	interaction.Notify(NewObject(Property{Name: "type", Value: "device_code"}, Property{Name: "userCode", Value: user}, Property{Name: "verificationUri", Value: uri}, Property{Name: "intervalSeconds", Value: interval}, Property{Name: "expiresInSeconds", Value: expires}))
	identity, err := PollOAuthDeviceCodeFlow(&OAuthDeviceCodePollOptions{Context: ctx, Now: options.Now, Sleep: options.PollSleep, IntervalSeconds: interval, ExpiresInSeconds: expires, WaitBeforeFirstPoll: true, Poll: func() (OAuthDeviceCodePollResult, error) {
		response, err := postMetaOAuthForm(ctx, metaOAuthTokenURL, options, Property{Name: "grant_type", Value: "urn:ietf:params:oauth:grant-type:device_code"}, Property{Name: "device_code", Value: device}, Property{Name: "client_id", Value: metaOAuthClientID})
		if err != nil {
			return OAuthDeviceCodePollResult{}, err
		}
		body := oauthReadObjectJSON(response)
		if token, ok := catalogProperty(body, "access_token").(string); ok && token != "" && response.StatusCode >= 200 && response.StatusCode < 300 {
			return OAuthDeviceCodePollResult{Status: "complete", Value: token}, nil
		}
		kind, _ := catalogProperty(body, "error").(string)
		switch kind {
		case "authorization_pending":
			return OAuthDeviceCodePollResult{Status: "pending"}, nil
		case "slow_down":
			interval := catalogProperty(body, "interval")
			if !oauthPositiveNumber(interval) {
				interval = Undefined
			}
			return OAuthDeviceCodePollResult{Status: "slow_down", IntervalSeconds: interval}, nil
		case "access_denied":
			return OAuthDeviceCodePollResult{Status: "failed", Message: "Meta login was denied."}, nil
		case "expired_token":
			return OAuthDeviceCodePollResult{Status: "failed", Message: "Meta device authorization expired. Please restart login."}, nil
		default:
			return OAuthDeviceCodePollResult{Status: "failed", Message: fmt.Sprintf("Meta device token request failed with status %d%s", response.StatusCode, metaOAuthErrorDetail(body))}, nil
		}
	}})
	if err != nil {
		return nil, err
	}
	interaction.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "Enabling Meta Model API access..."}))
	return mintMetaOAuthKey(ctx, identity, options)
}
