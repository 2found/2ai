package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
	"golang.org/x/text/encoding/unicode"
)

const xaiOAuthClientID = "b1a00492-073a-47ea-816f-4c329264a828"
const xaiOAuthDeviceURL = "https://auth.x.ai/oauth2/device/code"
const xaiOAuthTokenURL = "https://auth.x.ai/oauth2/token"
const xaiOAuthScope = "openid profile email offline_access grok-cli:access api:access"

// XaiOAuthOptions provides concrete HTTP and clock dependencies. The source has
// no per-request timeout: cancellation is owned by the caller's context.
type XaiOAuthOptions struct {
	Client *http.Client
	Now    func() float64
	Sleep  func(float64, context.Context, string) error
}

func XaiOAuth(settings ...XaiOAuthOptions) *OAuthAuth {
	var options XaiOAuthOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.Now == nil {
		options.Now = func() float64 { return float64(time.Now().UnixMilli()) }
	}
	subscription, label := true, "Sign in with SuperGrok or X Premium"
	return &OAuthAuth{Name: "xAI (Grok/X subscription)", IsSubscription: &subscription, LoginLabel: &label,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) { return loginXaiOAuth(interaction, options) })
		},
		Refresh: func(ctx context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.refresh")
				}
				refresh := catalogProperty(credential, "refresh")
				response, err := postXaiOAuthForm(ctx, xaiOAuthTokenURL, options, Property{Name: "grant_type", Value: "refresh_token"}, Property{Name: "client_id", Value: xaiOAuthClientID}, Property{Name: "refresh_token", Value: refresh})
				if err != nil {
					return nil, err
				}
				if !response.ok {
					return nil, response.failure("token refresh")
				}
				return xaiOAuthCredentials(response.body, refresh, options.Now)
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

type xaiOAuthResponse struct {
	ok     bool
	status int
	body   *Object
}

func (r xaiOAuthResponse) failure(action string) error {
	details := []string{}
	for _, key := range []string{"error", "error_description"} {
		if text, ok := r.body.Get(key).(string); ok && text != "" {
			details = append(details, text)
		}
	}
	message := fmt.Sprintf("xAI OAuth %s failed (HTTP %d)", action, r.status)
	if len(details) > 0 {
		message += ": " + strings.Join(details, ": ")
	}
	return errors.New(message)
}

func postXaiOAuthForm(ctx context.Context, endpoint string, options XaiOAuthOptions, fields ...Property) (xaiOAuthResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	body, err := oauthFormEncode(fields...)
	if err != nil {
		return xaiOAuthResponse{}, err
	}
	request, err := http.NewRequestWithContext(ctx, "POST", endpoint, strings.NewReader(body))
	if err != nil {
		return xaiOAuthResponse{}, err
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := options.Client.Do(request)
	if err != nil {
		if ctx.Err() != nil {
			return xaiOAuthResponse{}, errors.New(deviceCodeCancelMessage)
		}
		if wrapped, ok := err.(*url.Error); ok {
			err = wrapped.Err
		}
		return xaiOAuthResponse{}, err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	var parsed any
	if err == nil {
		raw, _ = unicode.UTF8BOM.NewDecoder().Bytes(raw)
		err = jsonjs.ValidateJSON(raw)
	}
	if err == nil {
		parsed, err = jsonjs.DecodeValue(raw)
	}
	if err != nil {
		if ctx.Err() != nil {
			return xaiOAuthResponse{}, errors.New(deviceCodeCancelMessage)
		}
		return xaiOAuthResponse{}, fmt.Errorf("xAI OAuth returned invalid JSON (HTTP %d)", response.StatusCode)
	}
	object, ok := parsed.(*Object)
	if !ok || object == nil {
		object = NewObject()
	}
	return xaiOAuthResponse{ok: response.StatusCode >= 200 && response.StatusCode < 300, status: response.StatusCode, body: object}, nil
}

func xaiOAuthRequiredString(body *Object, field string) (string, error) {
	value, ok := body.Get(field).(string)
	if !ok || value == "" {
		return "", fmt.Errorf("Invalid xAI OAuth response field: %s", field)
	}
	return value, nil
}
func xaiOAuthPositiveNumber(body *Object, field string) (float64, error) {
	value, ok := deviceCodeNumber(body.Get(field))
	if !ok || math.IsNaN(value) || math.IsInf(value, 0) || value <= 0 {
		return 0, fmt.Errorf("Invalid xAI OAuth response field: %s", field)
	}
	return value, nil
}
func xaiOAuthVerificationURI(raw string) (string, error) {
	parsed, err := whatwg.Parse(string(jsonjs.StringCodePoints(raw)))
	if err != nil || parsed.Protocol() != "https:" {
		return "", errors.New("Untrusted verification URI in xAI OAuth response")
	}
	return parsed.Href(false), nil
}
func xaiOAuthCredentials(body *Object, previousRefresh any, now func() float64) (any, error) {
	access, err := xaiOAuthRequiredString(body, "access_token")
	if err != nil {
		return nil, err
	}
	var refresh any
	if jsonjs.IsUndefined(catalogProperty(body, "refresh_token")) && catalogEntryTruthy(previousRefresh) {
		refresh = previousRefresh
	} else {
		refresh, err = xaiOAuthRequiredString(body, "refresh_token")
		if err != nil {
			return nil, err
		}
	}
	expiry := float64(3600)
	if !jsonjs.IsUndefined(catalogProperty(body, "expires_in")) {
		expiry, err = xaiOAuthPositiveNumber(body, "expires_in")
		if err != nil {
			return nil, err
		}
	}
	return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: access}, Property{Name: "refresh", Value: refresh}, Property{Name: "expires", Value: now() + expiry*1000 - 300000}), nil
}
func loginXaiOAuth(interaction ProviderAuthInteraction, options XaiOAuthOptions) (any, error) {
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	response, err := postXaiOAuthForm(ctx, xaiOAuthDeviceURL, options, Property{Name: "client_id", Value: xaiOAuthClientID}, Property{Name: "scope", Value: xaiOAuthScope}, Property{Name: "referrer", Value: "pi"})
	if err != nil {
		return nil, err
	}
	if !response.ok {
		return nil, response.failure("device authorization")
	}
	body := response.body
	var interval any = Undefined
	if value, ok := deviceCodeNumber(body.Get("interval")); ok && !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0 {
		interval = value
	}
	// The complete URI is validated before the required fields in the source.
	complete := ""
	if value, ok := body.Get("verification_uri_complete").(string); ok && value != "" {
		complete, err = xaiOAuthVerificationURI(value)
		if err != nil {
			return nil, err
		}
	}
	code, err := xaiOAuthRequiredString(body, "device_code")
	if err != nil {
		return nil, err
	}
	userCode, err := xaiOAuthRequiredString(body, "user_code")
	if err != nil {
		return nil, err
	}
	uri, err := xaiOAuthRequiredString(body, "verification_uri")
	if err != nil {
		return nil, err
	}
	uri, err = xaiOAuthVerificationURI(uri)
	if err != nil {
		return nil, err
	}
	expires, err := xaiOAuthPositiveNumber(body, "expires_in")
	if err != nil {
		return nil, err
	}
	if complete != "" {
		uri = complete
	}
	interaction.Notify(NewObject(Property{Name: "type", Value: "device_code"}, Property{Name: "userCode", Value: userCode}, Property{Name: "verificationUri", Value: uri}, Property{Name: "intervalSeconds", Value: interval}, Property{Name: "expiresInSeconds", Value: expires}))
	return PollOAuthDeviceCodeFlow(&OAuthDeviceCodePollOptions{Context: ctx, Now: options.Now, Sleep: options.Sleep, IntervalSeconds: interval, ExpiresInSeconds: expires, WaitBeforeFirstPoll: true, Poll: func() (OAuthDeviceCodePollResult, error) {
		response, err := postXaiOAuthForm(ctx, xaiOAuthTokenURL, options, Property{Name: "grant_type", Value: "urn:ietf:params:oauth:grant-type:device_code"}, Property{Name: "client_id", Value: xaiOAuthClientID}, Property{Name: "device_code", Value: code})
		if err != nil {
			return OAuthDeviceCodePollResult{}, err
		}
		if response.ok {
			value, err := xaiOAuthCredentials(response.body, Undefined, options.Now)
			return OAuthDeviceCodePollResult{Status: "complete", Value: value}, err
		}
		kind, _ := response.body.Get("error").(string)
		switch kind {
		case "authorization_pending":
			return OAuthDeviceCodePollResult{Status: "pending"}, nil
		case "slow_down":
			interval := any(Undefined)
			if value, ok := deviceCodeNumber(response.body.Get("interval")); ok {
				interval = value
			}
			return OAuthDeviceCodePollResult{Status: "slow_down", IntervalSeconds: interval}, nil
		case "access_denied", "authorization_denied":
			return OAuthDeviceCodePollResult{Status: "failed", Message: "xAI device authorization was denied"}, nil
		case "expired_token":
			return OAuthDeviceCodePollResult{Status: "failed", Message: "xAI device code expired"}, nil
		default:
			return OAuthDeviceCodePollResult{Status: "failed", Message: response.failure("device token polling").Error()}, nil
		}
	}})
}
