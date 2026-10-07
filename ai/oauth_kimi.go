package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
	"golang.org/x/text/encoding/unicode"
)

const kimiOAuthClientID = "17e5f671-d194-4dfb-9706-5516cb48c098"

// KimiCodingOAuthOptions supplies concrete HTTP, environment and clock hooks.
// RetrySleep retains the cancellation cause; PollSleep uses the poller's fixed
// cancellation message. Env is consulted anew for each login/refresh operation.
type KimiCodingOAuthOptions struct {
	Client         *http.Client
	Env            func(string) string
	Now            func() float64
	PollSleep      func(float64, context.Context, string) error
	RetrySleep     func(float64, context.Context) error
	RequestTimeout time.Duration
}

func KimiCodingOAuth(settings ...KimiCodingOAuthOptions) *OAuthAuth {
	var options KimiCodingOAuthOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.Env == nil {
		options.Env = os.Getenv
	}
	if options.Now == nil {
		options.Now = func() float64 { return float64(time.Now().UnixMilli()) }
	}
	if options.RetrySleep == nil {
		options.RetrySleep = oauthSleep
	}
	if options.RequestTimeout == 0 {
		options.RequestTimeout = 30 * time.Second
	}
	host := func() string {
		value := options.Env("KIMI_CODE_OAUTH_HOST")
		if value == "" {
			value = options.Env("KIMI_OAUTH_HOST")
		}
		if value == "" {
			value = "https://auth.kimi.com"
		}
		return strings.TrimRight(value, "/")
	}
	subscription, label := true, "Sign in with Kimi Code"
	return &OAuthAuth{Name: "Kimi Code (subscription)", IsSubscription: &subscription, LoginLabel: &label,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) { return loginKimiOAuth(host(), interaction, options) })
		},
		Refresh: func(ctx context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				endpoint := host()
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.refresh")
				}
				return refreshKimiOAuth(endpoint, catalogProperty(credential, "refresh"), ctx, options)
			})
		},
		ToAuth: func(credential any) (any, error) {
			if jsonjs.IsNullish(credential) {
				return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.access")
			}
			access, err := catalogKey(catalogProperty(credential, "access"), make(map[*Array]bool))
			if err != nil {
				return nil, err
			}
			return NewObject(Property{Name: "headers", Value: NewObject(Property{Name: "Authorization", Value: "Bearer " + access})}), nil
		},
	}
}
func postKimiOAuthForm(ctx context.Context, endpoint string, options KimiCodingOAuthOptions, fields ...Property) (*http.Response, error) {
	body, err := oauthFormEncode(fields...)
	if err != nil {
		return nil, err
	}
	requestContext := oauthRequestTimeout(ctx, options.RequestTimeout)
	request, err := http.NewRequestWithContext(requestContext, "POST", endpoint, strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.Header.Set("Accept", "application/json")
	response, err := options.Client.Do(request)
	if err != nil {
		if wrapped, ok := err.(*url.Error); ok {
			err = wrapped.Err
		}
		if (err == context.Canceled || err == context.DeadlineExceeded) && requestContext.Err() != nil {
			err = oauthCancelCause(requestContext)
		}
	}
	return response, err
}
func readKimiOAuthText(response *http.Response) string {
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return ""
	}
	raw, _ = unicode.UTF8BOM.NewDecoder().Bytes(raw)
	return string(raw)
}
func kimiOAuthJSON(value any) string { raw, _ := jsonjs.MarshalValue(value); return string(raw) }
func kimiOAuthTrustedURI(value any) bool {
	text, ok := value.(string)
	if !ok || text == "" {
		return false
	}
	parsed, err := whatwg.Parse(string(jsonjs.StringCodePoints(text)))
	return err == nil && (parsed.Protocol() == "http:" || parsed.Protocol() == "https:")
}
func kimiOAuthCredentials(body any, operation string, options KimiCodingOAuthOptions) (any, error) {
	access, accessOK := catalogProperty(body, "access_token").(string)
	refresh, refreshOK := catalogProperty(body, "refresh_token").(string)
	expiry := catalogProperty(body, "expires_in")
	if !accessOK || access == "" || !refreshOK || refresh == "" || !oauthPositiveNumber(expiry) {
		return nil, fmt.Errorf("Kimi Code token %s response missing fields: %s", operation, kimiOAuthJSON(body))
	}
	seconds, _ := deviceCodeNumber(expiry)
	return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: access}, Property{Name: "refresh", Value: refresh}, Property{Name: "expires", Value: options.Now() + seconds*1000}), nil
}
func kimiOAuthStatusError(prefix string, response *http.Response) error {
	text := readKimiOAuthText(response)
	message := fmt.Sprintf("%s with status %d", prefix, response.StatusCode)
	if text != "" {
		message += ": " + text
	}
	return errors.New(message)
}
func loginKimiOAuth(host string, interaction ProviderAuthInteraction, options KimiCodingOAuthOptions) (any, error) {
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	response, err := postKimiOAuthForm(ctx, host+"/api/oauth/device_authorization", options, Property{Name: "client_id", Value: kimiOAuthClientID})
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, kimiOAuthStatusError("Kimi Code device authorization failed", response)
	}
	body := oauthReadObjectJSON(response)
	code, codeOK := catalogProperty(body, "device_code").(string)
	user, userOK := catalogProperty(body, "user_code").(string)
	uri, complete := catalogProperty(body, "verification_uri"), catalogProperty(body, "verification_uri_complete")
	if !codeOK || !userOK || !kimiOAuthTrustedURI(complete) || !kimiOAuthTrustedURI(uri) {
		return nil, errors.New("Invalid Kimi Code device authorization response: " + kimiOAuthJSON(body))
	}
	interval, expires := catalogProperty(body, "interval"), catalogProperty(body, "expires_in")
	if !oauthPositiveNumber(interval) {
		interval = 5
	}
	if !oauthPositiveNumber(expires) {
		expires = 900
	}
	interaction.Notify(NewObject(Property{Name: "type", Value: "device_code"}, Property{Name: "userCode", Value: user}, Property{Name: "verificationUri", Value: complete}, Property{Name: "intervalSeconds", Value: interval}, Property{Name: "expiresInSeconds", Value: expires}))
	return PollOAuthDeviceCodeFlow(&OAuthDeviceCodePollOptions{Context: ctx, Now: options.Now, Sleep: options.PollSleep, IntervalSeconds: interval, ExpiresInSeconds: expires, WaitBeforeFirstPoll: true, Poll: func() (OAuthDeviceCodePollResult, error) {
		response, err := postKimiOAuthForm(ctx, host+"/api/oauth/token", options, Property{Name: "client_id", Value: kimiOAuthClientID}, Property{Name: "device_code", Value: code}, Property{Name: "grant_type", Value: "urn:ietf:params:oauth:grant-type:device_code"})
		if err != nil {
			return OAuthDeviceCodePollResult{}, err
		}
		if response.StatusCode >= 500 {
			return OAuthDeviceCodePollResult{Status: "failed", Message: kimiOAuthStatusError("Kimi Code device token request failed", response).Error()}, nil
		}
		body := oauthReadObjectJSON(response)
		if _, ok := catalogProperty(body, "access_token").(string); ok && response.StatusCode >= 200 && response.StatusCode < 300 {
			value, err := kimiOAuthCredentials(body, "poll", options)
			if err != nil {
				return OAuthDeviceCodePollResult{Status: "failed", Message: err.Error()}, nil
			}
			return OAuthDeviceCodePollResult{Status: "complete", Value: value}, nil
		}
		kind, isString := catalogProperty(body, "error").(string)
		switch kind {
		case "authorization_pending":
			return OAuthDeviceCodePollResult{Status: "pending"}, nil
		case "slow_down":
			var interval any = Undefined
			if number, ok := deviceCodeNumber(catalogProperty(body, "interval")); ok && number > 0 {
				interval = number
			}
			return OAuthDeviceCodePollResult{Status: "slow_down", IntervalSeconds: interval}, nil
		case "expired_token":
			return OAuthDeviceCodePollResult{Status: "failed", Message: "Kimi Code device authorization expired. Please restart login."}, nil
		case "access_denied":
			return OAuthDeviceCodePollResult{Status: "failed", Message: "Kimi Code login was denied."}, nil
		}
		message := fmt.Sprintf("Kimi Code device token request failed (status %d)", response.StatusCode)
		if isString {
			message += ": " + kind
			if description, ok := catalogProperty(body, "error_description").(string); ok {
				message += ": " + description
			}
		}
		return OAuthDeviceCodePollResult{Status: "failed", Message: message}, nil
	}})
}
func refreshKimiOAuth(host string, refresh any, ctx context.Context, options KimiCodingOAuthOptions) (any, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	var lastError error
	for attempt := 0; attempt <= 3; attempt++ {
		if attempt > 0 {
			if err := options.RetrySleep(float64(int(1000)<<(attempt-1)), ctx); err != nil {
				return nil, err
			}
		}
		if ctx.Err() != nil {
			return nil, errors.New("Kimi Code token refresh aborted")
		}
		response, err := postKimiOAuthForm(ctx, host+"/api/oauth/token", options, Property{Name: "client_id", Value: kimiOAuthClientID}, Property{Name: "grant_type", Value: "refresh_token"}, Property{Name: "refresh_token", Value: refresh})
		if err != nil {
			lastError = err
			continue
		}
		body := oauthReadObjectJSON(response)
		if response.StatusCode >= 200 && response.StatusCode < 300 {
			return kimiOAuthCredentials(body, "refresh", options)
		}
		kind, _ := catalogProperty(body, "error").(string)
		if response.StatusCode == 401 || response.StatusCode == 403 || kind == "invalid_grant" {
			message := fmt.Sprintf("Kimi Code token refresh unauthorized (status %d)", response.StatusCode)
			if description, ok := catalogProperty(body, "error_description").(string); ok {
				message += ": " + description
			}
			return nil, errors.New(message)
		}
		if (response.StatusCode == 429 || response.StatusCode >= 500) && attempt < 3 {
			lastError = fmt.Errorf("Kimi Code token refresh failed with status %d", response.StatusCode)
			continue
		}
		return nil, fmt.Errorf("Kimi Code token refresh failed with status %d: %s", response.StatusCode, kimiOAuthJSON(body))
	}
	if lastError == nil {
		lastError = errors.New("Kimi Code token refresh failed")
	}
	return nil, lastError
}
