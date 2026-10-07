package ai

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

const copilotOAuthClientID = "Iv1.b507a08c87ecfe98"
const copilotOAuthAgent = "GitHubCopilotChat/0.35.0"

// GitHubCopilotOAuthOptions supplies native dependencies. KnownModels is the
// source GITHUB_COPILOT_MODELS object: own keys determine which unconfigured
// account models may be enabled. The pinned source omits its generated data,
// so callers must supply this catalog rather than silently using today's list.
type GitHubCopilotOAuthOptions struct {
	KnownModels    *Object
	Client         *http.Client
	Now            func() float64
	Sleep          func(float64, context.Context) error
	PollSleep      func(float64, context.Context, string) error
	RequestTimeout time.Duration
	RetryBudget    time.Duration
}

func GitHubCopilotOAuth(options GitHubCopilotOAuthOptions) (*OAuthAuth, error) {
	if options.KnownModels == nil {
		return nil, errors.New("GitHub Copilot OAuth requires its model catalog")
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.Now == nil {
		options.Now = func() float64 { return float64(time.Now().UnixMilli()) }
	}
	if options.Sleep == nil {
		options.Sleep = oauthSleep
	}
	if options.RequestTimeout == 0 {
		options.RequestTimeout = 5 * time.Second
	}
	if options.RetryBudget == 0 {
		options.RetryBudget = 5 * time.Second
	}
	subscription := true
	return &OAuthAuth{Name: "GitHub Copilot", IsSubscription: &subscription,
		Login: func(interaction ProviderAuthInteraction, _ *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) { return loginCopilotOAuth(interaction, options) })
		},
		Refresh: func(ctx context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.refresh")
				}
				enterprise := copilotOAuthEnterprise(credential)
				value, err := copilotOAuthAccessToken(ctx, catalogProperty(credential, "refresh"), enterprise, options)
				if err != nil {
					return nil, err
				}
				models, err := copilotOAuthModels(ctx, value.Get("access"), enterprise, 0, options)
				if err != nil {
					return nil, err
				}
				value.Set("availableModelIds", models.available)
				return value, nil
			})
		},
		ToAuth: func(credential any) (any, error) {
			if jsonjs.IsNullish(credential) {
				return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.access")
			}
			access := catalogProperty(credential, "access")
			base, err := copilotOAuthBaseURL(access, copilotOAuthEnterprise(credential))
			if err != nil {
				return nil, err
			}
			return NewObject(Property{Name: "apiKey", Value: access}, Property{Name: "baseUrl", Value: base}), nil
		},
	}, nil
}
func copilotOAuthDomain(input string) string {
	domain, _ := parseCopilotOAuthDomain(input)
	return domain
}
func parseCopilotOAuthDomain(input string) (string, bool) {
	input = strings.TrimFunc(input, jsWhitespace)
	if input == "" {
		return "", false
	}
	if !strings.Contains(input, "://") {
		input = "https://" + input
	}
	parsed, err := whatwg.Parse(string(jsonjs.StringCodePoints(input)))
	if err != nil {
		return "", false
	}
	return parsed.Hostname(), true
}
func copilotOAuthEnterprise(credential any) any {
	if value, ok := catalogProperty(credential, "enterpriseUrl").(string); ok && value != "" {
		if domain, ok := parseCopilotOAuthDomain(value); ok {
			return domain
		}
	}
	return Undefined
}
func copilotOAuthBaseURL(token, enterprise any) (string, error) {
	if catalogEntryTruthy(token) {
		text, ok := token.(string)
		if !ok {
			return "", errors.New("token.match is not a function. (In 'token.match(/proxy-ep=([^;]+)/)', 'token.match' is undefined)")
		}
		// The unanchored source regexp accepts the first nonempty proxy-ep value,
		// even when the match occurs inside another token field.
		for offset := 0; offset < len(text); {
			at := strings.Index(text[offset:], "proxy-ep=")
			if at < 0 {
				break
			}
			start := offset + at + len("proxy-ep=")
			end := strings.IndexByte(text[start:], ';')
			if end < 0 {
				end = len(text) - start
			}
			host := text[start : start+end]
			if host != "" {
				if strings.HasPrefix(host, "proxy.") {
					host = "api." + strings.TrimPrefix(host, "proxy.")
				}
				return "https://" + host, nil
			}
			offset = start
		}
	}
	if catalogEntryTruthy(enterprise) {
		text, err := catalogKey(enterprise, make(map[*Array]bool))
		return "https://copilot-api." + text, err
	}
	return "https://api.individual.githubcopilot.com", nil
}
func copilotOAuthHeaders(token any) (http.Header, error) {
	value, err := catalogKey(token, make(map[*Array]bool))
	if err != nil {
		return nil, err
	}
	return http.Header{"Accept": {"application/json"}, "Authorization": {"Bearer " + value}, "User-Agent": {copilotOAuthAgent}, "Editor-Version": {"vscode/1.107.0"}, "Editor-Plugin-Version": {"copilot-chat/0.35.0"}, "Copilot-Integration-Id": {"vscode-chat"}}, nil
}
func copilotOAuthStatusError(response *http.Response) error {
	raw, err := oauthRead(response)
	if err != nil {
		return err
	}
	statusText := http.StatusText(response.StatusCode)
	if response.Status != "" {
		_, statusText, _ = strings.Cut(response.Status, " ")
	}
	return fmt.Errorf("%d %s: %s", response.StatusCode, statusText, string(raw))
}
func copilotOAuthFetchJSON(ctx context.Context, endpoint, method string, headers http.Header, body string, options GitHubCopilotOAuthOptions) (any, error) {
	response, err := oauthFetch(ctx, options.Client, endpoint, method, headers, body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, copilotOAuthStatusError(response)
	}
	return oauthReadJSON(response)
}
func copilotOAuthDate(value string) float64 {
	for _, layout := range []string{http.TimeFormat, time.RFC850, time.ANSIC, time.RFC3339Nano, "2006-01-02"} {
		if date, err := time.Parse(layout, value); err == nil {
			return float64(date.UnixMilli())
		}
	}
	return math.NaN()
}
func copilotOAuthRetry(ctx context.Context, endpoint, method string, headers http.Header, body string, retries int, options GitHubCopilotOAuthOptions) (*http.Response, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	deadline := math.Inf(1)
	if retries > 0 && options.RetryBudget > 0 {
		ctx = oauthRequestTimeout(ctx, options.RetryBudget)
		deadline = options.Now() + float64(options.RetryBudget)/float64(time.Millisecond)
	}
	for retry := 0; ; retry++ {
		response, err := oauthFetch(oauthRequestTimeout(ctx, options.RequestTimeout), options.Client, endpoint, method, headers, body)
		if err != nil {
			return nil, err
		}
		if response.StatusCode != 429 || retry == retries {
			return response, nil
		}
		delay := 500 * math.Pow(2, float64(retry))
		after := response.Header.Get("retry-after")
		if after != "" {
			seconds := completionsParseFloat(after)
			delay = seconds * 1000
			if math.IsNaN(seconds) {
				delay = copilotOAuthDate(after) - options.Now()
			}
			if math.IsNaN(delay) || math.IsInf(delay, 0) {
				return response, nil
			}
		}
		delay = math.Max(0, delay)
		if delay >= deadline-options.Now() {
			return response, nil
		}
		if err = response.Body.Close(); err != nil {
			return nil, err
		}
		if err = options.Sleep(delay, ctx); err != nil {
			return nil, err
		}
	}
}

type copilotOAuthCatalog struct{ available, policy *Array }

func parseCopilotOAuthCatalog(raw any, fallback bool, known *Object) (copilotOAuthCatalog, error) {
	result := copilotOAuthCatalog{available: NewArray(), policy: NewArray()}
	data, ok := catalogProperty(raw, "data").(*Array)
	if !ok {
		return result, errors.New("Invalid Copilot models response")
	}
	type entry struct {
		id     string
		picker bool
		state  any
	}
	entries := []entry{}
	for _, value := range data.Values() {
		id, ok := catalogProperty(value, "id").(string)
		if !ok {
			continue
		}
		if catalogProperty(catalogProperty(catalogProperty(value, "capabilities"), "supports"), "tool_calls") == false {
			continue
		}
		picker := catalogProperty(value, "model_picker_enabled") == true
		state := catalogProperty(catalogProperty(value, "policy"), "state")
		entries = append(entries, entry{id, picker, state})
		if picker && state != "disabled" {
			result.available.Append(id)
		}
	}
	useFallback := fallback && result.available.Len() == 0
	for _, entry := range entries {
		if useFallback && entry.state == "enabled" {
			result.available.Append(entry.id)
		}
		if _, exists := known.Lookup(entry.id); exists && entry.state == "unconfigured" && (entry.picker || useFallback) {
			result.policy.Append(entry.id)
		}
	}
	return result, nil
}
func copilotOAuthModels(ctx context.Context, token, enterprise any, retries int, options GitHubCopilotOAuthOptions) (copilotOAuthCatalog, error) {
	base, err := copilotOAuthBaseURL(token, enterprise)
	if err != nil {
		return copilotOAuthCatalog{}, err
	}
	headers, err := copilotOAuthHeaders(token)
	if err != nil {
		return copilotOAuthCatalog{}, err
	}
	headers.Set("X-GitHub-Api-Version", "2026-06-01")
	response, err := copilotOAuthRetry(ctx, base+"/models", "GET", headers, "", retries, options)
	if err != nil {
		return copilotOAuthCatalog{}, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return copilotOAuthCatalog{}, copilotOAuthStatusError(response)
	}
	raw, err := oauthReadJSON(response)
	if err != nil {
		return copilotOAuthCatalog{}, err
	}
	return parseCopilotOAuthCatalog(raw, base == "https://api.individual.githubcopilot.com", options.KnownModels)
}
func copilotOAuthAccessToken(ctx context.Context, refresh, enterprise any, options GitHubCopilotOAuthOptions) (*Object, error) {
	domain := "github.com"
	if catalogEntryTruthy(enterprise) {
		domain, _ = catalogKey(enterprise, make(map[*Array]bool))
	}
	headers, err := copilotOAuthHeaders(refresh)
	if err != nil {
		return nil, err
	}
	raw, err := copilotOAuthFetchJSON(ctx, "https://api."+domain+"/copilot_internal/v2/token", "GET", headers, "", options)
	if err != nil {
		return nil, err
	}
	switch raw.(type) {
	case *Object, *Array:
	default:
		return nil, errors.New("Invalid Copilot token response")
	}
	token, ok := catalogProperty(raw, "token").(string)
	expires, number := deviceCodeNumber(catalogProperty(raw, "expires_at"))
	if !ok || !number {
		return nil, errors.New("Invalid Copilot token response fields")
	}
	return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "refresh", Value: refresh}, Property{Name: "access", Value: token}, Property{Name: "expires", Value: expires*1000 - 300000}, Property{Name: "enterpriseUrl", Value: enterprise}), nil
}
func copilotOAuthEnable(ctx context.Context, token, enterprise any, id string, options GitHubCopilotOAuthOptions) (bool, error) {
	base, err := copilotOAuthBaseURL(token, enterprise)
	if err != nil {
		return false, err
	}
	headers, err := copilotOAuthHeaders(token)
	if err != nil {
		return false, err
	}
	headers.Del("Accept")
	headers.Set("Content-Type", "application/json")
	headers.Set("openai-intent", "chat-policy")
	headers.Set("x-interaction-type", "chat-policy")
	response, err := copilotOAuthRetry(ctx, base+"/models/"+id+"/policy", "POST", headers, `{"state":"enabled"}`, 2, options)
	if err != nil {
		if ctx.Err() != nil {
			return false, err
		}
		return false, nil
	}
	if response.StatusCode == 429 {
		return false, copilotOAuthStatusError(response)
	}
	defer response.Body.Close()
	return response.StatusCode >= 200 && response.StatusCode < 300, nil
}
func loginCopilotOAuth(interaction ProviderAuthInteraction, options GitHubCopilotOAuthOptions) (any, error) {
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	input, err := interaction.Prompt(context.Background(), NewObject(Property{Name: "type", Value: "text"}, Property{Name: "message", Value: "GitHub Enterprise URL/domain (blank for github.com)"}, Property{Name: "placeholder", Value: "company.ghe.com"}))
	if err != nil {
		return nil, err
	}
	if ctx.Err() != nil {
		return nil, errors.New("Login cancelled")
	}
	text, ok := input.(string)
	if !ok {
		if jsonjs.IsNullish(input) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(input), "input.trim")
		}
		return nil, errors.New("input.trim is not a function. (In 'input.trim()', 'input.trim' is undefined)")
	}
	enterprise := any(Undefined)
	domain := copilotOAuthDomain(text)
	if strings.TrimFunc(text, jsWhitespace) != "" && domain == "" {
		return nil, errors.New("Invalid GitHub Enterprise URL/domain")
	}
	if domain == "" {
		domain = "github.com"
	} else {
		enterprise = domain
	}
	headers := http.Header{"Accept": {"application/json"}, "Content-Type": {"application/x-www-form-urlencoded"}, "User-Agent": {copilotOAuthAgent}}
	body, _ := oauthFormEncode(Property{Name: "client_id", Value: copilotOAuthClientID}, Property{Name: "scope", Value: "read:user"})
	raw, err := copilotOAuthFetchJSON(ctx, "https://"+domain+"/login/device/code", "POST", headers, body, options)
	if err != nil {
		return nil, err
	}
	switch raw.(type) {
	case *Object, *Array:
	default:
		return nil, errors.New("Invalid device code response")
	}
	code, codeOK := catalogProperty(raw, "device_code").(string)
	user, userOK := catalogProperty(raw, "user_code").(string)
	uri, uriOK := catalogProperty(raw, "verification_uri").(string)
	interval := catalogProperty(raw, "interval")
	_, intervalOK := deviceCodeNumber(interval)
	expires, expiresOK := deviceCodeNumber(catalogProperty(raw, "expires_in"))
	if !codeOK || !userOK || !uriOK || (!jsonjs.IsUndefined(interval) && !intervalOK) || !expiresOK {
		return nil, errors.New("Invalid device code response fields")
	}
	parsed, err := whatwg.Parse(string(jsonjs.StringCodePoints(uri)))
	if err != nil || (parsed.Protocol() != "http:" && parsed.Protocol() != "https:") {
		return nil, errors.New("Untrusted verification_uri in device code response")
	}
	interaction.Notify(NewObject(Property{Name: "type", Value: "device_code"}, Property{Name: "userCode", Value: user}, Property{Name: "verificationUri", Value: parsed.Href(false)}, Property{Name: "intervalSeconds", Value: interval}, Property{Name: "expiresInSeconds", Value: expires}))
	refresh, err := PollOAuthDeviceCodeFlow(&OAuthDeviceCodePollOptions{Context: ctx, Now: options.Now, Sleep: options.PollSleep, IntervalSeconds: interval, ExpiresInSeconds: expires, WaitBeforeFirstPoll: true, Poll: func() (OAuthDeviceCodePollResult, error) {
		body, err := oauthFormEncode(Property{Name: "client_id", Value: copilotOAuthClientID}, Property{Name: "device_code", Value: code}, Property{Name: "grant_type", Value: "urn:ietf:params:oauth:grant-type:device_code"})
		if err != nil {
			return OAuthDeviceCodePollResult{}, err
		}
		raw, err := copilotOAuthFetchJSON(ctx, "https://"+domain+"/login/oauth/access_token", "POST", headers, body, options)
		if err != nil {
			return OAuthDeviceCodePollResult{}, err
		}
		if access, ok := catalogProperty(raw, "access_token").(string); ok {
			return OAuthDeviceCodePollResult{Status: "complete", Value: access}, nil
		}
		if kind, ok := catalogProperty(raw, "error").(string); ok {
			if kind == "authorization_pending" {
				return OAuthDeviceCodePollResult{Status: "pending"}, nil
			}
			if kind == "slow_down" {
				interval := catalogProperty(raw, "interval")
				if _, ok := deviceCodeNumber(interval); !ok {
					interval = Undefined
				}
				return OAuthDeviceCodePollResult{Status: "slow_down", IntervalSeconds: interval}, nil
			}
			message := "Device flow failed: " + kind
			if description := catalogProperty(raw, "error_description"); catalogEntryTruthy(description) {
				value, err := catalogKey(description, make(map[*Array]bool))
				if err != nil {
					return OAuthDeviceCodePollResult{}, err
				}
				message += ": " + value
			}
			return OAuthDeviceCodePollResult{Status: "failed", Message: message}, nil
		}
		return OAuthDeviceCodePollResult{Status: "failed", Message: "Invalid device token response"}, nil
	}})
	if err != nil {
		return nil, err
	}
	credential, err := copilotOAuthAccessToken(ctx, refresh, enterprise, options)
	if err != nil {
		return nil, err
	}
	catalog, err := copilotOAuthModels(ctx, credential.Get("access"), enterprise, 2, options)
	if err != nil {
		return nil, err
	}
	available := NewArray()
	seen := map[string]bool{}
	appendID := func(value any) {
		id := value.(string)
		if !seen[id] {
			seen[id] = true
			available.Append(id)
		}
	}
	for _, id := range catalog.available.Values() {
		appendID(id)
	}
	if catalog.policy.Len() > 0 {
		interaction.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "Enabling models..."}))
		for _, id := range catalog.policy.Values() {
			enabled, err := copilotOAuthEnable(ctx, credential.Get("access"), enterprise, id.(string), options)
			if err != nil {
				if ctx.Err() != nil {
					return nil, err
				}
				break
			}
			if enabled {
				appendID(id)
			}
		}
	}
	credential.Set("availableModelIds", available)
	return credential, nil
}
