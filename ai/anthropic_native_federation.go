package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"unicode/utf16"
)

type anthropicFederationConfig struct{ Rule, Organization, IdentityFile, ServiceAccount, Workspace string }

// HasAnthropicFederationConfig reports whether Pi would select federation for
// these controls. It only resolves configuration; it never reads or exchanges
// an identity token. Hosts can use it before allowing an absent API key.
func HasAnthropicFederationConfig(provider string, rawOptions json.RawMessage) bool {
	controls := map[string]json.RawMessage{}
	if len(rawOptions) > 0 {
		if err := json.Unmarshal(rawOptions, &controls); err != nil {
			return false
		}
	}
	return anthropicFederation(provider, controls) != nil
}

func anthropicFederation(provider string, controls map[string]json.RawMessage) *anthropicFederationConfig {
	if provider != "anthropic" || anthropicRequestAuth(provider, controls) == nil {
		return nil
	}
	env, _ := samplingObject(controls["env"])
	get := func(name string) string {
		if value := samplingString(env[name]); value != "" {
			return value
		}
		return os.Getenv(name)
	}
	config := &anthropicFederationConfig{get("ANTHROPIC_FEDERATION_RULE_ID"), get("ANTHROPIC_ORGANIZATION_ID"), get("ANTHROPIC_IDENTITY_TOKEN_FILE"), get("ANTHROPIC_SERVICE_ACCOUNT_ID"), get("ANTHROPIC_WORKSPACE_ID")}
	if config.Rule == "" || config.Organization == "" || config.IdentityFile == "" {
		return nil
	}
	return config
}

// Pi keeps one current federation client, keyed by config, base URL and fetch
// identity. Go uses the concrete HTTP client pointer for that fetch identity.
var anthropicFederationClient struct {
	sync.Mutex
	key    string
	client *http.Client
	cache  *anthropicTokenCache
}

func anthropicFederationCache(config anthropicFederationConfig, baseURL string, client *http.Client, now func() int64) *anthropicTokenCache {
	encoded, _ := json.Marshal([]any{baseURL, config})
	key := string(encoded)
	anthropicFederationClient.Lock()
	defer anthropicFederationClient.Unlock()
	if anthropicFederationClient.cache != nil && anthropicFederationClient.key == key && anthropicFederationClient.client == client {
		return anthropicFederationClient.cache
	}
	cache := &anthropicTokenCache{now: func() float64 { return math.Floor(float64(now()) / 1000) }}
	cache.provider = func(bool) (anthropicAccessToken, error) {
		return exchangeAnthropicFederation(config, baseURL, client, cache.now)
	}
	anthropicFederationClient.key, anthropicFederationClient.client, anthropicFederationClient.cache = key, client, cache
	return cache
}

func anthropicRedactTokenBody(raw json.RawMessage) json.RawMessage {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return json.RawMessage(`null`)
	}
	if raw[0] == '"' {
		text := samplingString(raw)
		if json.Valid([]byte(text)) {
			return json.RawMessage(marshalSamplingString(string(anthropicRedactTokenBody([]byte(text)))))
		}
		units := utf16.Encode([]rune(text))
		if len(units) > 2000 {
			text = string(utf16.Decode(units[:2000])) + fmt.Sprintf("... <%d more chars>", len(units)-2000)
		}
		return json.RawMessage(marshalSamplingString(text))
	}
	if fields, ok := samplingObject(raw); ok {
		keep := map[string]json.RawMessage{}
		keys := []string{}
		for _, key := range samplingObjectKeys(raw) {
			if key == "error" || key == "error_description" || key == "error_uri" {
				keep[key] = fields[key]
				keys = append(keys, key)
			}
		}
		return marshalSamplingObject(keep, keys)
	}
	return json.RawMessage(`null`)
}

func exchangeAnthropicFederation(config anthropicFederationConfig, baseURL string, client *http.Client, now func() float64) (anthropicAccessToken, error) {
	fail := func(format string, args ...any) (anthropicAccessToken, error) {
		return anthropicAccessToken{}, fmt.Errorf(format, args...)
	}
	baseURL = strings.TrimRight(baseURL, "/")
	u, err := url.Parse(baseURL)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return fail("Invalid token endpoint base URL %q", baseURL)
	}
	if u.Scheme != "https" && !(u.Scheme == "http" && (strings.EqualFold(u.Hostname(), "localhost") || u.Hostname() == "127.0.0.1" || u.Hostname() == "::1")) {
		return fail("Refusing to send credential over non-https token endpoint %q", baseURL)
	}
	data, err := os.ReadFile(config.IdentityFile)
	if err != nil {
		return fail("Failed to read identity token file at %s: %s", config.IdentityFile, err)
	}
	token := strings.TrimFunc(strings.ToValidUTF8(string(data), "\ufffd"), jsWhitespace)
	if token == "" {
		return fail("Identity token file at %s is empty", config.IdentityFile)
	}
	length := estimateUTF16Length(token)
	if length > 16*1024 {
		return fail("Identity token is %d KiB, exceeds the 16 KiB assertion limit", (length+1023)/1024)
	}
	body := map[string]string{"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": token, "federation_rule_id": config.Rule, "organization_id": config.Organization}
	if config.ServiceAccount != "" {
		body["service_account_id"] = config.ServiceAccount
	}
	if config.Workspace != "" {
		body["workspace_id"] = config.Workspace
	}
	encoded, _ := json.Marshal(body)
	endpoint := baseURL + "/v1/oauth/token"
	request, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(encoded))
	if err != nil {
		return fail("Failed to reach token endpoint %s: %s", endpoint, err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Anthropic-Beta", "oauth-2025-04-20,oidc-federation-2026-04-01")
	request.Header.Set("User-Agent", "Anthropic/JS 0.129.0")
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(request)
	if err != nil {
		return fail("Failed to reach token endpoint %s: %s", endpoint, err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		text, _ := io.ReadAll(response.Body)
		redacted := samplingString(anthropicRedactTokenBody(json.RawMessage(marshalSamplingString(string(text)))))
		hint := ""
		if response.StatusCode == 401 {
			middle := ""
			if config.Workspace == "" {
				middle = "If your federation rule is scoped to multiple workspaces, set the ANTHROPIC_WORKSPACE_ID environment variable, the 'workspace_id' config key, or the `workspaceId` option. "
			}
			hint = " Ensure your federation rule matches your identity token. " + middle + "View your authentication events in the Workload identity page of Claude Console for more details."
		}
		id := ""
		if value := response.Header.Get("Request-Id"); value != "" {
			id = " (request-id " + value + ")"
		}
		return fail("Token exchange failed with status %d%s: %s%s", response.StatusCode, id, redacted, hint)
	}
	data, err = io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return anthropicAccessToken{}, err
	}
	data = bytes.TrimPrefix(data, []byte{0xef, 0xbb, 0xbf})
	if !json.Valid(data) {
		return fail("Token endpoint returned non-JSON response (status %d)", response.StatusCode)
	}
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return fail("null is not an object (evaluating 'data.access_token')")
	}
	fields, _ := samplingObject(data)
	if !samplingTruthy(fields["access_token"]) {
		return fail("Token endpoint response missing access_token: %s", anthropicRedactTokenBody(data))
	}
	if samplingTruthy(fields["token_type"]) && strings.ToLower(samplingString(fields["token_type"])) != "bearer" {
		return fail("Token endpoint response: unsupported token_type %q (want Bearer)", samplingString(fields["token_type"]))
	}
	expires := anthropicTokenExpiryNumber(fields["expires_in"])
	if math.IsNaN(expires) || math.IsInf(expires, 0) {
		return fail("Token endpoint response missing required fields: %s", anthropicRedactTokenBody(data))
	}
	expires += now()
	return anthropicAccessToken{completionsErrorString(fields["access_token"]), &expires}, nil
}

func anthropicTokenExpiryNumber(raw json.RawMessage) float64 {
	raw = bytes.TrimSpace(raw)
	switch string(raw) {
	case "":
		return math.NaN()
	case "null", "false":
		return 0
	case "true":
		return 1
	}
	if raw[0] == '{' {
		return math.NaN()
	}
	value := string(raw)
	if raw[0] == '"' || raw[0] == '[' {
		value = completionsErrorString(raw)
	}
	return ParseJSNumber(value)
}
