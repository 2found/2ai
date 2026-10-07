package ai

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"strings"
	"syscall"
	"time"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

const chatGPTOAuthRedirect = "http://127.0.0.1:1455/auth/callback"
const chatGPTOAuthTokenURL = "https://auth.openai.com/api/accounts/oauth/token"
const chatGPTOAuthResource = "https://api.openai.com/v1"
const chatGPTOAuthDirectScope = "chatgpt.tokens.use.direct"

var chatGPTOAuthUUID = regexp.MustCompile(`(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

type OpenAIChatGPTOAuthOptions struct {
	Client        *http.Client
	PKCE          func() (PKCE, error)
	RandomValue   func() (string, error)
	StartCallback func(ChatGPTOAuthCallbackOptions) (*ChatGPTOAuthCallback, error)
	CallbackHost  *string
	Now           func() float64
}

func OpenAIChatGPTOAuth(settings ...OpenAIChatGPTOAuthOptions) *OAuthAuth {
	var options OpenAIChatGPTOAuthOptions
	if len(settings) > 0 {
		options = settings[0]
	}
	if options.Client == nil {
		options.Client = http.DefaultClient
	}
	if options.PKCE == nil {
		options.PKCE = GeneratePKCE
	}
	if options.RandomValue == nil {
		options.RandomValue = func() (string, error) {
			var b [32]byte
			if _, err := rand.Read(b[:]); err != nil {
				return "", err
			}
			return base64.RawURLEncoding.EncodeToString(b[:]), nil
		}
	}
	if options.StartCallback == nil {
		options.StartCallback = StartChatGPTOAuthCallback
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
	subscription, label := true, "Sign in with ChatGPT"
	return &OAuthAuth{Name: "OpenAI (ChatGPT subscription)", IsSubscription: &subscription, LoginLabel: &label,
		Login: func(interaction ProviderAuthInteraction, login *OAuthLoginOptions) (any, error) {
			return invokeAuth(func() (any, error) { return loginChatGPTOAuth(interaction, login, options) })
		},
		Refresh: func(ctx context.Context, credential any) (any, error) {
			return invokeAuth(func() (any, error) {
				if jsonjs.IsNullish(credential) {
					return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.clientId")
				}
				client, ok := catalogProperty(credential, "clientId").(string)
				if !ok || strings.TrimFunc(client, jsWhitespace) == "" {
					return nil, errors.New("Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT")
				}
				token, err := requestChatGPTOAuthToken(ctx, options, Property{Name: "grant_type", Value: "refresh_token"}, Property{Name: "client_id", Value: client}, Property{Name: "refresh_token", Value: catalogProperty(credential, "refresh")}, Property{Name: "resource", Value: chatGPTOAuthResource})
				if err != nil {
					return nil, err
				}
				return chatGPTOAuthCredential(token, client, options)
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
func chatGPTOAuthCallbackAuthorization(query map[string]string, state string) (ChatGPTOAuthAuthorization, error) {
	if query["code"] == "" {
		return ChatGPTOAuthAuthorization{}, errors.New("Missing authorization code")
	}
	if query["state"] == "" {
		return ChatGPTOAuthAuthorization{}, errors.New("Missing OAuth state")
	}
	if query["state"] != state {
		return ChatGPTOAuthAuthorization{}, errors.New("OAuth state mismatch")
	}
	client := strings.TrimFunc(query["client_id"], jsWhitespace)
	if client == "" {
		return ChatGPTOAuthAuthorization{}, errors.New("OpenAI OAuth registration callback did not contain an issued client ID")
	}
	return ChatGPTOAuthAuthorization{Code: query["code"], ClientID: client}, nil
}
func chatGPTOAuthManual(input any, state string) (ChatGPTOAuthAuthorization, error) {
	text, ok := input.(string)
	if !ok {
		return ChatGPTOAuthAuthorization{}, errors.New("Paste the full callback URL from the browser")
	}
	parsed, err := whatwg.Parse(string(jsonjs.StringCodePoints(strings.TrimFunc(text, jsWhitespace))))
	if err != nil {
		return ChatGPTOAuthAuthorization{}, errors.New("Paste the full callback URL from the browser")
	}
	if parsed.Scheme() != "http" || parsed.Hostname() != "127.0.0.1" || parsed.Port() != "1455" || parsed.Pathname() != "/auth/callback" {
		return ChatGPTOAuthAuthorization{}, errors.New("The pasted callback URL must start with " + chatGPTOAuthRedirect)
	}
	query := urlQueryValues(parsed.Query())
	if problem := query["error"]; problem != "" {
		return ChatGPTOAuthAuthorization{}, errors.New("ChatGPT authorization failed: " + problem)
	}
	return chatGPTOAuthCallbackAuthorization(query, state)
}
func requestChatGPTOAuthToken(ctx context.Context, options OpenAIChatGPTOAuthOptions, fields ...Property) (*Object, error) {
	body, err := oauthFormEncode(fields...)
	if err != nil {
		return nil, err
	}
	response, err := oauthFetch(ctx, options.Client, chatGPTOAuthTokenURL, "POST", http.Header{"Accept": {"application/json"}, "Content-Type": {"application/x-www-form-urlencoded"}}, body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		raw, _ := oauthRead(response)
		text := string(raw)
		if text == "" {
			text = http.StatusText(response.StatusCode)
			if response.Status != "" {
				_, text, _ = strings.Cut(response.Status, " ")
			}
		}
		return nil, fmt.Errorf("OpenAI OAuth token request failed (%d): %s", response.StatusCode, text)
	}
	value, err := oauthReadJSON(response)
	if err != nil {
		return nil, err
	}
	token, ok := value.(*Object)
	if !ok {
		return nil, errors.New("OpenAI OAuth token response must be an object")
	}
	return token, nil
}
func chatGPTOAuthCredential(token *Object, client string, options OpenAIChatGPTOAuthOptions) (any, error) {
	values := map[string]string{}
	for _, field := range []string{"access_token", "refresh_token", "scope"} {
		value, ok := token.Get(field).(string)
		if !ok || strings.TrimFunc(value, jsWhitespace) == "" {
			return nil, errors.New("OpenAI OAuth token response has invalid " + field)
		}
		values[field] = value
	}
	expiry := token.Get("expires_in")
	if !oauthPositiveNumber(expiry) {
		return nil, errors.New("OpenAI OAuth token response has invalid expires_in")
	}
	scopes := NewArray()
	direct := false
	for _, scope := range strings.FieldsFunc(values["scope"], jsWhitespace) {
		scopes.Append(scope)
		if scope == chatGPTOAuthDirectScope {
			direct = true
		}
	}
	if !direct {
		return nil, errors.New("OpenAI OAuth grant did not include " + chatGPTOAuthDirectScope)
	}
	seconds, _ := deviceCodeNumber(expiry)
	return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: values["access_token"]}, Property{Name: "refresh", Value: values["refresh_token"]}, Property{Name: "expires", Value: options.Now() + seconds*1000 - 180000}, Property{Name: "clientId", Value: client}, Property{Name: "scopes", Value: scopes}), nil
}
func loginChatGPTOAuth(interaction ProviderAuthInteraction, login *OAuthLoginOptions, options OpenAIChatGPTOAuthOptions) (any, error) {
	device := ""
	if login != nil && login.GetDeviceID != nil {
		device = login.GetDeviceID()
	}
	if !chatGPTOAuthUUID.MatchString(device) {
		return nil, errors.New("Sign in with ChatGPT requires a device ID (UUID) for this installation")
	}
	pkce, err := options.PKCE()
	if err != nil {
		return nil, err
	}
	state, err := options.RandomValue()
	if err != nil {
		return nil, err
	}
	nonce, err := options.RandomValue()
	if err != nil {
		return nil, err
	}
	callback, err := options.StartCallback(ChatGPTOAuthCallbackOptions{Host: *options.CallbackHost, Port: 1455, State: state})
	if err != nil {
		var diagnostic *OAuthDiagnosticError
		if errors.Is(err, syscall.EADDRINUSE) || (errors.As(err, &diagnostic) && diagnostic.Code == "EADDRINUSE") {
			return nil, errors.New("Port 1455 is in use, probably by an unfinished login in another pi session or by the Codex CLI. Cancel that login and try again.")
		}
		return nil, err
	}
	query, err := oauthFormEncode(Property{Name: "client_id", Value: "dynamic_agent_client"}, Property{Name: "agent_name_hint", Value: "Pi"}, Property{Name: "ext_agent_host_id", Value: "urn:uuid:" + strings.ToLower(device)}, Property{Name: "response_type", Value: "code"}, Property{Name: "redirect_uri", Value: chatGPTOAuthRedirect}, Property{Name: "resource", Value: chatGPTOAuthResource}, Property{Name: "scope", Value: "openid profile email offline_access resource.invoke " + chatGPTOAuthDirectScope}, Property{Name: "state", Value: state}, Property{Name: "code_challenge", Value: pkce.Challenge}, Property{Name: "code_challenge_method", Value: "S256"}, Property{Name: "nonce", Value: nonce})
	if err != nil {
		return nil, err
	}
	// The source notification precedes its cleanup scope.
	interaction.Notify(NewObject(Property{Name: "type", Value: "auth_url"}, Property{Name: "url", Value: "https://auth.openai.com/api/accounts/authorize?" + query}, Property{Name: "instructions", Value: "Complete sign-in in your browser. If the callback does not complete, paste the final redirect URL here."}))
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
	}
	manualCtx, abortManual := context.WithCancel(ctx)
	type authorizationResult struct {
		value ChatGPTOAuthAuthorization
		err   error
	}
	manual := make(chan authorizationResult, 1)
	go func() {
		value, err := invokeAuth(func() (any, error) {
			input, err := interaction.Prompt(manualCtx, NewObject(Property{Name: "type", Value: "manual_code"}, Property{Name: "message", Value: "Complete login in your browser, or paste the final redirect URL here:"}, Property{Name: "placeholder", Value: chatGPTOAuthRedirect}))
			if err != nil {
				return nil, err
			}
			return chatGPTOAuthManual(input, state)
		})
		var result ChatGPTOAuthAuthorization
		if err == nil {
			result = value.(ChatGPTOAuthAuthorization)
		}
		manual <- authorizationResult{result, err}
	}()
	defer func() { abortManual(); callback.Close() }()
	var result ChatGPTOAuthAuthorization
	select {
	case <-callback.Ready:
		result, err = callback.Result()
	default:
		select {
		case <-callback.Ready:
			result, err = callback.Result()
		case done := <-manual:
			result, err = done.value, done.err
		}
	}
	value, err := invokeAuth(func() (any, error) {
		if err != nil {
			return nil, err
		}
		interaction.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "Exchanging authorization code for tokens..."}))
		token, err := requestChatGPTOAuthToken(ctx, options, Property{Name: "grant_type", Value: "authorization_code"}, Property{Name: "client_id", Value: result.ClientID}, Property{Name: "code", Value: result.Code}, Property{Name: "code_verifier", Value: pkce.Verifier}, Property{Name: "redirect_uri", Value: chatGPTOAuthRedirect}, Property{Name: "resource", Value: chatGPTOAuthResource})
		if err != nil {
			return nil, err
		}
		id, ok := token.Get("id_token").(string)
		if !ok || strings.TrimFunc(id, jsWhitespace) == "" {
			return nil, errors.New("OpenAI OAuth token response did not contain an ID token")
		}
		return chatGPTOAuthCredential(token, result.ClientID, options)
	})
	if err != nil && ctx.Err() != nil {
		return nil, errors.New("Login cancelled")
	}
	return value, err
}
