package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

type openRouterOAuthFixture struct {
	UpstreamCommit string
	Metadata       json.RawMessage
	Integration    json.RawMessage
	Cases          []struct {
		Input struct {
			Manual       json.RawMessage
			Body, Action string
			Status       int
		}
		Log, Output                   json.RawMessage
		PromptAborted, RequestAborted *bool
	}
	Helpers []struct {
		Credential, Auth json.RawMessage
		RefreshSame      bool
	}
	Browser []struct {
		Mode                       string
		Output, Requests           json.RawMessage
		Calls, Status              int
		Duplicate                  *int
		PromptAborted, SuccessPage bool
	}
}

func readOpenRouterOAuthFixture(t *testing.T) openRouterOAuthFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-openrouter.json")
	if err != nil {
		t.Fatal(err)
	}
	var f openRouterOAuthFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 77 || len(f.Helpers) != 9 || len(f.Browser) != 4 {
		t.Fatal("unexpected OpenRouter OAuth coverage")
	}
	return f
}
func normalizeOAuthCallbackPort(value any) any {
	raw, err := jsonjs.MarshalValue(value)
	if err != nil {
		panic(err)
	}
	raw = regexp.MustCompile(`127\.0\.0\.1:\d+`).ReplaceAll(raw, []byte("127.0.0.1:PORT"))
	raw = regexp.MustCompile(`127\.0\.0\.1%3A\d+`).ReplaceAll(raw, []byte("127.0.0.1%3APORT"))
	out, err := jsonjs.DecodeValue(raw)
	if err != nil {
		panic(err)
	}
	return out
}

type oauthTestBody struct{ read func([]byte) (int, error) }

func (b *oauthTestBody) Read(p []byte) (int, error) { return b.read(p) }
func (b *oauthTestBody) Close() error               { return nil }
func fixedOpenRouterOAuthOptions() OpenRouterOAuthOptions {
	return OpenRouterOAuthOptions{PKCE: func() (PKCE, error) { return pkceFromBytes([32]byte{}), nil }, RandomUUID: func() (string, error) { return "12345678-1234-4234-8234-123456789abc", nil }, CallbackHost: func() string { return "127.0.0.1" }}
}
func TestPiOpenRouterOAuthLogin(t *testing.T) {
	for i, tc := range readOpenRouterOAuthFixture(t).Cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			log := NewArray()
			var mu sync.Mutex
			record := func(value any) { mu.Lock(); log.Append(value); mu.Unlock() }
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			if tc.Input.Action == "preabort" {
				cancel(errors.New("original cause"))
			}
			failure := errors.New("injected failure")
			var promptContext, requestContext context.Context
			promptDone := make(chan struct{})
			options := fixedOpenRouterOAuthOptions()
			if strings.Contains(tc.Input.Action, "timeout") {
				options.ExchangeTimeout = 10 * time.Millisecond
			}
			options.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				requestContext = request.Context()
				raw, err := io.ReadAll(request.Body)
				if err != nil {
					return nil, err
				}
				record(NewObject(Property{Name: "request", Value: NewObject(Property{Name: "url", Value: request.URL.String()}, Property{Name: "method", Value: request.Method}, Property{Name: "headers", Value: NewObject(Property{Name: "accept", Value: request.Header.Get("accept")}, Property{Name: "content-type", Value: request.Header.Get("content-type")})}, Property{Name: "body", Value: string(raw)})}))
				if strings.HasPrefix(tc.Input.Action, "fetch-abort") {
					cancel(errors.New("original cause"))
				}
				if tc.Input.Action == "fetch-error" || tc.Input.Action == "fetch-abort-error" {
					return nil, failure
				}
				if tc.Input.Action == "fetch-timeout" {
					<-request.Context().Done()
					return nil, failure
				}
				reader := strings.NewReader(tc.Input.Body)
				var once sync.Once
				body := &oauthTestBody{read: func(p []byte) (int, error) {
					once.Do(func() {
						if strings.HasPrefix(tc.Input.Action, "body-abort") {
							cancel(errors.New("original cause"))
						}
					})
					if tc.Input.Action == "body-error" || tc.Input.Action == "body-abort-error" {
						return 0, failure
					}
					if tc.Input.Action == "body-timeout" {
						<-request.Context().Done()
						return 0, failure
					}
					return reader.Read(p)
				}}
				return &http.Response{StatusCode: tc.Input.Status, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: body}, nil
			})}
			auth := OpenRouterOAuth(options)
			value, err := auth.Login(ProviderAuthInteraction{Context: ctx, Notify: func(event *Object) {
				record(NewObject(Property{Name: "notify", Value: normalizeOAuthCallbackPort(event)}))
				if tc.Input.Action == "notify-error" {
					panic(failure)
				}
				if tc.Input.Action == "notify-abort" && event.Get("type") == "auth_url" {
					cancel(errors.New("original cause"))
				}
			}, Prompt: func(c context.Context, prompt *Object) (any, error) {
				defer close(promptDone)
				promptContext = c
				record(NewObject(Property{Name: "prompt", Value: normalizeOAuthCallbackPort(prompt)}))
				if tc.Input.Action == "prompt-error" {
					return nil, failure
				}
				if tc.Input.Action == "prompt-abort" {
					cancel(errors.New("original cause"))
				}
				if len(tc.Input.Manual) == 0 {
					return Undefined, nil
				}
				return catalogDecode(t, tc.Input.Manual), nil
			}}, nil)
			if tc.Input.Action == "post-success-abort" {
				cancel(errors.New("late abort"))
			}
			if tc.PromptAborted != nil {
				select {
				case <-promptDone:
				case <-time.After(time.Second):
					t.Fatal("prompt not called")
				}
			}
			output := callbackCapture(value, err)
			if err != nil {
				output["same"] = err == failure
			}
			catalogCompare(t, output, tc.Output)
			mu.Lock()
			catalogCompare(t, log, tc.Log)
			mu.Unlock()
			for _, pair := range []struct {
				ctx      context.Context
				expected *bool
			}{{promptContext, tc.PromptAborted}, {requestContext, tc.RequestAborted}} {
				if pair.expected == nil {
					if pair.ctx != nil {
						t.Fatal("unexpected context")
					}
				} else if pair.ctx == nil || (pair.ctx.Err() != nil) != *pair.expected {
					t.Fatalf("context aborted mismatch: %v, expected %v", pair.ctx, *pair.expected)
				}
			}
		})
	}
}
func TestPiOpenRouterOAuthMetadataAndCredentials(t *testing.T) {
	fixture := readOpenRouterOAuthFixture(t)
	auth := OpenRouterOAuth()
	var subscription any
	if auth.IsSubscription != nil {
		subscription = *auth.IsSubscription
	}
	catalogCompare(t, map[string]any{"name": auth.Name, "loginLabel": *auth.LoginLabel, "isSubscription": subscription}, fixture.Metadata)
	for _, tc := range fixture.Helpers {
		credential := catalogDecode(t, tc.Credential)
		if object, ok := credential.(*Object); ok && object.Get("absent") == true {
			credential = Undefined
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		refreshed, err := auth.Refresh(ctx, credential)
		if err != nil || (catalogStrictEqual(refreshed, credential)) != tc.RefreshSame {
			t.Fatalf("refresh identity: %v", err)
		}
		value, err := auth.ToAuth(credential)
		catalogCompare(t, callbackCapture(value, err), tc.Auth)
	}
}

func TestPiOpenRouterOAuthBrowserHTTP(t *testing.T) {
	for _, tc := range readOpenRouterOAuthFixture(t).Browser {
		t.Run(tc.Mode, func(t *testing.T) {
			started, release, promptDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var startOnce, releaseOnce sync.Once
			var mu sync.Mutex
			requests := NewArray()
			calls := 0
			tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != "POST" || r.URL.Path != "/api/v1/auth/keys" || r.Header.Get("Accept") != "application/json" || r.Header.Get("Content-Type") != "application/json" {
					t.Error("token request differs")
				}
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				request, err := jsonjs.DecodeValue(body)
				if err != nil {
					t.Error(err)
				}
				mu.Lock()
				calls++
				requests.Append(request)
				mu.Unlock()
				startOnce.Do(func() { close(started) })
				if tc.Mode == "duplicate" {
					<-release
				}
				w.Header().Set("Content-Type", "application/json")
				switch tc.Mode {
				case "failure":
					w.WriteHeader(401)
					_, _ = io.WriteString(w, `{"message":"denied"}`)
				case "missing-key":
					_, _ = io.WriteString(w, `{}`)
				default:
					_, _ = io.WriteString(w, `{"key":"browser-key"}`)
				}
			}))
			defer tokenServer.Close()
			defer releaseOnce.Do(func() { close(release) })
			localURL, err := url.Parse(tokenServer.URL)
			if err != nil {
				t.Fatal(err)
			}
			nativeTransport := tokenServer.Client().Transport
			options := fixedOpenRouterOAuthOptions()
			options.Client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if r.URL.String() != openRouterTokenURL {
					t.Error("token URL changed")
				}
				copy := r.Clone(r.Context())
				copy.URL.Scheme, copy.URL.Host = localURL.Scheme, localURL.Host
				return nativeTransport.RoundTrip(copy)
			})}
			browserClient := &http.Client{Timeout: 3 * time.Second}
			defer browserClient.CloseIdleConnections()
			var callbackURL string
			var promptContext context.Context
			responseDone := make(chan struct {
				status int
				body   string
				err    error
			}, 1)
			interaction := ProviderAuthInteraction{Context: context.Background(), Notify: func(event *Object) {
				if event.Get("type") != "auth_url" {
					return
				}
				authorize, err := url.Parse(event.Get("url").(string))
				if err != nil {
					t.Error(err)
					return
				}
				callbackURL = authorize.Query().Get("callback_url")
				go func() {
					response, err := browserClient.Get(callbackURL + "?code=browser-code")
					if err != nil {
						responseDone <- struct {
							status int
							body   string
							err    error
						}{err: err}
						return
					}
					defer response.Body.Close()
					body, err := io.ReadAll(response.Body)
					responseDone <- struct {
						status int
						body   string
						err    error
					}{response.StatusCode, string(body), err}
				}()
			}, Prompt: func(ctx context.Context, _ *Object) (any, error) {
				defer close(promptDone)
				promptContext = ctx
				<-ctx.Done()
				return nil, errors.New("manual aborted")
			}}
			loginDone := make(chan struct {
				value any
				err   error
			}, 1)
			go func() {
				v, e := OpenRouterOAuth(options).Login(interaction, nil)
				loginDone <- struct {
					value any
					err   error
				}{v, e}
			}()
			select {
			case <-started:
			case <-time.After(3 * time.Second):
				t.Fatal("exchange never started")
			}
			var duplicate any
			if tc.Mode == "duplicate" {
				response, err := browserClient.Get(callbackURL + "?code=second")
				if err != nil {
					t.Fatal(err)
				}
				duplicate = response.StatusCode
				response.Body.Close()
				releaseOnce.Do(func() { close(release) })
			}
			result := <-loginDone
			response := <-responseDone
			if response.err != nil {
				t.Fatal(response.err)
			}
			<-promptDone
			catalogCompare(t, callbackCapture(result.value, result.err), tc.Output)
			mu.Lock()
			catalogCompare(t, requests, tc.Requests)
			if calls != tc.Calls {
				t.Errorf("exchange calls: %d", calls)
			}
			mu.Unlock()
			if response.status != tc.Status || strings.Contains(response.body, "Authentication successful") != tc.SuccessPage || (promptContext.Err() != nil) != tc.PromptAborted {
				t.Fatalf("callback response: %d, prompt: %v", response.status, promptContext.Err())
			}
			if tc.Duplicate != nil {
				if duplicate != *tc.Duplicate {
					t.Fatal("duplicate callback admitted")
				}
			} else if duplicate != nil {
				t.Fatal("unexpected duplicate")
			}
		})
	}
}

func TestPiOpenRouterOAuthModelsIntegration(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryCredentialStore()
	models := NewModels(ModelsOptions{Credentials: &CredentialPersistence{Read: store.Read, List: store.List, Modify: store.Modify, Delete: store.Delete}, AuthContext: &AuthContext{Env: func(context.Context, string) (any, error) { return Undefined, nil }, FileExists: func(context.Context, string) (bool, error) { return false, nil }}})
	options := fixedOpenRouterOAuthOptions()
	options.Client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"key":"stored-key"}`)), Header: make(http.Header)}, nil
	})}
	loads := 0
	auth := LazyOAuth(&LazyOAuthOptions{Name: "OpenRouter OAuth", Load: func() (*OAuthAuth, error) { loads++; return OpenRouterOAuth(options), nil }})
	chat := NewObject(Property{Name: "id", Value: "chat"}, Property{Name: "provider", Value: "openrouter"}, Property{Name: "api", Value: "test"})
	image := NewObject(Property{Name: "id", Value: "image"}, Property{Name: "provider", Value: "openrouter"}, Property{Name: "api", Value: "test-image"}, Property{Name: "type", Value: "image"})
	provider, err := NewModelProvider(&ProviderFactoryOptions{ID: "openrouter", Auth: &ProviderAuth{APIKey: EnvAPIKeyAuth("OpenRouter API key", NewArray("OPENROUTER_API_KEY")), OAuth: auth}, Models: NewArray(chat, image), API: &ProviderStreams{}})
	if err != nil {
		t.Fatal(err)
	}
	models.SetProvider(provider)
	credential, err := models.Login("openrouter", "oauth", ProviderAuthInteraction{Context: ctx, Notify: func(*Object) {}, Prompt: func(context.Context, *Object) (any, error) { return "stored-code", nil }})
	if err != nil {
		t.Fatal(err)
	}
	stored, err := store.Read(ctx, "openrouter")
	if err != nil {
		t.Fatal(err)
	}
	chatAuth, err := models.GetAuth(ctx, chat)
	if err != nil {
		t.Fatal(err)
	}
	imageAuth, err := models.GetAuth(ctx, image)
	if err != nil {
		t.Fatal(err)
	}
	check, err := models.CheckAuth(ctx, "openrouter")
	if err != nil {
		t.Fatal(err)
	}
	if err = models.Logout(ctx, "openrouter"); err != nil {
		t.Fatal(err)
	}
	after, err := models.GetAuth(ctx, chat)
	if err != nil {
		t.Fatal(err)
	}
	catalogCompare(t, map[string]any{"loads": loads, "credential": credential, "storedSame": stored == credential, "chatAuth": chatAuth, "imageAuth": imageAuth, "check": check, "afterLogout": callbackCapture(after, nil)}, readOpenRouterOAuthFixture(t).Integration)
}
