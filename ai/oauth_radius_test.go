package ai

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

type radiusOAuthCase struct {
	Input struct {
		Name, Operation, Action, Gateway string
		Method, Credential               json.RawMessage
		Steps                            []struct {
			Body, Action string
			Status       int
		}
	}
	Log, Output json.RawMessage
}
type radiusOAuthFixture struct {
	UpstreamCommit, State string
	Metadata, Integration json.RawMessage
	Cases                 []radiusOAuthCase
	Helpers               []struct{ Credential, Output json.RawMessage }
	Normalized            []struct{ Input, Output string }
	Browser               []struct {
		Mode                    string
		Requests, Pages, Output json.RawMessage
	}
}

func readRadiusOAuthFixture(t *testing.T) radiusOAuthFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-radius.json")
	if err != nil {
		t.Fatal(err)
	}
	var f radiusOAuthFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 363 || len(f.Browser) != 5 || len(f.Helpers) != 7 || len(f.Normalized) != 10 || len(f.Integration) == 0 {
		t.Fatal("unexpected Radius coverage")
	}
	return f
}
func radiusTestOptions(f radiusOAuthFixture) RadiusOAuthRuntimeOptions {
	return RadiusOAuthRuntimeOptions{Now: func() float64 { return 1000000 }, PKCE: func() (PKCE, error) { return pkceFromBytes([32]byte{}), nil }, RandomUUID: func() (string, error) { return f.State, nil }}
}
func radiusTestHeaders(r *http.Request) *Object {
	headers := NewObject(Property{Name: "accept", Value: r.Header.Get("Accept")})
	if r.Header.Get("Content-Type") != "" {
		headers.Set("content-type", r.Header.Get("Content-Type"))
	}
	return headers
}
func TestPiRadiusOAuth(t *testing.T) {
	f := readRadiusOAuthFixture(t)
	for i, tc := range f.Cases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runRadiusOAuthCase(t, f, tc, false) })
	}
}
func TestPiRadiusOAuthHTTP(t *testing.T) {
	f := readRadiusOAuthFixture(t)
	for i, tc := range f.Cases {
		if tc.Input.Name != "success" && tc.Input.Name != "pending-slow-down" && tc.Input.Name != "slow-timeout" && !strings.HasPrefix(tc.Input.Name, "raw:") && !strings.HasPrefix(tc.Input.Name, "status:401") {
			continue
		}
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runRadiusOAuthCase(t, f, tc, true) })
	}
}
func runRadiusOAuthCase(t *testing.T, f radiusOAuthFixture, tc radiusOAuthCase, realHTTP bool) {
	t.Helper()
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	failure, reason := errors.New("injected failure"), errors.New("original cause")
	action := tc.Input.Action
	if action == "preabort" {
		cancel(reason)
	}
	now, index := float64(1000000), 0
	log := NewArray()
	options := radiusTestOptions(f)
	options.Now = func() float64 { return now }
	options.PollSleep = func(ms float64, ctx context.Context, message string) error {
		if ctx.Err() != nil {
			return errors.New(message)
		}
		log.Append(NewObject(Property{Name: "sleep", Value: ms}))
		now += ms
		if action == "sleep-abort" {
			cancel(reason)
			return errors.New(message)
		}
		return nil
	}
	if action == "pkce-error" {
		options.PKCE = func() (PKCE, error) { return PKCE{}, failure }
	}
	if action == "uuid-error" {
		options.RandomUUID = func() (string, error) { return "", failure }
	}
	options.StartCallback = func(input *OAuthCallbackServerOptions) (*OAuthCallbackServer, error) {
		var timeout any
		if input.TimeoutMS != nil {
			timeout = *input.TimeoutMS
		}
		log.Append(NewObject(Property{Name: "callback", Value: NewObject(Property{Name: "providerName", Value: input.ProviderName}, Property{Name: "host", Value: input.Host}, Property{Name: "port", Value: input.Port}, Property{Name: "path", Value: input.Path}, Property{Name: "state", Value: *input.State}, Property{Name: "aborted", Value: input.Context.Err() != nil}, Property{Name: "timeout", Value: timeout})}))
		if action == "bind-error" {
			return nil, failure
		}
		if input.Context.Err() != nil {
			return nil, errors.New("Login cancelled")
		}
		return &OAuthCallbackServer{RedirectURI: radiusOAuthRedirect, Wait: func() (any, error) {
			log.Append(NewObject(Property{Name: "wait", Value: true}))
			switch action {
			case "wait-error":
				return nil, failure
			case "wait-null":
				return nil, nil
			case "wait-false":
				return false, nil
			case "wait-undefined":
				return Undefined, nil
			}
			return input.Complete("browser +&雪")
		}, Close: func() { log.Append(NewObject(Property{Name: "close", Value: true})) }}, nil
	}
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		body := []byte{}
		if r.Body != nil {
			var err error
			body, err = io.ReadAll(r.Body)
			if err != nil {
				return nil, err
			}
		}
		log.Append(NewObject(Property{Name: "request", Value: NewObject(Property{Name: "url", Value: r.URL.String()}, Property{Name: "method", Value: r.Method}, Property{Name: "headers", Value: radiusTestHeaders(r)}, Property{Name: "body", Value: string(body)}, Property{Name: "aborted", Value: r.Context().Err() != nil})}))
		if index >= len(tc.Input.Steps) {
			return nil, errors.New("unexpected request")
		}
		step := tc.Input.Steps[index]
		index++
		if strings.HasPrefix(step.Action, "fetch-abort") {
			cancel(reason)
		}
		if step.Action == "fetch-error" || step.Action == "fetch-abort-error" {
			return nil, failure
		}
		reader := strings.NewReader(step.Body)
		return &http.Response{StatusCode: step.Status, Header: make(http.Header), Body: &oauthTestBody{read: func(p []byte) (int, error) {
			if strings.HasPrefix(step.Action, "read-abort") {
				cancel(reason)
			}
			if step.Action == "read-error" || step.Action == "read-abort-error" {
				return 0, failure
			}
			return reader.Read(p)
		}}}, nil
	})
	if realHTTP {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r.URL.Scheme = "https"
			r.URL.Host = "radius.test"
			response, err := transport.RoundTrip(r)
			if err != nil {
				t.Error(err)
				w.WriteHeader(500)
				return
			}
			defer response.Body.Close()
			w.WriteHeader(response.StatusCode)
			if _, err = io.Copy(w, response.Body); err != nil {
				t.Error(err)
			}
		}))
		defer server.Close()
		endpoint, err := url.Parse(server.URL)
		if err != nil {
			t.Fatal(err)
		}
		client := server.Client()
		options.Client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			clone := r.Clone(r.Context())
			copyURL := *r.URL
			copyURL.Scheme, copyURL.Host = endpoint.Scheme, endpoint.Host
			clone.URL = &copyURL
			return client.Transport.RoundTrip(clone)
		})}
	} else {
		options.Client = &http.Client{Transport: transport}
	}
	input := &RadiusOAuthOptions{Name: "Radius test", Gateway: tc.Input.Gateway}
	auth := RadiusOAuth(input, options)
	if action == "rename" {
		input.Name = "Changed name"
		input.Gateway = "ignored.test"
	}
	var value any
	var err error
	if tc.Input.Operation == "refresh" {
		credential := catalogDecode(t, tc.Input.Credential)
		if catalogProperty(credential, "absent") == true {
			credential = Undefined
		}
		value, err = auth.Refresh(ctx, credential)
	} else {
		value, err = auth.Login(ProviderAuthInteraction{Context: ctx, Prompt: func(promptCtx context.Context, prompt *Object) (any, error) {
			if promptCtx.Err() != nil || promptCtx.Done() != nil {
				t.Error("method selection inherited login cancellation")
			}
			log.Append(NewObject(Property{Name: "prompt", Value: prompt}))
			if action == "prompt-error" {
				return nil, failure
			}
			if action == "prompt-abort" {
				cancel(reason)
			}
			method := catalogDecode(t, tc.Input.Method)
			if catalogProperty(method, "absent") == true {
				method = Undefined
			}
			return method, nil
		}, Notify: func(event *Object) {
			log.Append(NewObject(Property{Name: "notify", Value: event}))
			if action == "notify-abort" || action == "notify-abort-error" {
				cancel(reason)
			}
			if action == "notify-error" || action == "notify-abort-error" || action == "auth-url-error" && event.Get("type") == "auth_url" {
				panic(failure)
			}
		}}, nil)
	}
	output := NewObject(Property{Name: "value", Value: value})
	if err != nil {
		output = NewObject(Property{Name: "error", Value: err.Error()}, Property{Name: "same", Value: err == failure})
		if responseError, ok := err.(*RadiusOAuthResponseError); ok {
			output.Set("status", responseError.Status)
			if responseError.OAuthError != nil {
				output.Set("oauthError", *responseError.OAuthError)
			}
		}
	}
	catalogCompare(t, output, tc.Output)
	catalogCompare(t, log, tc.Log)
}
func TestPiRadiusOAuthMetadataAndHelpers(t *testing.T) {
	f := readRadiusOAuthFixture(t)
	auth := RadiusOAuth(&RadiusOAuthOptions{Name: "Radius test", Gateway: "radius.test"})
	if auth.IsSubscription != nil || auth.LoginLabel != nil {
		t.Fatal("unexpected Radius metadata")
	}
	catalogCompare(t, map[string]any{"name": auth.Name}, f.Metadata)
	for i, tc := range f.Helpers {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			credential := catalogDecode(t, tc.Credential)
			if catalogProperty(credential, "absent") == true {
				credential = Undefined
			}
			value, err := auth.ToAuth(credential)
			catalogCompare(t, callbackCapture(value, err), tc.Output)
		})
	}
	for _, tc := range f.Normalized {
		if got := NormalizeRadiusGatewayURL(tc.Input); got != tc.Output {
			t.Fatalf("normalize %q: got %q, want %q", tc.Input, got, tc.Output)
		}
	}
}

func TestPiRadiusOAuthBrowserHTTP(t *testing.T) {
	f := readRadiusOAuthFixture(t)
	for _, tc := range f.Browser {
		t.Run(tc.Mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			ready, started, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var releaseOnce sync.Once
			defer releaseOnce.Do(func() { close(release) })
			var mu sync.Mutex
			requests := NewArray()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
					return
				}
				mu.Lock()
				requests.Append(NewObject(Property{Name: "url", Value: "https://radius.test" + r.URL.RequestURI()}, Property{Name: "method", Value: r.Method}, Property{Name: "body", Value: string(body)}, Property{Name: "headers", Value: radiusTestHeaders(r)}))
				mu.Unlock()
				if r.URL.Path == "/v1/oauth" {
					_, _ = io.WriteString(w, `{"authorizationEndpoint":"https://auth.test/authorize?old=yes#fragment"}`)
					return
				}
				close(started)
				<-release
				if tc.Mode == "token-failure" {
					w.WriteHeader(401)
					_, _ = io.WriteString(w, "denied")
					return
				}
				_, _ = io.WriteString(w, `{"access_token":"access","refresh_token":"refresh","expires_in":3600,"scope":"gateway offline_access"}`)
			}))
			defer server.Close()
			endpoint, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			client := server.Client()
			options := radiusTestOptions(f)
			options.Client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				clone := r.Clone(r.Context())
				copyURL := *r.URL
				copyURL.Scheme, copyURL.Host = endpoint.Scheme, endpoint.Host
				clone.URL = &copyURL
				return client.Transport.RoundTrip(clone)
			})}
			var callback *OAuthCallbackServer
			options.StartCallback = func(input *OAuthCallbackServerOptions) (*OAuthCallbackServer, error) {
				input.Port = 0
				var err error
				callback, err = StartOAuthCallbackServer(input)
				return callback, err
			}
			done := make(chan any, 1)
			go func() {
				value, err := RadiusOAuth(&RadiusOAuthOptions{Name: "Radius test", Gateway: "radius.test"}, options).Login(ProviderAuthInteraction{Context: ctx, Prompt: func(context.Context, *Object) (any, error) { return "browser", nil }, Notify: func(event *Object) {
					if event.Get("type") == "auth_url" {
						close(ready)
					}
				}}, nil)
				done <- callbackCapture(value, err)
			}()
			publicationAwait(t, ready)
			defer callback.Close()
			browser := &http.Client{Timeout: 2 * time.Second}
			defer browser.CloseIdleConnections()
			call := func(path string) (any, error) {
				response, err := browser.Get(strings.TrimSuffix(callback.RedirectURI, "/oauth/callback") + path)
				if err != nil {
					return nil, err
				}
				defer response.Body.Close()
				body, err := io.ReadAll(response.Body)
				if err != nil {
					return nil, err
				}
				return NewObject(Property{Name: "path", Value: path}, Property{Name: "status", Value: response.StatusCode}, Property{Name: "contentType", Value: response.Header.Get("Content-Type")}, Property{Name: "cacheControl", Value: response.Header.Get("Cache-Control")}, Property{Name: "hash", Value: fmt.Sprintf("%x", sha256.Sum256(body))}), nil
			}
			pages := NewArray()
			if tc.Mode == "validation" {
				for _, path := range []string{"/wrong", "/oauth/callback", "/oauth/callback?state=" + f.State} {
					page, err := call(path)
					if err != nil {
						t.Fatal(err)
					}
					pages.Append(page)
				}
			}
			if tc.Mode == "provider-error" {
				page, err := call("/oauth/callback?state=" + f.State + "&error=denied&error_description=%3Cfailure%3E")
				if err != nil {
					t.Fatal(err)
				}
				pages.Append(page)
				releaseOnce.Do(func() { close(release) })
			} else {
				path := "/oauth/callback?state=" + f.State + "&code=browser+%2B%26%E9%9B%AA"
				type result struct {
					page any
					err  error
				}
				first := make(chan result, 1)
				go func() { page, err := call(path); first <- result{page, err} }()
				publicationAwait(t, started)
				if tc.Mode == "duplicate" {
					page, err := call(path)
					if err != nil {
						t.Fatal(err)
					}
					pages.Append(page)
				}
				releaseOnce.Do(func() { close(release) })
				response := publicationAwait(t, first)
				if response.err != nil {
					t.Fatal(response.err)
				}
				ordered := NewArray(response.page)
				for _, page := range pages.Values() {
					ordered.Append(page)
				}
				pages = ordered
			}
			output := publicationAwait(t, done)
			catalogCompare(t, output, tc.Output)
			catalogCompare(t, pages, tc.Pages)
			mu.Lock()
			catalogCompare(t, requests, tc.Requests)
			mu.Unlock()
		})
	}
}

func TestPiRadiusOAuthModelsConcurrentRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := NewInMemoryCredentialStore()
	_, err := store.Modify(ctx, "radius", func(any) (any, error) {
		return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "refresh", Value: "stored-refresh"}, Property{Name: "expires", Value: 0}), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	readBoth, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	var mu sync.Mutex
	reads, loads, requests := 0, 0, 0
	unauthorized := false
	persistence := &CredentialPersistence{Modify: store.Modify, List: store.List, Delete: store.Delete, Read: func(ctx context.Context, id string) (any, error) {
		value, err := store.Read(ctx, id)
		mu.Lock()
		reads++
		if reads == 2 {
			close(readBoth)
		}
		mu.Unlock()
		return value, err
	}}
	models := NewModels(ModelsOptions{Credentials: persistence, AuthContext: &AuthContext{Env: func(context.Context, string) (any, error) { return Undefined, nil }, FileExists: func(context.Context, string) (bool, error) { return false, nil }}})
	options := RadiusOAuthRuntimeOptions{Now: func() float64 { return 1000000 }}
	options.Client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		mu.Lock()
		requests++
		mu.Unlock()
		<-release
		if unauthorized {
			return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":"invalid_grant"}`))}, nil
		}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"access_token":"rotated","refresh_token":"refresh","expires_in":3600,"scope":"gateway offline_access"}`))}, nil
	})}
	auth := LazyOAuth(&LazyOAuthOptions{Name: "Radius", Load: func() (*OAuthAuth, error) {
		mu.Lock()
		loads++
		mu.Unlock()
		return RadiusOAuth(&RadiusOAuthOptions{Name: "Radius test", Gateway: "radius.test"}, options), nil
	}})
	provider, err := NewModelProvider(&ProviderFactoryOptions{ID: "radius", Models: NewArray(), API: &ProviderStreams{}, Auth: &ProviderAuth{OAuth: auth}})
	if err != nil {
		t.Fatal(err)
	}
	models.SetProvider(provider)
	type completion struct {
		index int
		value any
		err   error
	}
	done := make(chan completion, 2)
	for index := range 2 {
		go func() {
			value, err := models.GetAuth(ctx, "radius", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
			done <- completion{index, value, err}
		}()
	}
	publicationAwait(t, readBoth)
	releaseOnce.Do(func() { close(release) })
	results := NewArray(nil, nil)
	for range 2 {
		result := publicationAwait(t, done)
		if result.err != nil {
			t.Fatal(result.err)
		}
		results.Set(result.index, result.value)
	}
	stored, err := store.Read(ctx, "radius")
	if err != nil {
		t.Fatal(err)
	}
	check, err := models.CheckAuth(ctx, "radius")
	if err != nil {
		t.Fatal(err)
	}
	unauthorized = true
	original := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "refresh", Value: "stored-refresh"}, Property{Name: "expires", Value: 0})
	if _, err = store.Modify(ctx, "radius", func(any) (any, error) { return original, nil }); err != nil {
		t.Fatal(err)
	}
	_, failure := models.GetAuth(ctx, "radius", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
	var modelError *ModelsError
	if !errors.As(failure, &modelError) || modelError.Cause == nil {
		t.Fatalf("missing OAuth failure: %v", failure)
	}
	retained, err := store.Read(ctx, "radius")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	trace := map[string]any{"loads": loads, "requests": requests, "results": results, "stored": stored, "check": check, "failure": map[string]any{"error": failure.Error(), "code": modelError.Code, "cause": modelError.Cause.Error()}, "retained": retained == original}
	mu.Unlock()
	catalogCompare(t, trace, readRadiusOAuthFixture(t).Integration)
}
