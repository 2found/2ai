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
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

type anthropicOAuthFixture struct {
	UpstreamCommit, StackPolicy string
	Metadata                    json.RawMessage
	Integration                 json.RawMessage
	Cases                       []struct {
		Input struct {
			Method, Manual json.RawMessage
			Status         int
			Body, Action   string
		}
		Log, Output    json.RawMessage
		PromptAborted  []*bool
		RequestAborted *bool
	}
	Diagnostics []struct {
		Descriptor json.RawMessage
		Output     string
	}
	Browser []struct {
		Mode                       string
		Output, Requests           json.RawMessage
		Calls, Status              int
		Duplicate                  *int
		SuccessPage, PromptAborted bool
	}
	Helpers []struct{ Credential, Output json.RawMessage }
}

func TestPiAnthropicOAuthModelsConcurrentRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := NewInMemoryCredentialStore()
	_, err := store.Modify(ctx, "anthropic", func(any) (any, error) {
		return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "refresh", Value: "old-refresh"}, Property{Name: "expires", Value: 0}), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	readBoth, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	var mu sync.Mutex
	reads, loads, requests := 0, 0, 0
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
	options := fixedAnthropicOAuthOptions()
	options.Client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		mu.Lock()
		requests++
		mu.Unlock()
		<-release
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"access_token":"rotated","refresh_token":"rotated-refresh","expires_in":3600}`))}, nil
	})}
	auth := LazyOAuth(&LazyOAuthOptions{Name: "Anthropic", Load: func() (*OAuthAuth, error) {
		mu.Lock()
		loads++
		mu.Unlock()
		return AnthropicOAuth(options), nil
	}})
	provider, err := NewModelProvider(&ProviderFactoryOptions{ID: "anthropic", Models: NewArray(), API: &ProviderStreams{}, Auth: &ProviderAuth{OAuth: auth}})
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
			value, err := models.GetAuth(ctx, "anthropic", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
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
	stored, err := store.Read(ctx, "anthropic")
	if err != nil {
		t.Fatal(err)
	}
	check, err := models.CheckAuth(ctx, "anthropic")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	trace := map[string]any{"loads": loads, "requests": requests, "results": results, "stored": stored, "check": check}
	mu.Unlock()
	catalogCompare(t, trace, readAnthropicOAuthFixture(t).Integration)
}

func readAnthropicOAuthFixture(t *testing.T) anthropicOAuthFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-anthropic.json")
	if err != nil {
		t.Fatal(err)
	}
	var f anthropicOAuthFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 112 || len(f.Diagnostics) != 7 || len(f.Browser) != 3 || len(f.Helpers) != 6 || f.StackPolicy == "" {
		t.Fatal("unexpected Anthropic OAuth coverage")
	}
	return f
}
func fixedAnthropicOAuthOptions() AnthropicOAuthOptions {
	host := "127.0.0.1"
	return AnthropicOAuthOptions{CallbackHost: &host, PKCE: func() (PKCE, error) { return pkceFromBytes([32]byte{}), nil }, Now: func() float64 { return 1000000 }, ErrorStack: func(name, message string) string { return "fixed:" + name + ":" + message }}
}
func anthropicInjectedFailure() *OAuthDiagnosticError {
	name := "TransportError"
	return &OAuthDiagnosticError{Name: &name, Message: "injected failure", Code: "EPIPE", Errno: 0, Cause: &OAuthDiagnosticError{Message: "nested", Stack: "nested-stack"}, Stack: "outer-stack"}
}
func TestPiAnthropicOAuthLoginAndRefresh(t *testing.T) {
	for i, tc := range readAnthropicOAuthFixture(t).Cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			method, manual := catalogDecode(t, tc.Input.Method), catalogDecode(t, tc.Input.Manual)
			log := NewArray()
			var mu sync.Mutex
			record := func(value any) { mu.Lock(); log.Append(value); mu.Unlock() }
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			abortReason := &OAuthDiagnosticError{Message: "original cause", Stack: "fixed:Error:original cause"}
			if tc.Input.Action == "preabort" {
				cancel(abortReason)
			}
			if tc.Input.Action == "preabort-default" {
				cancel(nil)
			}
			failure := anthropicInjectedFailure()
			options := fixedAnthropicOAuthOptions()
			if strings.Contains(tc.Input.Action, "timeout") {
				options.RequestTimeout = 10 * time.Millisecond
			}
			if tc.Input.Action == "pkce-error" {
				options.PKCE = func() (PKCE, error) { return PKCE{}, errors.New("entropy failed") }
			}
			options.StartCallback = func(input *OAuthCallbackServerOptions) (*OAuthCallbackServer, error) {
				record(NewObject(Property{Name: "start", Value: NewObject(Property{Name: "providerName", Value: input.ProviderName}, Property{Name: "host", Value: input.Host}, Property{Name: "port", Value: input.Port}, Property{Name: "path", Value: input.Path}, Property{Name: "state", Value: *input.State}, Property{Name: "aborted", Value: input.Context.Err() != nil})}))
				if tc.Input.Action == "callback-error" {
					return nil, errors.New("bind failed")
				}
				copy := *input
				copy.Port = 0
				server, err := StartOAuthCallbackServer(&copy)
				if err != nil {
					return nil, err
				}
				closeServer := server.Close
				server.Close = func() { record(NewObject(Property{Name: "close", Value: true})); closeServer() }
				return server, nil
			}
			var requestContext context.Context
			options.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				requestContext = request.Context()
				body, err := io.ReadAll(request.Body)
				if err != nil {
					return nil, err
				}
				record(NewObject(Property{Name: "request", Value: NewObject(Property{Name: "url", Value: request.URL.String()}, Property{Name: "method", Value: request.Method}, Property{Name: "headers", Value: NewObject(Property{Name: "Content-Type", Value: request.Header.Get("Content-Type")}, Property{Name: "Accept", Value: request.Header.Get("Accept")})}, Property{Name: "body", Value: string(body)}, Property{Name: "aborted", Value: request.Context().Err() != nil})}))
				if request.Context().Err() != nil {
					return nil, context.Cause(request.Context())
				}
				if tc.Input.Action == "fetch-default-abort" {
					cancel(nil)
					return nil, context.Cause(request.Context())
				}
				if tc.Input.Action == "fetch-abort-error" {
					cancel(abortReason)
					return nil, failure
				}
				if tc.Input.Action == "fetch-error" {
					return nil, failure
				}
				if tc.Input.Action == "fetch-timeout" {
					<-request.Context().Done()
					return nil, context.Cause(request.Context())
				}
				if tc.Input.Action == "fetch-abort-success" {
					cancel(abortReason)
				}
				reader := strings.NewReader(tc.Input.Body)
				return &http.Response{StatusCode: tc.Input.Status, Header: make(http.Header), Body: &oauthTestBody{read: func(p []byte) (int, error) {
					if tc.Input.Action == "read-abort-error" {
						cancel(abortReason)
						return 0, failure
					}
					if tc.Input.Action == "read-error" {
						return 0, failure
					}
					if tc.Input.Action == "body-timeout" {
						<-request.Context().Done()
						return 0, context.Cause(request.Context())
					}
					return reader.Read(p)
				}}}, nil
			})}
			promptContexts := []context.Context{}
			manualDone := make(chan struct{})
			var manualOnce sync.Once
			interaction := ProviderAuthInteraction{Context: ctx, Notify: func(event *Object) {
				record(NewObject(Property{Name: "notify", Value: event}))
				if tc.Input.Action == "notify-error" {
					panic(failure)
				}
			}, Prompt: func(c context.Context, prompt *Object) (any, error) {
				if prompt.Get("type") != "select" {
					defer manualOnce.Do(func() { close(manualDone) })
				}
				hasSignal := c.Done() != nil
				var aborted any
				if hasSignal {
					aborted = c.Err() != nil
				}
				mu.Lock()
				promptContexts = append(promptContexts, c)
				log.Append(NewObject(Property{Name: "prompt", Value: prompt}, Property{Name: "hasSignal", Value: hasSignal}, Property{Name: "aborted", Value: aborted}))
				mu.Unlock()
				if (prompt.Get("type") == "select" && tc.Input.Action == "select-error") || (prompt.Get("type") != "select" && tc.Input.Action == "prompt-error") {
					return nil, failure
				}
				if prompt.Get("type") == "select" {
					return method, nil
				}
				return manual, nil
			}}
			var value any
			var err error
			auth := AnthropicOAuth(options)
			if method == "refresh" {
				value, err = auth.Refresh(ctx, NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "refresh", Value: "stored-refresh"}))
			} else {
				value, err = auth.Login(interaction, nil)
			}
			if tc.Input.Action == "post-success-abort" {
				cancel(abortReason)
			}
			if len(tc.PromptAborted) > 1 {
				select {
				case <-manualDone:
				case <-time.After(time.Second):
					t.Fatal("manual prompt not observed")
				}
			}
			output := callbackCapture(value, err)
			if err != nil {
				output["same"] = err == failure
			}
			catalogCompare(t, output, tc.Output)
			mu.Lock()
			catalogCompare(t, log, tc.Log)
			if len(promptContexts) != len(tc.PromptAborted) {
				t.Fatalf("prompts: %d", len(promptContexts))
			}
			for i, c := range promptContexts {
				expected := tc.PromptAborted[i]
				if expected == nil {
					if c.Done() != nil {
						t.Error("unexpected prompt signal")
					}
				} else if (c.Err() != nil) != *expected {
					t.Errorf("prompt abort %d differs", i)
				}
			}
			mu.Unlock()
			if tc.RequestAborted == nil {
				if requestContext != nil {
					t.Fatal("unexpected request")
				}
			} else if requestContext == nil || (requestContext.Err() != nil) != *tc.RequestAborted {
				t.Fatal("request abort differs")
			}
		})
	}
}
func oauthDiagnosticFromFixture(value any) any {
	object, ok := value.(*Object)
	if !ok {
		if value == nil {
			return Null
		}
		return value
	}
	message, hasMessage := object.Lookup("message")
	if !hasMessage {
		return value
	}
	e := &OAuthDiagnosticError{Message: message.(string)}
	if name, exists := object.Lookup("name"); exists {
		s := name.(string)
		e.Name = &s
	}
	if code, exists := object.Lookup("code"); exists {
		e.Code = code
	}
	if errno, exists := object.Lookup("errno"); exists {
		if errno == nil {
			errno = Null
		}
		e.Errno = errno
	}
	if cause, exists := object.Lookup("cause"); exists {
		e.Cause = oauthDiagnosticFromFixture(cause)
	}
	if stack, ok := object.Get("stack").(string); ok {
		e.Stack = stack
	}
	return e
}
func TestPiAnthropicOAuthDiagnosticMetadata(t *testing.T) {
	for _, tc := range readAnthropicOAuthFixture(t).Diagnostics {
		failure := oauthDiagnosticFromFixture(catalogDecode(t, tc.Descriptor)).(error)
		options := fixedAnthropicOAuthOptions()
		options.Client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return nil, failure })}
		_, err := AnthropicOAuth(options).Refresh(context.Background(), NewObject(Property{Name: "refresh", Value: "r"}))
		if err == nil || err.Error() != tc.Output {
			t.Fatalf("Go %v\nPi %s", err, tc.Output)
		}
	}
}
func TestPiAnthropicOAuthMetadataAndToAuth(t *testing.T) {
	fixture := readAnthropicOAuthFixture(t)
	auth := AnthropicOAuth()
	catalogCompare(t, map[string]any{"name": auth.Name, "isSubscription": *auth.IsSubscription, "loginLabel": nil}, fixture.Metadata)
	for _, tc := range fixture.Helpers {
		credential := catalogDecode(t, tc.Credential)
		if object, ok := credential.(*Object); ok && object.Get("absent") == true {
			credential = Undefined
		}
		value, err := auth.ToAuth(credential)
		catalogCompare(t, callbackCapture(value, err), tc.Output)
	}
}

func TestPiAnthropicOAuthBrowserHTTP(t *testing.T) {
	for _, tc := range readAnthropicOAuthFixture(t).Browser {
		t.Run(tc.Mode, func(t *testing.T) {
			started, release, manualStarted, promptDone := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
			var startOnce, releaseOnce sync.Once
			var mu sync.Mutex
			requests := NewArray()
			calls := 0
			var manualContext context.Context
			tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				<-manualStarted
				if r.Method != "POST" || r.URL.Path != "/v1/oauth/token" || r.Header.Get("Accept") != "application/json" || r.Header.Get("Content-Type") != "application/json" {
					t.Error("token request differs")
				}
				raw, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				body, err := jsonjs.DecodeValue(raw)
				if err != nil {
					t.Error(err)
				}
				mu.Lock()
				calls++
				requests.Append(NewObject(Property{Name: "body", Value: body}, Property{Name: "manualAborted", Value: manualContext.Err() != nil}))
				mu.Unlock()
				startOnce.Do(func() { close(started) })
				if tc.Mode == "duplicate" {
					<-release
				}
				if tc.Mode == "failure" {
					w.WriteHeader(401)
					_, _ = io.WriteString(w, "denied")
				} else {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"access_token":"a","refresh_token":"r","expires_in":3600}`)
				}
			}))
			defer tokenServer.Close()
			defer releaseOnce.Do(func() { close(release) })
			localURL, err := url.Parse(tokenServer.URL)
			if err != nil {
				t.Fatal(err)
			}
			nativeTransport := tokenServer.Client().Transport
			options := fixedAnthropicOAuthOptions()
			options.Client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if r.URL.String() != anthropicOAuthTokenURL {
					t.Error("token URL changed")
				}
				copy := r.Clone(r.Context())
				copy.URL.Scheme, copy.URL.Host = localURL.Scheme, localURL.Host
				return nativeTransport.RoundTrip(copy)
			})}
			var callbackURL string
			options.StartCallback = func(input *OAuthCallbackServerOptions) (*OAuthCallbackServer, error) {
				copy := *input
				copy.Port = 0
				server, err := StartOAuthCallbackServer(&copy)
				if err == nil {
					callbackURL = server.RedirectURI
				}
				return server, err
			}
			browserClient := &http.Client{Timeout: 3 * time.Second}
			defer browserClient.CloseIdleConnections()
			pageDone := make(chan struct {
				status int
				body   string
				err    error
			}, 1)
			interaction := ProviderAuthInteraction{Context: context.Background(), Notify: func(event *Object) {
				if event.Get("type") != "auth_url" {
					return
				}
				go func() {
					response, err := browserClient.Get(callbackURL + "?code=browser-code&state=" + pkceFromBytes([32]byte{}).Verifier)
					if err != nil {
						pageDone <- struct {
							status int
							body   string
							err    error
						}{err: err}
						return
					}
					defer response.Body.Close()
					body, err := io.ReadAll(response.Body)
					pageDone <- struct {
						status int
						body   string
						err    error
					}{response.StatusCode, string(body), err}
				}()
			}, Prompt: func(ctx context.Context, prompt *Object) (any, error) {
				if prompt.Get("type") == "select" {
					return "browser", nil
				}
				defer close(promptDone)
				manualContext = ctx
				close(manualStarted)
				<-ctx.Done()
				return nil, errors.New("manual aborted")
			}}
			completed := make(chan struct {
				value any
				err   error
			}, 1)
			go func() {
				v, e := AnthropicOAuth(options).Login(interaction, nil)
				completed <- struct {
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
				response, err := browserClient.Get(callbackURL + "?code=second&state=" + pkceFromBytes([32]byte{}).Verifier)
				if err != nil {
					t.Fatal(err)
				}
				duplicate = response.StatusCode
				response.Body.Close()
				releaseOnce.Do(func() { close(release) })
			}
			result := <-completed
			page := <-pageDone
			if page.err != nil {
				t.Fatal(page.err)
			}
			<-promptDone
			catalogCompare(t, callbackCapture(result.value, result.err), tc.Output)
			mu.Lock()
			catalogCompare(t, requests, tc.Requests)
			if calls != tc.Calls {
				t.Errorf("calls: %d", calls)
			}
			mu.Unlock()
			if page.status != tc.Status || strings.Contains(page.body, "Authentication successful") != tc.SuccessPage || (manualContext.Err() != nil) != tc.PromptAborted {
				t.Fatal("callback page or prompt differs")
			}
			if tc.Duplicate != nil && duplicate != *tc.Duplicate {
				t.Fatal("duplicate callback admitted")
			}
		})
	}
}
