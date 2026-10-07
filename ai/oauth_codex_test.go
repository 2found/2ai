package ai

import (
	"context"
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

type codexOAuthCase struct {
	Input struct {
		Name, Action               string
		Method, Manual, Credential json.RawMessage
		Steps                      []struct {
			Body, Action, StatusText string
			Status                   int
		}
	}
	Log, Output   json.RawMessage
	PromptAborted []*bool
}
type codexOAuthFixture struct {
	Integration json.RawMessage
	Browser     []struct {
		Mode                       string
		Output, Requests           json.RawMessage
		Calls, Status              int
		Duplicate                  *int
		SuccessPage, PromptAborted bool
	}

	UpstreamCommit, AccessToken string
	Metadata                    json.RawMessage
	Cases                       []codexOAuthCase
	Helpers                     []struct{ Credential, Output json.RawMessage }
}

func readCodexOAuthFixture(t *testing.T) codexOAuthFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-codex.json")
	if err != nil {
		t.Fatal(err)
	}
	var f codexOAuthFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 246 || len(f.Helpers) != 6 || len(f.Browser) != 3 || len(f.Integration) == 0 {
		t.Fatal("unexpected Codex OAuth coverage")
	}
	return f
}
func fixedCodexOAuthOptions() OpenAICodexOAuthOptions {
	return OpenAICodexOAuthOptions{PKCE: func() (PKCE, error) { return pkceFromBytes([32]byte{}), nil }, State: func() (string, error) { return strings.Repeat("0", 32), nil }, CallbackHost: func() string { return "127.0.0.1" }, Now: func() float64 { return 1000000 }}
}
func TestPiCodexOAuth(t *testing.T) {
	for i, tc := range readCodexOAuthFixture(t).Cases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runCodexOAuthCase(t, tc, false) })
	}
}
func TestPiCodexOAuthHTTP(t *testing.T) {
	for i, tc := range readCodexOAuthFixture(t).Cases {
		method := catalogDecode(t, tc.Input.Method)
		if method == "browser" || !(tc.Input.Name == "success" || strings.HasPrefix(tc.Input.Name, "poll-status:") || strings.HasPrefix(tc.Input.Name, "refresh-status:") || strings.HasPrefix(tc.Input.Name, "poll-error:") || strings.HasPrefix(tc.Input.Name, "token-raw:")) {
			continue
		}
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runCodexOAuthCase(t, tc, true) })
	}
}
func runCodexOAuthCase(t *testing.T, tc codexOAuthCase, realHTTP bool) {
	t.Helper()
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	failure, reason := errors.New("injected failure"), errors.New("original cause")
	if tc.Input.Action == "preabort" {
		cancel(reason)
	}
	method := catalogDecode(t, tc.Input.Method)
	now, index := float64(1000000), 0
	log := NewArray()
	var mu sync.Mutex
	record := func(value any) { mu.Lock(); log.Append(value); mu.Unlock() }
	options := fixedCodexOAuthOptions()
	options.Now = func() float64 { return now }
	options.Sleep = func(ms float64, c context.Context, message string) error {
		if c.Err() != nil {
			return errors.New(message)
		}
		record(NewObject(Property{Name: "sleep", Value: ms}))
		now += ms
		if tc.Input.Action == "sleep-abort" {
			cancel(reason)
			return errors.New(message)
		}
		return nil
	}
	if tc.Input.Action == "pkce-error" {
		options.PKCE = func() (PKCE, error) { return PKCE{}, errors.New("entropy failed") }
	}
	if tc.Input.Action == "state-error" {
		options.State = func() (string, error) { return "", errors.New("entropy failed") }
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
		t.Cleanup(closeServer)
		server.Close = func() { record(NewObject(Property{Name: "close", Value: true})); closeServer() }
		return server, nil
	}
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		record(NewObject(Property{Name: "request", Value: NewObject(Property{Name: "url", Value: request.URL.String()}, Property{Name: "method", Value: request.Method}, Property{Name: "headers", Value: NewObject(Property{Name: "content-type", Value: request.Header.Get("Content-Type")})}, Property{Name: "body", Value: string(body)}, Property{Name: "aborted", Value: request.Context().Err() != nil})}))
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
		return &http.Response{StatusCode: step.Status, Status: fmt.Sprintf("%d %s", step.Status, step.StatusText), Header: make(http.Header), Body: &oauthTestBody{read: func(p []byte) (int, error) {
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
			r.URL.Host = r.Host
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
			clone.Host = copyURL.Host
			copyURL.Scheme = endpoint.Scheme
			copyURL.Host = endpoint.Host
			clone.URL = &copyURL
			return client.Transport.RoundTrip(clone)
		})}
	} else {
		options.Client = &http.Client{Transport: transport}
	}
	auth := OpenAICodexOAuth(options)
	promptContexts := []context.Context{}
	manualDone := make(chan struct{})
	var manualOnce sync.Once
	interaction := ProviderAuthInteraction{Context: ctx, Prompt: func(c context.Context, prompt *Object) (any, error) {
		if prompt.Get("type") != "select" {
			defer manualOnce.Do(func() { close(manualDone) })
		}
		var aborted any
		if c.Done() != nil {
			aborted = c.Err() != nil
		}
		mu.Lock()
		promptContexts = append(promptContexts, c)
		log.Append(NewObject(Property{Name: "prompt", Value: prompt}, Property{Name: "hasSignal", Value: c.Done() != nil}, Property{Name: "aborted", Value: aborted}))
		mu.Unlock()
		if (prompt.Get("type") == "select" && tc.Input.Action == "select-error") || (prompt.Get("type") != "select" && tc.Input.Action == "prompt-error") {
			return nil, failure
		}
		if prompt.Get("type") == "select" {
			return method, nil
		}
		return catalogDecode(t, tc.Input.Manual), nil
	}, Notify: func(event *Object) {
		record(NewObject(Property{Name: "notify", Value: event}))
		if tc.Input.Action == "notify-error" {
			panic(failure)
		}
		if tc.Input.Action == "notify-abort" {
			cancel(reason)
		}
	}}
	var value any
	var err error
	if method == "refresh" {
		value, err = auth.Refresh(ctx, catalogDecode(t, tc.Input.Credential))
	} else {
		value, err = auth.Login(interaction, nil)
	}
	if len(tc.PromptAborted) > 1 {
		publicationAwait(t, manualDone)
	}
	output := map[string]any{"value": value}
	if err != nil {
		output = map[string]any{"error": err.Error(), "same": err == failure}
	}
	catalogCompare(t, output, tc.Output)
	mu.Lock()
	defer mu.Unlock()
	catalogCompare(t, log, tc.Log)
	aborted := []any{}
	for _, c := range promptContexts {
		var value any
		if c.Done() != nil {
			value = c.Err() != nil
		}
		aborted = append(aborted, value)
	}
	expected, _ := json.Marshal(tc.PromptAborted)
	catalogCompare(t, aborted, expected)
}
func TestPiCodexOAuthMetadataAndToAuth(t *testing.T) {
	f := readCodexOAuthFixture(t)
	auth := OpenAICodexOAuth()
	catalogCompare(t, map[string]any{"name": auth.Name, "isSubscription": *auth.IsSubscription, "loginLabel": nil}, f.Metadata)
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
}

func TestPiCodexOAuthBrowserHTTP(t *testing.T) {
	for _, tc := range readCodexOAuthFixture(t).Browser {
		t.Run(tc.Mode, func(t *testing.T) {
			started, release, manualStarted, promptDone := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
			var startOnce, releaseOnce sync.Once
			var mu sync.Mutex
			requests := NewArray()
			calls := 0
			var manualContext context.Context
			tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				<-manualStarted
				if r.Method != "POST" || r.URL.Path != "/oauth/token" || r.Header.Get("Accept") != "" || r.Header.Get("Content-Type") != "application/x-www-form-urlencoded" {
					t.Error("token request differs")
				}
				raw, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				body := NewObject()
				for _, part := range strings.Split(string(raw), "&") {
					key, value, _ := strings.Cut(part, "=")
					body.Set(urlQueryComponent(key), urlQueryComponent(value))
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
					_, _ = io.WriteString(w, `{"access_token":"`+readCodexOAuthFixture(t).AccessToken+`","refresh_token":"refresh","expires_in":3600}`)
				}
			}))
			defer tokenServer.Close()
			defer releaseOnce.Do(func() { close(release) })
			localURL, err := url.Parse(tokenServer.URL)
			if err != nil {
				t.Fatal(err)
			}
			nativeTransport := tokenServer.Client().Transport
			options := fixedCodexOAuthOptions()
			options.Client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if r.URL.String() != codexOAuthTokenURL {
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
					response, err := browserClient.Get(callbackURL + "?code=browser-code&state=" + strings.Repeat("0", 32))
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
				v, e := OpenAICodexOAuth(options).Login(interaction, nil)
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
				response, err := browserClient.Get(callbackURL + "?code=second&state=" + strings.Repeat("0", 32))
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

func TestPiCodexOAuthModelsConcurrentRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := NewInMemoryCredentialStore()
	_, err := store.Modify(ctx, "codex", func(any) (any, error) {
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
	options := fixedCodexOAuthOptions()
	options.Client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		mu.Lock()
		requests++
		mu.Unlock()
		<-release
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"access_token":"` + readCodexOAuthFixture(t).AccessToken + `","refresh_token":"rotated-refresh","expires_in":3600}`))}, nil
	})}
	auth := LazyOAuth(&LazyOAuthOptions{Name: "Codex", Load: func() (*OAuthAuth, error) {
		mu.Lock()
		loads++
		mu.Unlock()
		return OpenAICodexOAuth(options), nil
	}})
	provider, err := NewModelProvider(&ProviderFactoryOptions{ID: "codex", Models: NewArray(), API: &ProviderStreams{}, Auth: &ProviderAuth{OAuth: auth}})
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
			value, err := models.GetAuth(ctx, "codex", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
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
	stored, err := store.Read(ctx, "codex")
	if err != nil {
		t.Fatal(err)
	}
	check, err := models.CheckAuth(ctx, "codex")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	trace := map[string]any{"loads": loads, "requests": requests, "results": results, "stored": stored, "check": check}
	mu.Unlock()
	catalogCompare(t, trace, readCodexOAuthFixture(t).Integration)
}
