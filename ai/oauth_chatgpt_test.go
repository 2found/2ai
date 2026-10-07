package ai

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
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

type chatGPTOAuthCase struct {
	Input struct {
		Name, Operation, Action      string
		DeviceID, Manual, Credential json.RawMessage
		Step                         struct {
			Body, Action, StatusText string
			Status                   int
		}
	}
	Log, Output   json.RawMessage
	PromptAborted *bool
}
type chatGPTOAuthFixture struct {
	Integration json.RawMessage
	Browser     []struct {
		Mode  string
		Pages []struct {
			Path        string `json:"path"`
			Status      int    `json:"status"`
			ContentType string `json:"contentType"`
			Hash        string `json:"hash"`
		}
		Requests, Output json.RawMessage
		PromptAborted    bool
		SpareClosed      *bool
	}

	UpstreamCommit, State, Nonce, DeviceID string
	Metadata                               json.RawMessage
	Cases                                  []chatGPTOAuthCase
	Helpers                                []struct{ Credential, Output json.RawMessage }
}

func readChatGPTOAuthFixture(t *testing.T) chatGPTOAuthFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-chatgpt.json")
	if err != nil {
		t.Fatal(err)
	}
	var f chatGPTOAuthFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 214 || len(f.Helpers) != 6 || len(f.Browser) != 6 || len(f.Integration) == 0 {
		t.Fatal("unexpected ChatGPT OAuth coverage")
	}
	return f
}
func fixedChatGPTOAuthOptions(f chatGPTOAuthFixture) OpenAIChatGPTOAuthOptions {
	host := "127.0.0.1"
	calls := 0
	return OpenAIChatGPTOAuthOptions{CallbackHost: &host, PKCE: func() (PKCE, error) { return pkceFromBytes([32]byte{}), nil }, Now: func() float64 { return 1000000 }, RandomValue: func() (string, error) {
		calls++
		if calls == 1 {
			return f.State, nil
		}
		return f.Nonce, nil
	}}
}
func TestPiChatGPTOAuth(t *testing.T) {
	f := readChatGPTOAuthFixture(t)
	for i, tc := range f.Cases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runChatGPTOAuthCase(t, tc, f, false) })
	}
}
func TestPiChatGPTOAuthHTTP(t *testing.T) {
	f := readChatGPTOAuthFixture(t)
	for i, tc := range f.Cases {
		if tc.Input.Name != "success" && !strings.HasPrefix(tc.Input.Name, "status:401") && !strings.HasPrefix(tc.Input.Name, "raw:") {
			continue
		}
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runChatGPTOAuthCase(t, tc, f, true) })
	}
}
func runChatGPTOAuthCase(t *testing.T, tc chatGPTOAuthCase, f chatGPTOAuthFixture, realHTTP bool) {
	t.Helper()
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	failure, reason := errors.New("injected failure"), errors.New("original cause")
	if tc.Input.Action == "preabort" {
		cancel(reason)
	}
	log := NewArray()
	var mu sync.Mutex
	record := func(value any) { mu.Lock(); log.Append(value); mu.Unlock() }
	options := fixedChatGPTOAuthOptions(f)
	if tc.Input.Action == "pkce-error" {
		options.PKCE = func() (PKCE, error) { return PKCE{}, errors.New("entropy failed") }
	}
	random := options.RandomValue
	calls := 0
	options.RandomValue = func() (string, error) {
		calls++
		if (tc.Input.Action == "state-error" && calls == 1) || (tc.Input.Action == "nonce-error" && calls == 2) {
			return "", errors.New("entropy failed")
		}
		return random()
	}
	options.StartCallback = func(input ChatGPTOAuthCallbackOptions) (*ChatGPTOAuthCallback, error) {
		record(NewObject(Property{Name: "start", Value: NewObject(Property{Name: "host", Value: input.Host}, Property{Name: "port", Value: input.Port})}))
		if input.State != f.State {
			t.Error("callback state differs")
		}
		if tc.Input.Action == "bind-error" || tc.Input.Action == "port-in-use" {
			code := "EACCES"
			if tc.Input.Action == "port-in-use" {
				code = "EADDRINUSE"
			}
			return nil, &OAuthDiagnosticError{Message: "bind failed", Code: code}
		}
		input.Port = 0
		server, err := StartChatGPTOAuthCallback(input)
		if err != nil {
			return nil, err
		}
		closeServer := server.Close
		t.Cleanup(closeServer)
		server.Close = func() {
			record(NewObject(Property{Name: "close", Value: true}))
			record(NewObject(Property{Name: "closeAll", Value: true}))
			closeServer()
		}
		return server, nil
	}
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		record(NewObject(Property{Name: "request", Value: NewObject(Property{Name: "url", Value: request.URL.String()}, Property{Name: "method", Value: request.Method}, Property{Name: "headers", Value: NewObject(Property{Name: "accept", Value: request.Header.Get("Accept")}, Property{Name: "content-type", Value: request.Header.Get("Content-Type")})}, Property{Name: "body", Value: string(body)}, Property{Name: "aborted", Value: request.Context().Err() != nil})}))
		step := tc.Input.Step
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
	var promptContext context.Context
	interaction := ProviderAuthInteraction{Context: ctx, Prompt: func(c context.Context, prompt *Object) (any, error) {
		mu.Lock()
		promptContext = c
		log.Append(NewObject(Property{Name: "prompt", Value: prompt}, Property{Name: "aborted", Value: c.Err() != nil}))
		mu.Unlock()
		if tc.Input.Action == "prompt-error" {
			return nil, failure
		}
		if tc.Input.Action == "prompt-abort" {
			cancel(reason)
			return nil, failure
		}
		return catalogDecode(t, tc.Input.Manual), nil
	}, Notify: func(event *Object) {
		record(NewObject(Property{Name: "notify", Value: event}))
		if tc.Input.Action == "notify-error" || (tc.Input.Action == "progress-error" && event.Get("type") == "progress") {
			panic(failure)
		}
	}}
	auth := OpenAIChatGPTOAuth(options)
	var value any
	var err error
	if tc.Input.Operation == "refresh" {
		value, err = auth.Refresh(ctx, catalogDecode(t, tc.Input.Credential))
	} else {
		var login *OAuthLoginOptions
		if len(tc.Input.DeviceID) > 0 {
			device, _ := catalogDecode(t, tc.Input.DeviceID).(string)
			login = &OAuthLoginOptions{GetDeviceID: func() string {
				if tc.Input.Action == "get-device-error" {
					panic(failure)
				}
				return device
			}}
		}
		value, err = auth.Login(interaction, login)
	}
	output := map[string]any{"value": value}
	if err != nil {
		output = map[string]any{"error": err.Error(), "same": err == failure}
	}
	catalogCompare(t, output, tc.Output)
	mu.Lock()
	defer mu.Unlock()
	catalogCompare(t, log, tc.Log)
	var aborted any
	if promptContext != nil {
		aborted = promptContext.Err() != nil
	}
	expected, _ := json.Marshal(tc.PromptAborted)
	catalogCompare(t, aborted, expected)
}
func TestPiChatGPTOAuthMetadataAndToAuth(t *testing.T) {
	f := readChatGPTOAuthFixture(t)
	auth := OpenAIChatGPTOAuth()
	catalogCompare(t, map[string]any{"name": auth.Name, "isSubscription": *auth.IsSubscription, "loginLabel": *auth.LoginLabel}, f.Metadata)
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

func TestPiChatGPTOAuthBrowserHTTP(t *testing.T) {
	f := readChatGPTOAuthFixture(t)
	for _, tc := range f.Browser {
		t.Run(tc.Mode, func(t *testing.T) {
			manualStarted, manualDone, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var releaseOnce sync.Once
			defer releaseOnce.Do(func() { close(release) })
			requests := NewArray()
			var mu sync.Mutex
			var manualContext context.Context
			tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				<-manualStarted
				raw, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				mu.Lock()
				requests.Append(NewObject(Property{Name: "url", Value: chatGPTOAuthTokenURL}, Property{Name: "body", Value: string(raw)}, Property{Name: "manualAborted", Value: manualContext.Err() != nil}))
				mu.Unlock()
				<-release
				if tc.Mode == "token-failure" {
					w.WriteHeader(401)
					_, _ = io.WriteString(w, "denied")
				} else {
					_, _ = io.WriteString(w, `{"access_token":"access","refresh_token":"refresh","expires_in":3600,"id_token":"id-token","scope":"openid profile chatgpt.tokens.use.direct"}`)
				}
			}))
			defer tokenServer.Close()
			endpoint, err := url.Parse(tokenServer.URL)
			if err != nil {
				t.Fatal(err)
			}
			client := tokenServer.Client()
			options := fixedChatGPTOAuthOptions(f)
			options.Client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if r.URL.String() != chatGPTOAuthTokenURL {
					t.Error("token URL differs")
				}
				clone := r.Clone(r.Context())
				copyURL := *r.URL
				copyURL.Scheme = endpoint.Scheme
				copyURL.Host = endpoint.Host
				clone.URL = &copyURL
				return client.Transport.RoundTrip(clone)
			})}
			var callbackURL string
			options.StartCallback = func(input ChatGPTOAuthCallbackOptions) (*ChatGPTOAuthCallback, error) {
				input.Port = 0
				server, err := StartChatGPTOAuthCallback(input)
				if err == nil {
					callbackURL = server.URL
					t.Cleanup(server.Close)
				}
				return server, err
			}
			done := make(chan map[string]any, 1)
			go func() {
				value, err := OpenAIChatGPTOAuth(options).Login(ProviderAuthInteraction{Context: context.Background(), Notify: func(*Object) {}, Prompt: func(ctx context.Context, _ *Object) (any, error) {
					mu.Lock()
					manualContext = ctx
					mu.Unlock()
					close(manualStarted)
					defer close(manualDone)
					<-ctx.Done()
					return nil, errors.New("manual aborted")
				}}, &OAuthLoginOptions{GetDeviceID: func() string { return f.DeviceID }})
				done <- callbackCapture(value, err)
			}()
			publicationAwait(t, manualStarted)
			callback, err := url.Parse(callbackURL)
			if err != nil {
				t.Fatal(err)
			}
			var spare net.Conn
			if tc.Mode == "idle-http-connection" {
				spare, err = net.DialTimeout("tcp", callback.Host, time.Second)
				if err != nil {
					t.Fatal(err)
				}
				defer spare.Close()
				// Complete an HTTP request before testing idle connection cleanup.
				// Bun can leave pre-HTTP TCP sockets open; Go's Server.Close also
				// closes those sockets, so their lifecycle is not parity evidence.
				if err := spare.SetDeadline(time.Now().Add(2 * time.Second)); err != nil {
					t.Fatal(err)
				}
				if _, err := io.WriteString(spare, "GET /wrong HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n"); err != nil {
					t.Fatal(err)
				}
				response, err := http.ReadResponse(bufio.NewReader(spare), nil)
				if err != nil {
					t.Fatal(err)
				}
				_, err = io.Copy(io.Discard, response.Body)
				response.Body.Close()
				if err != nil || response.StatusCode != http.StatusNotFound {
					t.Fatalf("idle HTTP setup: status %d, error %v", response.StatusCode, err)
				}
			}
			pages := NewArray()
			browserClient := &http.Client{Timeout: 2 * time.Second}
			defer browserClient.CloseIdleConnections()
			for _, expected := range tc.Pages {
				response, err := browserClient.Get(callback.Scheme + "://" + callback.Host + expected.Path)
				if err != nil {
					t.Fatal(err)
				}
				body, err := io.ReadAll(response.Body)
				response.Body.Close()
				if err != nil {
					t.Fatal(err)
				}
				pages.Append(NewObject(Property{Name: "path", Value: expected.Path}, Property{Name: "status", Value: response.StatusCode}, Property{Name: "contentType", Value: response.Header.Get("Content-Type")}, Property{Name: "hash", Value: fmt.Sprintf("%x", sha256.Sum256(body))}))
			}
			releaseOnce.Do(func() { close(release) })
			output := publicationAwait(t, done)
			publicationAwait(t, manualDone)
			catalogCompare(t, output, tc.Output)
			expectedPages, _ := json.Marshal(tc.Pages)
			catalogCompare(t, pages, expectedPages)
			mu.Lock()
			catalogCompare(t, requests, tc.Requests)
			aborted := manualContext.Err() != nil
			mu.Unlock()
			if aborted != tc.PromptAborted {
				t.Fatal("manual cancellation differs")
			}
			if spare != nil {
				if err := spare.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
					t.Fatal(err)
				}
				var b [1]byte
				_, err := spare.Read(b[:])
				closed := err != nil
				if timeout, ok := err.(net.Error); ok && timeout.Timeout() {
					closed = false
				}
				if tc.SpareClosed == nil || closed != *tc.SpareClosed {
					t.Fatalf("spare connection closure differs: %v", err)
				}
			}
		})
	}
}

func TestPiChatGPTOAuthModelsConcurrentRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := NewInMemoryCredentialStore()
	_, err := store.Modify(ctx, "chatgpt", func(any) (any, error) {
		return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "refresh", Value: "stored-refresh"}, Property{Name: "clientId", Value: "issued-client"}, Property{Name: "expires", Value: 0}), nil
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
	options := fixedChatGPTOAuthOptions(readChatGPTOAuthFixture(t))
	options.Client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		mu.Lock()
		requests++
		mu.Unlock()
		<-release
		if unauthorized {
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"access_token":"access","refresh_token":"refresh","expires_in":3600,"scope":"openid"}`))}, nil
		}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"access_token":"access","refresh_token":"refresh","expires_in":3600,"scope":"openid profile chatgpt.tokens.use.direct"}`))}, nil
	})}
	auth := LazyOAuth(&LazyOAuthOptions{Name: "ChatGPT", Load: func() (*OAuthAuth, error) {
		mu.Lock()
		loads++
		mu.Unlock()
		return OpenAIChatGPTOAuth(options), nil
	}})
	provider, err := NewModelProvider(&ProviderFactoryOptions{ID: "chatgpt", Models: NewArray(), API: &ProviderStreams{}, Auth: &ProviderAuth{OAuth: auth}})
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
			value, err := models.GetAuth(ctx, "chatgpt", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
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
	stored, err := store.Read(ctx, "chatgpt")
	if err != nil {
		t.Fatal(err)
	}
	check, err := models.CheckAuth(ctx, "chatgpt")
	if err != nil {
		t.Fatal(err)
	}
	unauthorized = true
	original := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "refresh", Value: "stored-refresh"}, Property{Name: "clientId", Value: "issued-client"}, Property{Name: "expires", Value: 0})
	if _, err = store.Modify(ctx, "chatgpt", func(any) (any, error) { return original, nil }); err != nil {
		t.Fatal(err)
	}
	_, failure := models.GetAuth(ctx, "chatgpt", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
	var modelError *ModelsError
	if !errors.As(failure, &modelError) || modelError.Cause == nil {
		t.Fatalf("missing OAuth failure: %v", failure)
	}
	retained, err := store.Read(ctx, "chatgpt")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	trace := map[string]any{"loads": loads, "requests": requests, "results": results, "stored": stored, "check": check, "failure": map[string]any{"error": failure.Error(), "code": modelError.Code, "cause": modelError.Cause.Error()}, "retained": retained == original}
	mu.Unlock()
	catalogCompare(t, trace, readChatGPTOAuthFixture(t).Integration)
}
