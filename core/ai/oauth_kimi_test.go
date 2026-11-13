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
)

type kimiOAuthCase struct {
	Input struct {
		Name, Operation, Action string
		Env                     map[string]string
		Credential              json.RawMessage
		Steps                   []struct {
			Body, Action string
			Status       int
		}
	}
	Log, Output    json.RawMessage
	RequestAborted []bool
}
type kimiOAuthFixture struct {
	Integration    json.RawMessage
	UpstreamCommit string
	Metadata       json.RawMessage
	Cases          []kimiOAuthCase
	Helpers        []struct{ Credential, Output json.RawMessage }
}

func readKimiOAuthFixture(t *testing.T) kimiOAuthFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-kimi.json")
	if err != nil {
		t.Fatal(err)
	}
	var f kimiOAuthFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 275 || len(f.Helpers) != 9 {
		t.Fatal("unexpected Kimi OAuth coverage")
	}
	return f
}
func TestPiKimiOAuth(t *testing.T) {
	for i, tc := range readKimiOAuthFixture(t).Cases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runKimiOAuthCase(t, tc, false) })
	}
}
func TestPiKimiOAuthHTTP(t *testing.T) {
	for i, tc := range readKimiOAuthFixture(t).Cases {
		if tc.Input.Name != "success" && tc.Input.Name != "pending-slow-down" && tc.Input.Name != "slow-timeout" && !strings.HasPrefix(tc.Input.Name, "status:401") && !strings.Contains(tc.Input.Name, "\uFEFF") && !strings.HasPrefix(tc.Input.Name, "retry-exhausted:") && !strings.HasPrefix(tc.Input.Name, "status:500") {
			continue
		}
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runKimiOAuthCase(t, tc, true) })
	}
}
func runKimiOAuthCase(t *testing.T, tc kimiOAuthCase, realHTTP bool) {
	t.Helper()
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	failure, reason := errors.New("injected failure"), errors.New("original cause")
	if tc.Input.Action == "preabort" {
		cancel(reason)
	}
	now, index := float64(1000000), 0
	log := NewArray()
	options := KimiCodingOAuthOptions{Env: func(name string) string { return tc.Input.Env[name] }, Now: func() float64 { return now }, PollSleep: func(ms float64, ctx context.Context, message string) error {
		if ctx.Err() != nil {
			return errors.New(message)
		}
		log.Append(NewObject(Property{Name: "sleep", Value: ms}))
		now += ms
		if tc.Input.Action == "sleep-abort" {
			cancel(reason)
			return errors.New(message)
		}
		return nil
	}}
	options.RetrySleep = func(ms float64, ctx context.Context) error {
		if ctx.Err() != nil {
			return context.Cause(ctx)
		}
		log.Append(NewObject(Property{Name: "sleep", Value: ms}))
		now += ms
		if tc.Input.Action == "sleep-abort" {
			cancel(reason)
			return reason
		}
		return nil
	}
	for _, step := range tc.Input.Steps {
		if strings.Contains(step.Action, "timeout") {
			options.RequestTimeout = 2 * time.Millisecond
		}
	}
	signals := []context.Context{}
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		signals = append(signals, request.Context())
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		log.Append(NewObject(Property{Name: "request", Value: NewObject(Property{Name: "url", Value: request.URL.String()}, Property{Name: "method", Value: request.Method}, Property{Name: "headers", Value: NewObject(Property{Name: "Accept", Value: request.Header.Get("Accept")}, Property{Name: "Content-Type", Value: request.Header.Get("Content-Type")})}, Property{Name: "body", Value: string(body)}, Property{Name: "aborted", Value: request.Context().Err() != nil})}))
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
		if step.Action == "fetch-timeout" {
			<-request.Context().Done()
			return nil, context.Cause(request.Context())
		}
		reader := strings.NewReader(step.Body)
		return &http.Response{StatusCode: step.Status, Header: make(http.Header), Body: &oauthTestBody{read: func(p []byte) (int, error) {
			if strings.HasPrefix(step.Action, "read-abort") {
				cancel(reason)
			}
			if step.Action == "read-error" || step.Action == "read-abort-error" {
				return 0, failure
			}
			if step.Action == "read-timeout" {
				<-request.Context().Done()
				return 0, context.Cause(request.Context())
			}
			return reader.Read(p)
		}}}, nil
	})
	if realHTTP {
		// The real socket server delegates to the same trace recorder. Its request
		// URL is restored to the provider endpoint before recording fixture bytes.
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r.URL.Scheme = "https"
			r.URL.Host = "auth.kimi.com"
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
			copyURL.Scheme = endpoint.Scheme
			copyURL.Host = endpoint.Host
			clone.URL = &copyURL
			return client.Transport.RoundTrip(clone)
		})}
	} else {
		options.Client = &http.Client{Transport: transport}
	}
	auth := KimiCodingOAuth(options)
	var value any
	var err error
	if tc.Input.Operation == "refresh" {
		value, err = auth.Refresh(ctx, catalogDecode(t, tc.Input.Credential))
	} else {
		value, err = auth.Login(ProviderAuthInteraction{Context: ctx, Prompt: func(context.Context, *Object) (any, error) { return nil, errors.New("unexpected prompt") }, Notify: func(event *Object) {
			log.Append(NewObject(Property{Name: "notify", Value: event}))
			if tc.Input.Action == "notify-error" {
				panic(failure)
			}
			if tc.Input.Action == "notify-abort" {
				cancel(reason)
			}
		}}, nil)
	}
	output := map[string]any{"value": value}
	if err != nil {
		output = map[string]any{"error": err.Error(), "same": err == failure}
	}
	if tc.Input.Action == "post-success-abort" {
		cancel(reason)
	}
	if !realHTTP {
		if options.RequestTimeout != 0 {
			for _, signal := range signals {
				<-signal.Done()
			}
		}
		aborted := make([]bool, len(signals))
		for i, signal := range signals {
			aborted[i] = signal.Err() != nil
		}
		expected, _ := json.Marshal(tc.RequestAborted)
		catalogCompare(t, aborted, expected)
	}
	catalogCompare(t, output, tc.Output)
	catalogCompare(t, log, tc.Log)
}
func TestPiKimiOAuthMetadataAndToAuth(t *testing.T) {
	f := readKimiOAuthFixture(t)
	auth := KimiCodingOAuth()
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

func TestPiKimiOAuthModelsConcurrentRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := NewInMemoryCredentialStore()
	_, err := store.Modify(ctx, "kimi", func(any) (any, error) {
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
	options := KimiCodingOAuthOptions{Env: func(string) string { return "" }, Now: func() float64 { return 1000000 }}
	options.Client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		mu.Lock()
		requests++
		mu.Unlock()
		<-release
		if unauthorized {
			return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{}`))}, nil
		}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"access_token":"rotated","refresh_token":"new-refresh","expires_in":3600}`))}, nil
	})}
	auth := LazyOAuth(&LazyOAuthOptions{Name: "Kimi", Load: func() (*OAuthAuth, error) {
		mu.Lock()
		loads++
		mu.Unlock()
		return KimiCodingOAuth(options), nil
	}})
	provider, err := NewModelProvider(&ProviderFactoryOptions{ID: "kimi", Models: NewArray(), API: &ProviderStreams{}, Auth: &ProviderAuth{OAuth: auth}})
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
			value, err := models.GetAuth(ctx, "kimi", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
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
	stored, err := store.Read(ctx, "kimi")
	if err != nil {
		t.Fatal(err)
	}
	check, err := models.CheckAuth(ctx, "kimi")
	if err != nil {
		t.Fatal(err)
	}
	unauthorized = true
	original := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "refresh", Value: "stored-refresh"}, Property{Name: "expires", Value: 0})
	if _, err = store.Modify(ctx, "kimi", func(any) (any, error) { return original, nil }); err != nil {
		t.Fatal(err)
	}
	_, failure := models.GetAuth(ctx, "kimi", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
	var modelError *ModelsError
	if !errors.As(failure, &modelError) || modelError.Cause == nil {
		t.Fatalf("missing OAuth failure: %v", failure)
	}
	retained, err := store.Read(ctx, "kimi")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	trace := map[string]any{"loads": loads, "requests": requests, "results": results, "stored": stored, "check": check, "failure": map[string]any{"error": failure.Error(), "code": modelError.Code, "cause": modelError.Cause.Error()}, "retained": retained == original}
	mu.Unlock()
	catalogCompare(t, trace, readKimiOAuthFixture(t).Integration)
}
