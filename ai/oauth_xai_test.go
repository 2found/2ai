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
)

type xaiOAuthCase struct {
	Input struct {
		Name, Operation, Action string
		Credential              json.RawMessage
		Steps                   []struct {
			Body, Action string
			Status       int
		}
	}
	Log, Output json.RawMessage
}
type xaiOAuthFixture struct {
	Integration    json.RawMessage
	UpstreamCommit string
	Metadata       json.RawMessage
	Cases          []xaiOAuthCase
	Helpers        []struct{ Credential, Output json.RawMessage }
}

func readXaiOAuthFixture(t *testing.T) xaiOAuthFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-xai.json")
	if err != nil {
		t.Fatal(err)
	}
	var f xaiOAuthFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 254 || len(f.Helpers) != 6 {
		t.Fatal("unexpected xAI OAuth coverage")
	}
	return f
}
func TestPiXaiOAuth(t *testing.T) {
	for i, tc := range readXaiOAuthFixture(t).Cases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runXaiOAuthCase(t, tc, false) })
	}
}
func TestPiXaiOAuthHTTP(t *testing.T) {
	for i, tc := range readXaiOAuthFixture(t).Cases {
		if tc.Input.Name != "success" && tc.Input.Name != "pending-slow-down" && tc.Input.Name != "slow-timeout" && !strings.HasPrefix(tc.Input.Name, "status:401") && !strings.Contains(tc.Input.Name, "\uFEFF") {
			continue
		}
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runXaiOAuthCase(t, tc, true) })
	}
}
func runXaiOAuthCase(t *testing.T, tc xaiOAuthCase, realHTTP bool) {
	t.Helper()
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	failure, reason := errors.New("injected failure"), errors.New("original cause")
	if tc.Input.Action == "preabort" {
		cancel(reason)
	}
	now, index := float64(1000000), 0
	log := NewArray()
	options := XaiOAuthOptions{Now: func() float64 { return now }, Sleep: func(ms float64, ctx context.Context, message string) error {
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
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
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
		// The real socket server delegates to the same trace recorder. Its request
		// URL is restored to the provider endpoint before recording fixture bytes.
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r.URL.Scheme = "https"
			r.URL.Host = "auth.x.ai"
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
	auth := XaiOAuth(options)
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
	catalogCompare(t, output, tc.Output)
	catalogCompare(t, log, tc.Log)
}
func TestPiXaiOAuthMetadataAndToAuth(t *testing.T) {
	f := readXaiOAuthFixture(t)
	auth := XaiOAuth()
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

func TestPiXaiOAuthModelsConcurrentRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := NewInMemoryCredentialStore()
	_, err := store.Modify(ctx, "xai", func(any) (any, error) {
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
	options := XaiOAuthOptions{Now: func() float64 { return 1000000 }}
	options.Client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		mu.Lock()
		requests++
		mu.Unlock()
		<-release
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"access_token":"rotated"}`))}, nil
	})}
	auth := LazyOAuth(&LazyOAuthOptions{Name: "xAI", Load: func() (*OAuthAuth, error) {
		mu.Lock()
		loads++
		mu.Unlock()
		return XaiOAuth(options), nil
	}})
	provider, err := NewModelProvider(&ProviderFactoryOptions{ID: "xai", Models: NewArray(), API: &ProviderStreams{}, Auth: &ProviderAuth{OAuth: auth}})
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
			value, err := models.GetAuth(ctx, "xai", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
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
	stored, err := store.Read(ctx, "xai")
	if err != nil {
		t.Fatal(err)
	}
	check, err := models.CheckAuth(ctx, "xai")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	trace := map[string]any{"loads": loads, "requests": requests, "results": results, "stored": stored, "check": check}
	mu.Unlock()
	catalogCompare(t, trace, readXaiOAuthFixture(t).Integration)
}
