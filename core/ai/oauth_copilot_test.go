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

type copilotOAuthCase struct {
	Input struct {
		Name, Operation, Action string
		Input, Credential       json.RawMessage
		Steps                   []struct {
			Body, Action, StatusText string
			Status                   int
			RetryAfter               *string
		}
	}
	Log, Output json.RawMessage
}
type copilotOAuthFixture struct {
	Integration                   json.RawMessage
	UpstreamCommit, CatalogPolicy string
	KnownModelIds                 []string
	Metadata                      json.RawMessage
	Cases                         []copilotOAuthCase
	Helpers                       []struct{ Credential, Output json.RawMessage }
}

func readCopilotOAuthFixture(t *testing.T) copilotOAuthFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-copilot.json")
	if err != nil {
		t.Fatal(err)
	}
	var f copilotOAuthFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || f.CatalogPolicy == "" || len(f.Cases) != 224 || len(f.Helpers) != 33 {
		t.Fatal("unexpected Copilot OAuth coverage")
	}
	return f
}
func copilotTestCatalog(f copilotOAuthFixture) *Object {
	catalog := NewObject()
	for _, id := range f.KnownModelIds {
		catalog.Set(id, NewObject())
	}
	return catalog
}
func TestPiCopilotOAuth(t *testing.T) {
	f := readCopilotOAuthFixture(t)
	for i, tc := range f.Cases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runCopilotOAuthCase(t, tc, copilotTestCatalog(f), false) })
	}
}
func TestPiCopilotOAuthHTTP(t *testing.T) {
	f := readCopilotOAuthFixture(t)
	for i, tc := range f.Cases {
		if tc.Input.Name != "success" && tc.Input.Name != "retry-exhausted" && tc.Input.Name != "refresh-no-retry" && !strings.HasPrefix(tc.Input.Name, "policy-status:") && !strings.HasPrefix(tc.Input.Name, "retry:") && !strings.HasSuffix(tc.Input.Name, ":bad") && !strings.Contains(tc.Input.Name, "\uFEFF") {
			continue
		}
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runCopilotOAuthCase(t, tc, copilotTestCatalog(f), true) })
	}
}

type copilotTestBody struct {
	read  func([]byte) (int, error)
	close func() error
}

func (b *copilotTestBody) Read(p []byte) (int, error) { return b.read(p) }
func (b *copilotTestBody) Close() error               { return b.close() }
func runCopilotOAuthCase(t *testing.T, tc copilotOAuthCase, known *Object, realHTTP bool) {
	t.Helper()
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	failure, reason := errors.New("injected failure"), errors.New("original cause")
	if tc.Input.Action == "preabort" {
		cancel(reason)
	}
	now, index := float64(1000000), 0
	log := NewArray()
	sleep := func(ms float64, ctx context.Context, message string) error {
		cause := func() error {
			if message != "" {
				return errors.New(message)
			}
			return context.Cause(ctx)
		}
		if ctx.Err() != nil {
			return cause()
		}
		log.Append(NewObject(Property{Name: "sleep", Value: ms}))
		now += ms
		if tc.Input.Action == "retry-budget-timeout" && ms == 0 {
			<-ctx.Done()
			return cause()
		}
		if tc.Input.Action == "sleep-abort" || (tc.Input.Action == "retry-sleep-abort" && ms == 500) {
			cancel(reason)
			return cause()
		}
		return nil
	}
	options := GitHubCopilotOAuthOptions{KnownModels: known, Now: func() float64 { return now }, PollSleep: sleep, Sleep: func(ms float64, ctx context.Context) error { return sleep(ms, ctx, "") }}
	if strings.Contains(tc.Input.Name, "timeout") {
		options.RequestTimeout = 10 * time.Millisecond
	}
	if tc.Input.Action == "retry-budget-timeout" {
		options.RetryBudget = 10 * time.Millisecond
		options.RequestTimeout = time.Second
	}
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		var body []byte
		var err error
		if request.Body != nil {
			body, err = io.ReadAll(request.Body)
			if err != nil {
				return nil, err
			}
		}
		headers := NewObject()
		for _, key := range []string{"accept", "authorization", "content-type", "user-agent", "editor-version", "editor-plugin-version", "copilot-integration-id", "x-github-api-version", "openai-intent", "x-interaction-type"} {
			if value := request.Header.Values(key); len(value) > 0 {
				headers.Set(key, strings.TrimSpace(strings.Join(value, ", ")))
			}
		}
		log.Append(NewObject(Property{Name: "request", Value: NewObject(Property{Name: "url", Value: request.URL.String()}, Property{Name: "method", Value: request.Method}, Property{Name: "headers", Value: headers}, Property{Name: "body", Value: string(body)}, Property{Name: "aborted", Value: request.Context().Err() != nil})}))
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
		readStarted := false
		response := &http.Response{StatusCode: step.Status, Status: fmt.Sprintf("%d %s", step.Status, step.StatusText), Header: make(http.Header), Body: &copilotTestBody{read: func(p []byte) (int, error) {
			readStarted = true
			if step.Action == "read-timeout" {
				<-request.Context().Done()
				return 0, context.Cause(request.Context())
			}
			if strings.HasPrefix(step.Action, "read-abort") {
				cancel(reason)
			}
			if step.Action == "read-error" || step.Action == "read-abort-error" {
				return 0, failure
			}
			return reader.Read(p)
		}, close: func() error {
			if !readStarted && step.Status == 429 {
				log.Append(NewObject(Property{Name: "cancel", Value: true}))
				if step.Action == "cancel-error" {
					return failure
				}
			}
			return nil
		}}}
		if step.RetryAfter != nil {
			response.Header.Set("retry-after", *step.RetryAfter)
		}
		return response, nil
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
			for key, values := range response.Header {
				w.Header()[key] = values
			}
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
	auth, err := GitHubCopilotOAuth(options)
	if err != nil {
		t.Fatal(err)
	}
	var value any
	if tc.Input.Operation == "refresh" {
		value, err = auth.Refresh(ctx, catalogDecode(t, tc.Input.Credential))
	} else {
		value, err = auth.Login(ProviderAuthInteraction{Context: ctx, Prompt: func(c context.Context, prompt *Object) (any, error) {
			log.Append(NewObject(Property{Name: "prompt", Value: prompt}, Property{Name: "hasSignal", Value: c.Done() != nil}))
			if tc.Input.Action == "prompt-error" {
				return nil, failure
			}
			return catalogDecode(t, tc.Input.Input), nil
		}, Notify: func(event *Object) {
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
	expected := tc.Log
	if realHTTP {
		// Socket traces check the HTTP exchanges; direct transport traces also
		// inspect explicit ReadableStream.cancel calls used to discard 429 bodies.
		values := catalogDecode(t, expected).(*Array)
		filtered := NewArray()
		for _, value := range values.Values() {
			if catalogProperty(value, "cancel") != true {
				filtered.Append(value)
			}
		}
		raw, err := filtered.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		expected = raw
	}
	catalogCompare(t, log, expected)
}
func TestPiCopilotOAuthMetadataAndToAuth(t *testing.T) {
	f := readCopilotOAuthFixture(t)
	auth, err := GitHubCopilotOAuth(GitHubCopilotOAuthOptions{KnownModels: copilotTestCatalog(f)})
	if err != nil {
		t.Fatal(err)
	}
	catalogCompare(t, map[string]any{"name": auth.Name, "isSubscription": *auth.IsSubscription, "loginLabel": nil}, f.Metadata)
	for i, tc := range f.Helpers {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			value, err := auth.ToAuth(catalogDecode(t, tc.Credential))
			catalogCompare(t, callbackCapture(value, err), tc.Output)
		})
	}
	if _, err := GitHubCopilotOAuth(GitHubCopilotOAuthOptions{}); err == nil {
		t.Fatal("missing catalog was silently accepted")
	}
}

func TestPiCopilotOAuthModelsConcurrentRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := NewInMemoryCredentialStore()
	_, err := store.Modify(ctx, "copilot", func(any) (any, error) {
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
	options := GitHubCopilotOAuthOptions{KnownModels: copilotTestCatalog(readCopilotOAuthFixture(t)), Now: func() float64 { return 1000000 }}
	options.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		mu.Lock()
		requests++
		mu.Unlock()
		<-release
		if request.URL.Path == "/models" {
			if unauthorized {
				return &http.Response{StatusCode: 401, Status: "401 Unauthorized", Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`denied`))}, nil
			}
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"data":[{"id":"known-a","model_picker_enabled":true,"policy":{"state":"enabled"}}]}`))}, nil
		}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"token":"tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=999","expires_at":5000}`))}, nil
	})}
	auth := LazyOAuth(&LazyOAuthOptions{Name: "Copilot", Load: func() (*OAuthAuth, error) {
		mu.Lock()
		loads++
		mu.Unlock()
		return GitHubCopilotOAuth(options)
	}})
	provider, err := NewModelProvider(&ProviderFactoryOptions{ID: "copilot", Models: NewArray(), API: &ProviderStreams{}, Auth: &ProviderAuth{OAuth: auth}})
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
			value, err := models.GetAuth(ctx, "copilot", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
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
	stored, err := store.Read(ctx, "copilot")
	if err != nil {
		t.Fatal(err)
	}
	check, err := models.CheckAuth(ctx, "copilot")
	if err != nil {
		t.Fatal(err)
	}
	unauthorized = true
	original := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "refresh", Value: "stored-refresh"}, Property{Name: "expires", Value: 0})
	if _, err = store.Modify(ctx, "copilot", func(any) (any, error) { return original, nil }); err != nil {
		t.Fatal(err)
	}
	_, failure := models.GetAuth(ctx, "copilot", AuthResolutionOverrides{Now: func() int64 { return 1000000 }})
	var modelError *ModelsError
	if !errors.As(failure, &modelError) || modelError.Cause == nil {
		t.Fatalf("missing OAuth failure: %v", failure)
	}
	retained, err := store.Read(ctx, "copilot")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	trace := map[string]any{"loads": loads, "requests": requests, "results": results, "stored": stored, "check": check, "failure": map[string]any{"error": failure.Error(), "code": modelError.Code, "cause": modelError.Cause.Error()}, "retained": retained == original}
	mu.Unlock()
	catalogCompare(t, trace, readCopilotOAuthFixture(t).Integration)
}
