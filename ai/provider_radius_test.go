package ai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

func TestPiRadiusProvider(t *testing.T) {
	f := readRadiusCatalogFixture(t)
	for i, tc := range f.ProviderCases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			mode := tc.Input.Mode
			if mode == "preabort" {
				cancel()
			}
			failure := errors.New("injected failure")
			log := NewArray()
			id, name := "custom", "My Radius"
			deps := RadiusProviderDependencies{BaselineModels: catalogDecode(t, f.Baseline).(*Object), Streams: &ProviderStreams{}, Now: func() float64 { return 1000000 }}
			model := radiusFixtureModel(t, f)
			model.Delete("api")
			model.Delete("provider")
			model.Delete("baseUrl")
			deps.Client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				headers := NewObject(Property{Name: "accept", Value: r.Header.Get("Accept")})
				if value, ok := r.Header["Authorization"]; ok {
					headers.Set("authorization", value[0])
				}
				log.Append(NewObject(Property{Name: "request", Value: NewObject(Property{Name: "url", Value: r.URL.String()}, Property{Name: "headers", Value: headers}, Property{Name: "aborted", Value: r.Context().Err() != nil})}))
				if mode == "fetch-error" {
					return nil, failure
				}
				if mode == "fetch-abort" {
					cancel()
				}
				first, second := authSpread(model), authSpread(model)
				first.Set("name", "Online")
				second.Set("id", "online")
				raw, _ := jsonjs.MarshalValue(NewObject(Property{Name: "baseUrl", Value: "https://online.test"}, Property{Name: "models", Value: NewArray(first, second)}))
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(string(raw)))}, nil
			})}
			provider, err := RadiusProvider(RadiusProviderOptions{ID: &id, Name: &name, Gateway: tc.Input.Gateway}, deps)
			if err != nil {
				t.Fatal(err)
			}
			initial, _ := provider.GetModels()
			catalogCompare(t, initial, tc.Initial)
			before, _ := provider.GetModels()
			second, _ := provider.GetModels()
			catalogCompare(t, map[string]any{"freshList": before != second, "sharedRows": before.Len() == 0 || before.Get(0) == second.Get(0)}, tc.Identity)
			credential := NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "stored-key"})
			if strings.Contains(mode, "legacy") {
				credential = NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "oauth-key"}, Property{Name: "gatewayConfig", Value: NewObject(Property{Name: "baseUrl", Value: "https://inference.test/messages"}, Property{Name: "models", Value: NewArray(model)})})
			}
			var stored *Object
			if strings.Contains(mode, "stored") {
				row := authSpread(model)
				row.Set("provider", id)
				row.Set("name", "Stored")
				if mode == "stored-other" {
					row.Set("provider", "other")
				}
				rows := NewArray(row)
				if mode == "stored-empty" {
					rows = NewArray()
				}
				stored = NewObject(Property{Name: "models", Value: rows}, Property{Name: "checkedAt", Value: 1})
			}
			if mode == "duplicates" {
				rows := NewArray()
				for _, name := range []string{"first", "last", "Model A"} {
					row := authSpread(model)
					row.Set("provider", id)
					row.Set("name", name)
					if name == "Model A" {
						row.Set("id", "z")
					}
					rows.Append(row)
				}
				stored = NewObject(Property{Name: "models", Value: rows}, Property{Name: "checkedAt", Value: 1})
			}
			network := strings.Contains(mode, "network") || mode == "fetch-abort" || mode == "fetch-error" || mode == "publish-error"
			err = provider.RefreshModels(ModelRefreshContext{Context: ctx, Credential: credential, Stored: stored, AllowNetwork: network, Publish: func(publication ModelsPublication) (bool, error) {
				before, err := provider.GetModels()
				if err != nil {
					return false, err
				}
				log.Append(NewObject(Property{Name: "publish", Value: publication.Persist}, Property{Name: "before", Value: before}))
				if mode == "publish-error" {
					return false, failure
				}
				if strings.HasPrefix(mode, "reject-") {
					return false, nil
				}
				return true, publication.Update()
			}})
			output := map[string]any{"ok": true}
			if err != nil {
				output = map[string]any{"error": err.Error(), "same": err == failure}
			}
			models, readErr := provider.GetModels()
			if readErr != nil {
				t.Fatal(readErr)
			}
			catalogCompare(t, output, tc.Output)
			catalogCompare(t, log, tc.Log)
			catalogCompare(t, models, tc.Models)
		})
	}
}
func TestPiRadiusProviderWiring(t *testing.T) {
	f := readRadiusCatalogFixture(t)
	id, name, gateway := "identity", "Identity", "custom.test"
	loads := 0
	var loadedOptions any
	model, options := NewObject(Property{Name: "id", Value: "stream"}), NewObject(Property{Name: "hint", Value: true})
	transcript := TranscriptContext{messages: []Message{{Role: "user"}}}
	source := &ProviderEventSource{}
	calls := NewArray()
	stream := func(mode string) ModelStreamFunc {
		return func(_ context.Context, m any, c TranscriptContext, o *Object) (*ProviderEventSource, error) {
			calls.Append(NewObject(Property{Name: "mode", Value: mode}, Property{Name: "sameModel", Value: m == model}, Property{Name: "sameContext", Value: len(c.messages) == 1 && &c.messages[0] == &transcript.messages[0]}, Property{Name: "sameOptions", Value: o == options}))
			return source, nil
		}
	}
	provider, err := RadiusProvider(RadiusProviderOptions{ID: &id, Name: &name, Gateway: &gateway}, RadiusProviderDependencies{Streams: &ProviderStreams{Stream: stream("stream"), StreamSimple: stream("simple")}, LoadOAuth: func(input *RadiusOAuthOptions) (*OAuthAuth, error) {
		loads++
		loadedOptions = map[string]any{"name": input.Name, "gateway": input.Gateway}
		return &OAuthAuth{ToAuth: func(c any) (any, error) {
			return NewObject(Property{Name: "apiKey", Value: catalogProperty(c, "access")}), nil
		}, Refresh: func(_ context.Context, c any) (any, error) {
			copy := authSpread(c)
			copy.Set("access", "rotated")
			return copy, nil
		}}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	authBefore := map[string]any{"name": provider.Auth.OAuth.Name, "loads": loads}
	auth, err := provider.Auth.OAuth.ToAuth(NewObject(Property{Name: "access", Value: "token"}))
	if err != nil {
		t.Fatal(err)
	}
	refreshed, err := provider.Auth.OAuth.Refresh(context.Background(), NewObject(Property{Name: "access", Value: "old"}))
	if err != nil {
		t.Fatal(err)
	}
	first, err := provider.Stream(context.Background(), model, transcript, options)
	if err != nil {
		t.Fatal(err)
	}
	second, err := provider.StreamSimple(context.Background(), model, transcript, options)
	if err != nil {
		t.Fatal(err)
	}
	catalogCompare(t, map[string]any{"authBefore": authBefore, "loads": loads, "loadedOptions": loadedOptions, "auth": auth, "refreshed": refreshed, "streamCalls": calls, "sameStream": first == source && second == source}, f.Wiring)
}
func TestRadiusProviderRequiresMissingDependencies(t *testing.T) {
	if _, err := RadiusProvider(RadiusProviderOptions{}, RadiusProviderDependencies{}); err == nil {
		t.Fatal("absent baseline substituted")
	}
	if _, err := RadiusProvider(RadiusProviderOptions{}, RadiusProviderDependencies{Streams: &ProviderStreams{}}); err == nil {
		t.Fatal("absent baseline substituted")
	}
	gateway := "custom.test"
	if _, err := RadiusProvider(RadiusProviderOptions{Gateway: &gateway}, RadiusProviderDependencies{}); err != nil {
		t.Fatal(err)
	}
}

func TestPiRadiusProviderModelsHTTP(t *testing.T) {
	f := readRadiusCatalogFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store, credentials := NewInMemoryModelsStore(), NewInMemoryCredentialStore()
	_, err := credentials.Modify(ctx, "custom", func(any) (any, error) {
		return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "stored-key"}), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	model := radiusFixtureModel(t, f)
	model.Delete("api")
	model.Delete("provider")
	model.Delete("baseUrl")
	cached := authSpread(model)
	cached.Set("id", "cached")
	cached.Set("provider", "custom")
	cached.Set("api", "pi-messages")
	cached.Set("baseUrl", "cached-url")
	if err = store.Write(ctx, "custom", NewObject(Property{Name: "models", Value: NewArray(cached)}, Property{Name: "checkedAt", Value: 1})); err != nil {
		t.Fatal(err)
	}
	started, release, lateDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	var mu sync.Mutex
	count, networkCalls := 0, 0
	failNetwork := false
	requests := NewArray()
	signals := []context.Context{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		count++
		n := count
		fail := failNetwork
		requests.Append(NewObject(Property{Name: "url", Value: "https://custom.test" + r.URL.RequestURI()}, Property{Name: "authorization", Value: r.Header.Get("Authorization")}))
		mu.Unlock()
		if n == 1 {
			close(started)
			<-release
		}
		if fail {
			w.WriteHeader(503)
			_, _ = io.WriteString(w, " unavailable ")
			return
		}
		row := authSpread(model)
		row.Set("id", "fresh")
		if n == 1 {
			row.Set("id", "stale")
		}
		raw, err := jsonjs.MarshalValue(NewObject(Property{Name: "baseUrl", Value: "https://online.test"}, Property{Name: "models", Value: NewArray(row)}))
		if err != nil {
			t.Error(err)
			return
		}
		_, _ = w.Write(raw)
	}))
	defer func() { releaseOnce.Do(func() { close(release) }); server.Close() }()
	endpoint, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	local := server.Client()
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		mu.Lock()
		signals = append(signals, r.Context())
		mu.Unlock()
		// Deliberately let the obsolete response finish, as in the source trace.
		// Radius must check the original signal before attempting publication.
		clone := r.Clone(context.WithoutCancel(r.Context()))
		copyURL := *r.URL
		copyURL.Scheme, copyURL.Host = endpoint.Scheme, endpoint.Host
		clone.URL = &copyURL
		return local.Transport.RoundTrip(clone)
	})}
	id, gateway := "custom", "custom.test"
	provider, err := RadiusProvider(RadiusProviderOptions{ID: &id, Gateway: &gateway}, RadiusProviderDependencies{Streams: &ProviderStreams{}, Client: client, Now: func() float64 { return 1000000 }})
	if err != nil {
		t.Fatal(err)
	}
	refreshOriginal := provider.RefreshModels
	provider.RefreshModels = func(refresh ModelRefreshContext) error {
		mu.Lock()
		first := false
		if refresh.AllowNetwork {
			networkCalls++
			first = networkCalls == 1
		}
		mu.Unlock()
		if first {
			defer close(lateDone)
		}
		return refreshOriginal(refresh)
	}
	registry := NewModels(ModelsOptions{Persistence: &ModelsPersistence{Read: store.Read, Write: store.Write, Delete: store.Delete}, Credentials: &CredentialPersistence{Read: credentials.Read, Modify: credentials.Modify, List: credentials.List, Delete: credentials.Delete}, AuthContext: &AuthContext{Env: func(context.Context, string) (any, error) { return Undefined, nil }, FileExists: func(context.Context, string) (bool, error) { return false, nil }}})
	registry.SetProvider(provider)
	summarize := func(result ModelsRefreshResult) any {
		failures := NewArray()
		for _, failure := range result.Errors {
			failures.Append(NewObject(Property{Name: "providerId", Value: failure.ProviderID}, Property{Name: "error", Value: failure.Error.Error()}))
		}
		return NewObject(Property{Name: "aborted", Value: result.Aborted}, Property{Name: "errors", Value: failures})
	}
	offline := false
	trace := map[string]any{"offline": summarize(registry.Refresh(ctx, ModelsRefreshOptions{AllowNetwork: &offline})), "restored": registry.GetModels("custom")}
	pending := make(chan ModelsRefreshResult, 1)
	go func() { pending <- registry.Refresh(ctx) }()
	publicationAwait(t, started)
	trace["second"] = summarize(registry.Refresh(ctx))
	trace["first"] = summarize(publicationAwait(t, pending))
	releaseOnce.Do(func() { close(release) })
	publicationAwait(t, lateDone)
	trace["after"] = registry.GetModels("custom")
	trace["stored"], err = store.Read(ctx, "custom")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	failNetwork = true
	mu.Unlock()
	trace["failed"] = summarize(registry.Refresh(ctx))
	trace["afterFailure"] = registry.GetModels("custom")
	mu.Lock()
	trace["requests"] = requests
	aborted := NewArray()
	for _, signal := range signals {
		aborted.Append(signal.Err() != nil)
	}
	trace["signals"] = aborted
	mu.Unlock()
	catalogCompare(t, trace, f.Integration)
}
