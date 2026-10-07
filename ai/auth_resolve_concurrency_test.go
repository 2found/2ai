package ai

import (
	"context"
	"errors"
	"sync"
	"testing"
)

func TestPiOAuthResolutionConcurrentRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	memory := NewInMemoryCredentialStore()
	_, err := memory.Modify(ctx, "p", func(any) (any, error) {
		return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "expires", Value: 0}), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	readBoth := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	var mu sync.Mutex
	reads, refreshCalls := 0, 0
	derived := []any{}
	var retained context.Context
	persistence := &CredentialPersistence{Read: func(ctx context.Context, id string) (any, error) {
		value, err := memory.Read(ctx, id)
		mu.Lock()
		reads++
		if reads == 2 {
			close(readBoth)
		}
		mu.Unlock()
		return value, err
	}, Modify: memory.Modify}
	models := NewModels(ModelsOptions{Credentials: persistence})
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{OAuth: &OAuthAuth{Refresh: func(refreshCtx context.Context, current any) (any, error) {
		mu.Lock()
		refreshCalls++
		retained = refreshCtx
		if refreshCalls == 1 {
			close(entered)
		}
		mu.Unlock()
		<-release
		return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "rotated"}, Property{Name: "expires", Value: 900000}), nil
	}, ToAuth: func(credential any) (any, error) {
		mu.Lock()
		derived = append(derived, credential)
		mu.Unlock()
		return NewObject(Property{Name: "apiKey", Value: catalogProperty(credential, "access")}), nil
	}}}})
	type completion struct {
		index int
		value any
		err   error
	}
	done := make(chan completion, 2)
	for index := range 2 {
		go func() {
			value, err := models.GetAuth(ctx, "p", AuthResolutionOverrides{Now: func() int64 { return 1000 }})
			done <- completion{index, value, err}
		}()
	}
	publicationAwait(t, entered)
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
	stored, err := memory.Read(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	trace := map[string]any{"results": results, "refreshCalls": refreshCalls, "sameCredential": len(derived) == 2 && derived[0] == derived[1], "stored": stored, "signalAborted": retained.Err() != nil}
	mu.Unlock()
	catalogCompare(t, trace, readAuthResolveFixture(t).Concurrent)
}

func TestPiOAuthResolutionCancellationDoesNotCommit(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	memory := NewInMemoryCredentialStore()
	_, err := memory.Modify(ctx, "p", func(any) (any, error) {
		return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "expires", Value: 0}), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	var mu sync.Mutex
	refreshCalls := 0
	var retained context.Context
	models := NewModels(ModelsOptions{Credentials: &CredentialPersistence{Read: memory.Read, Modify: memory.Modify}})
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{OAuth: &OAuthAuth{Refresh: func(refreshCtx context.Context, current any) (any, error) {
		mu.Lock()
		refreshCalls++
		first := refreshCalls == 1
		if first {
			retained = refreshCtx
			close(entered)
		}
		mu.Unlock()
		if first {
			<-release
		}
		return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "rotated"}, Property{Name: "expires", Value: 900000}), nil
	}, ToAuth: func(credential any) (any, error) {
		return NewObject(Property{Name: "apiKey", Value: catalogProperty(credential, "access")}), nil
	}}}})
	oldCtx, abort := context.WithCancelCause(ctx)
	defer abort(nil)
	oldDone := make(chan any, 1)
	go func() {
		value, err := models.GetAuth(oldCtx, "p", AuthResolutionOverrides{Now: func() int64 { return 1000 }})
		oldDone <- authResolveCapture(value, err)
	}()
	publicationAwait(t, entered)
	abort(errors.New("cancelled by caller"))
	trace := map[string]any{"result": publicationAwait(t, oldDone)}
	trace["before"], err = memory.Read(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	nextDone := make(chan any, 1)
	go func() {
		value, err := models.GetAuth(ctx, "p", AuthResolutionOverrides{Now: func() int64 { return 1000 }})
		if err != nil {
			nextDone <- authResolveCapture(value, err)
		} else {
			nextDone <- value
		}
	}()
	once.Do(func() { close(release) })
	trace["next"] = publicationAwait(t, nextDone)
	trace["stored"], err = memory.Read(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	trace["refreshCalls"], trace["signalAborted"] = refreshCalls, retained.Err() != nil
	mu.Unlock()
	catalogCompare(t, trace, readAuthResolveFixture(t).Cancelled)
}

func TestPiAuthResolutionRetainsInFlightHandler(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	memory := NewInMemoryCredentialStore()
	if _, err := memory.Modify(ctx, "p", func(any) (any, error) {
		return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "expires", Value: 0}), nil
	}); err != nil {
		t.Fatal(err)
	}
	log := NewArray()
	replacement := &OAuthAuth{Refresh: func(_ context.Context, credential any) (any, error) { return credential, nil }, ToAuth: func(any) (any, error) {
		log.Append("new")
		return NewObject(Property{Name: "apiKey", Value: "wrong"}), nil
	}}
	provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{}}
	provider.Auth.OAuth = &OAuthAuth{Refresh: func(context.Context, any) (any, error) {
		provider.ID, provider.Auth.OAuth = "changed", replacement
		return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "rotated"}, Property{Name: "expires", Value: 900000}), nil
	}, ToAuth: func(any) (any, error) {
		log.Append("old")
		return nil, errors.New("derive failed")
	}}
	models := NewModels(ModelsOptions{Credentials: &CredentialPersistence{Read: memory.Read, Modify: memory.Modify}})
	models.SetProvider(provider)
	overrides := AuthResolutionOverrides{Now: func() int64 { return 1000 }}
	trace := map[string]any{"first": authResolveCapture(models.GetAuth(ctx, "p", overrides))}
	trace["second"] = authResolveCapture(models.GetAuth(ctx, "p", overrides))
	trace["log"] = log
	trace["stored"] = authResolveCapture(memory.Read(ctx, "p"))
	trace["changed"] = authResolveCapture(memory.Read(ctx, "changed"))
	catalogCompare(t, trace, readAuthResolveFixture(t).CallbackMutation)
}

func TestAuthResolutionFailureCauseAndPanicBoundary(t *testing.T) {
	for _, stage := range []string{"read", "modify", "api", "refresh", "derive"} {
		for _, panics := range []bool{false, true} {
			name := stage
			if panics {
				name += "/panic"
			}
			t.Run(name, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				cause := errors.New("original failure")
				fail := func() (any, error) {
					if panics {
						panic(cause)
					}
					return nil, cause
				}
				memory := NewInMemoryCredentialStore()
				credential := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "expires", Value: 0})
				if stage == "derive" {
					credential.Set("expires", 900000)
				}
				if stage == "api" {
					credential.Set("type", "api_key")
				}
				if _, err := memory.Modify(ctx, "p", func(any) (any, error) { return credential, nil }); err != nil {
					t.Fatal(err)
				}
				store := CredentialPersistence{Read: memory.Read, Modify: memory.Modify}
				if stage == "read" {
					store.Read = func(context.Context, string) (any, error) { return fail() }
				}
				if stage == "modify" {
					store.Modify = func(context.Context, string, func(any) (any, error)) (any, error) { return fail() }
				}
				provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(APIKeyAuthInput) (any, error) { return fail() }}, OAuth: &OAuthAuth{Refresh: func(context.Context, any) (any, error) { return fail() }, ToAuth: func(any) (any, error) { return fail() }}}}
				_, err := ResolveProviderAuth(ctx, provider, &store, DefaultProviderAuthContext(), AuthResolutionOverrides{Now: func() int64 { return 1000 }})
				var typed *ModelsError
				if !errors.As(err, &typed) || !errors.Is(err, cause) {
					t.Fatalf("lost error boundary/cause: %v", err)
				}
				code := "auth"
				if stage == "refresh" || stage == "derive" {
					code = "oauth"
				}
				if typed.Code != code {
					t.Fatalf("stage %s code %s, want %s", stage, typed.Code, code)
				}
			})
		}
	}
}
