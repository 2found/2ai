package ai

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type lazyOAuthFixture struct {
	UpstreamCommit string
	Cases          []struct {
		First, Mode            string
		Metadata, Log, Outputs json.RawMessage
		Loads                  int
	}
	Metadata []struct {
		Input struct {
			IsSubscription *bool
			LoginLabel     *string
		}
		Output json.RawMessage
	}
	Concurrent []struct {
		Fail                    bool
		Before, Called, Outputs json.RawMessage
		Loads                   int
	}
	Integration []json.RawMessage
}

func readLazyOAuthFixture(t *testing.T) lazyOAuthFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-lazy-oauth.json")
	if err != nil {
		t.Fatal(err)
	}
	var f lazyOAuthFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 27 || len(f.Metadata) != 9 || len(f.Concurrent) != 2 || len(f.Integration) != 2 {
		t.Fatal("unexpected lazy OAuth coverage")
	}
	return f
}
func TestPiLazyOAuth(t *testing.T) {
	methods := []string{"login", "refresh", "toAuth"}
	for _, tc := range readLazyOAuthFixture(t).Cases {
		t.Run(tc.First+"/"+tc.Mode, func(t *testing.T) {
			log, outputs := NewArray(), NewArray()
			loads := 0
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			if tc.Mode == "preabort" {
				cancel(errors.New("cancelled"))
			}
			interaction := ProviderAuthInteraction{Context: ctx}
			options := &OAuthLoginOptions{GetDeviceID: func() string { return "device" }}
			credential := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "token"})
			loadError, methodError := errors.New("load failed"), errors.New("method failed")
			results := map[string]*Object{}
			callback := func(kind, label string) func(any, any) (any, error) {
				result := NewObject(Property{Name: "kind", Value: kind}, Property{Name: "label", Value: label})
				results[kind] = result
				return func(a, b any) (any, error) {
					argumentSame, secondSame, aborted := a == credential, true, false
					if kind == "login" {
						argumentSame = a.(ProviderAuthInteraction).Context == interaction.Context
						secondSame = b == options
						aborted = ctx.Err() != nil
					}
					if kind == "refresh" {
						secondSame = b == ctx
						aborted = ctx.Err() != nil
					}
					log.Append(NewObject(Property{Name: "kind", Value: kind}, Property{Name: "label", Value: label}, Property{Name: "argumentSame", Value: argumentSame}, Property{Name: "secondSame", Value: secondSame}, Property{Name: "aborted", Value: aborted}))
					if tc.Mode == "method-throw" {
						panic(methodError)
					}
					if tc.Mode == "method-reject" {
						return nil, methodError
					}
					return result, nil
				}
			}
			flow := &OAuthAuth{}
			install := func(label string) {
				login, refresh, toAuth := callback("login", label), callback("refresh", label), callback("toAuth", label)
				flow.Login = func(i ProviderAuthInteraction, o *OAuthLoginOptions) (any, error) { return login(i, o) }
				flow.Refresh = func(c context.Context, value any) (any, error) { return refresh(value, c) }
				flow.ToAuth = func(value any) (any, error) { return toAuth(value, nil) }
			}
			install("original")
			loader := func(label string) func() (*OAuthAuth, error) {
				return func() (*OAuthAuth, error) {
					loads++
					log.Append(NewObject(Property{Name: "load", Value: label}))
					if tc.Mode == "load-reject" {
						return nil, loadError
					}
					if tc.Mode == "load-throw-once" && loads == 1 {
						panic(loadError)
					}
					return flow, nil
				}
			}
			subscription, label := false, "Sign in"
			input := &LazyOAuthOptions{Name: "Visible", IsSubscription: &subscription, LoginLabel: &label, Load: loader("original")}
			auth := LazyOAuth(input)
			input.Name = "Changed"
			subscription = true
			label = "Changed"
			catalogCompare(t, map[string]any{"name": auth.Name, "isSubscription": *auth.IsSubscription, "loginLabel": *auth.LoginLabel}, tc.Metadata)
			if tc.Mode == "replace-before" {
				input.Load = loader("replaced")
			}
			sequence := []string{tc.First}
			for _, kind := range methods {
				if kind != tc.First {
					sequence = append(sequence, kind)
				}
			}
			sequence = append(sequence, tc.First)
			for i, kind := range sequence {
				var value any
				var err error
				switch kind {
				case "login":
					value, err = auth.Login(interaction, options)
				case "refresh":
					value, err = auth.Refresh(ctx, credential)
				case "toAuth":
					value, err = auth.ToAuth(credential)
				}
				if err != nil {
					expected := methodError
					if tc.Mode == "load-reject" || tc.Mode == "load-throw-once" {
						expected = loadError
					}
					outputs.Append(NewObject(Property{Name: "error", Value: err.Error()}, Property{Name: "same", Value: err == expected}))
				} else {
					outputs.Append(NewObject(Property{Name: "value", Value: value}, Property{Name: "same", Value: value == results[kind]}))
				}
				if i == 0 && tc.Mode == "replace-after" {
					input.Load = loader("unused")
				}
				if i == 0 && tc.Mode == "method-replace" {
					install("replaced")
				}
			}
			if loads != tc.Loads {
				t.Fatalf("loads: %d, expected %d", loads, tc.Loads)
			}
			catalogCompare(t, log, tc.Log)
			catalogCompare(t, outputs, tc.Outputs)
		})
	}
}
func TestPiLazyOAuthMetadata(t *testing.T) {
	for _, tc := range readLazyOAuthFixture(t).Metadata {
		loads := 0
		auth := LazyOAuth(&LazyOAuthOptions{Name: "N", IsSubscription: tc.Input.IsSubscription, LoginLabel: tc.Input.LoginLabel, Load: func() (*OAuthAuth, error) { loads++; return nil, errors.New("eager load") }})
		output := NewObject(Property{Name: "name", Value: auth.Name}, Property{Name: "hasSubscription", Value: true}, Property{Name: "hasLabel", Value: true}, Property{Name: "loads", Value: loads})
		if auth.IsSubscription != nil {
			output.Set("isSubscription", *auth.IsSubscription)
		}
		if auth.LoginLabel != nil {
			output.Set("loginLabel", *auth.LoginLabel)
		}
		catalogCompare(t, output, tc.Output)
	}
}
func TestPiLazyOAuthConcurrentLoadIgnoresOperationCancellation(t *testing.T) {
	for _, tc := range readLazyOAuthFixture(t).Concurrent {
		t.Run(map[bool]string{true: "reject", false: "success"}[tc.Fail], func(t *testing.T) {
			var loads, settled atomic.Int32
			started, release := make(chan struct{}), make(chan struct{})
			var releaseOnce sync.Once
			defer releaseOnce.Do(func() { close(release) })
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			failure := errors.New("shared load failed")
			values := map[string]*Object{}
			for _, kind := range []string{"login", "refresh", "toAuth"} {
				values[kind] = NewObject(Property{Name: "kind", Value: kind})
			}
			var mu sync.Mutex
			called := []string{}
			record := func(kind string) (any, error) {
				mu.Lock()
				called = append(called, kind)
				mu.Unlock()
				return values[kind], nil
			}
			flow := &OAuthAuth{Login: func(ProviderAuthInteraction, *OAuthLoginOptions) (any, error) { return record("login") }, Refresh: func(context.Context, any) (any, error) { return record("refresh") }, ToAuth: func(any) (any, error) { return record("toAuth") }}
			auth := LazyOAuth(&LazyOAuthOptions{Name: "N", Load: func() (*OAuthAuth, error) {
				if loads.Add(1) == 1 {
					close(started)
				}
				<-release
				if tc.Fail {
					return nil, failure
				}
				return flow, nil
			}})
			results := make([]chan any, 3)
			entered := make([]chan struct{}, 3)
			for i, kind := range []string{"login", "refresh", "toAuth"} {
				results[i] = make(chan any, 1)
				entered[i] = make(chan struct{})
				go func(i int, kind string) {
					close(entered[i])
					var value any
					var err error
					switch kind {
					case "login":
						value, err = auth.Login(ProviderAuthInteraction{Context: ctx}, nil)
					case "refresh":
						value, err = auth.Refresh(ctx, NewObject())
					case "toAuth":
						value, err = auth.ToAuth(NewObject())
					}
					settled.Add(1)
					if err != nil {
						results[i] <- NewObject(Property{Name: "error", Value: err.Error()}, Property{Name: "same", Value: err == failure})
					} else {
						results[i] <- NewObject(Property{Name: "value", Value: value}, Property{Name: "same", Value: value == values[kind]})
					}
				}(i, kind)
				if i == 0 {
					<-started
				}
			}
			for _, signal := range entered {
				<-signal
			}
			cancel()
			mu.Lock()
			before := map[string]any{"loads": loads.Load(), "settled": settled.Load(), "called": append([]string{}, called...)}
			mu.Unlock()
			catalogCompare(t, before, tc.Before)
			releaseOnce.Do(func() { close(release) })
			outputs := NewArray()
			for _, result := range results {
				select {
				case value := <-result:
					outputs.Append(value)
				case <-time.After(time.Second):
					t.Fatal("load did not settle")
				}
			}
			if int(loads.Load()) != tc.Loads {
				t.Fatalf("loads: %d", loads.Load())
			}
			sort.Strings(called)
			catalogCompare(t, called, tc.Called)
			catalogCompare(t, outputs, tc.Outputs)
		})
	}
}

func TestPiLazyOAuthModelsIntegration(t *testing.T) {
	for _, expected := range readLazyOAuthFixture(t).Integration {
		first := catalogDecode(t, expected).(*Object).Get("first").(string)
		t.Run(first, func(t *testing.T) {
			ctx := context.Background()
			store := NewInMemoryCredentialStore()
			models := NewModels(ModelsOptions{Credentials: &CredentialPersistence{Read: store.Read, List: store.List, Modify: store.Modify, Delete: store.Delete}})
			log := NewArray()
			loads := 0
			issued := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "old"}, Property{Name: "refresh", Value: "refresh"}, Property{Name: "expires", Value: 0})
			renewed := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "new"}, Property{Name: "refresh", Value: "refresh"}, Property{Name: "expires", Value: 999999})
			flow := &OAuthAuth{Name: "Flow", Login: func(ProviderAuthInteraction, *OAuthLoginOptions) (any, error) {
				log.Append("login")
				return issued, nil
			}, Refresh: func(_ context.Context, c any) (any, error) {
				log.Append(NewObject(Property{Name: "refreshSame", Value: c == issued}))
				return renewed, nil
			}, ToAuth: func(c any) (any, error) {
				log.Append(NewObject(Property{Name: "toAuthSame", Value: c == renewed}))
				return NewObject(Property{Name: "apiKey", Value: catalogProperty(c, "access")}), nil
			}}
			auth := LazyOAuth(&LazyOAuthOptions{Name: "Visible", Load: func() (*OAuthAuth, error) { loads++; return flow, nil }})
			provider, err := NewModelProvider(&ProviderFactoryOptions{ID: "p", Auth: &ProviderAuth{OAuth: auth}, Models: NewArray(), API: &ProviderStreams{}})
			if err != nil {
				t.Fatal(err)
			}
			models.SetProvider(provider)
			if _, err = store.Modify(ctx, "p", func(any) (any, error) { return issued, nil }); err != nil {
				t.Fatal(err)
			}
			check, err := models.CheckAuth(ctx, "p")
			if err != nil {
				t.Fatal(err)
			}
			before := map[string]any{"check": check, "loads": loads}
			var loginSame any
			if first == "login" {
				credential, e := models.Login("p", "oauth", ProviderAuthInteraction{Context: ctx})
				if e != nil {
					t.Fatal(e)
				}
				loginSame = credential == issued
			}
			firstAuth, err := models.GetAuth(ctx, "p", AuthResolutionOverrides{Now: func() int64 { return 1000 }})
			if err != nil {
				t.Fatal(err)
			}
			secondAuth, err := models.GetAuth(ctx, "p", AuthResolutionOverrides{Now: func() int64 { return 1000 }})
			if err != nil {
				t.Fatal(err)
			}
			stored, err := store.Read(ctx, "p")
			if err != nil {
				t.Fatal(err)
			}
			storedSame := stored == renewed
			if err = models.Logout(ctx, "p"); err != nil {
				t.Fatal(err)
			}
			afterLogout, err := store.Read(ctx, "p")
			if err != nil {
				t.Fatal(err)
			}
			if afterLogout != Undefined {
				t.Fatal("logout retained credential")
			}
			credential, err := models.Login("p", "oauth", ProviderAuthInteraction{Context: ctx})
			if err != nil {
				t.Fatal(err)
			}
			catalogCompare(t, map[string]any{"first": first, "before": before, "loginSame": loginSame, "firstAuth": firstAuth, "secondAuth": secondAuth, "storedSame": storedSame, "afterLogout": nil, "reloginSame": credential == issued, "loads": loads, "log": log}, expected)
		})
	}
}
