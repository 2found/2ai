package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

type modelsLoginFixture struct {
	UpstreamCommit string
	QueuedStore    json.RawMessage
	Cases          []struct {
		Input, Result, Log, Stored json.RawMessage
		SameStored                 bool
	}
	Races []struct {
		Stage                      string
		Fail, SettledBeforeRelease bool
		Result, Stored             json.RawMessage
	}
}

func readModelsLoginFixture(t *testing.T) modelsLoginFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-models-login.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture modelsLoginFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 86 || len(fixture.Races) != 8 {
		t.Fatal("unexpected login coverage")
	}
	return fixture
}

func TestPiModelsLoginLogout(t *testing.T) {
	for index, tc := range readModelsLoginFixture(t).Cases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			input := catalogDecode(t, tc.Input).(*Object)
			memory := NewInMemoryCredentialStore()
			if _, err := memory.Modify(context.Background(), "p", func(any) (any, error) {
				return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "old"}), nil
			}); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			if input.Get("abort") == true {
				cancel(errors.New("cancelled"))
			}
			log := NewArray()
			failure := input.Get("failure")
			models := NewModels(ModelsOptions{Credentials: &CredentialPersistence{
				Modify: func(ctx context.Context, id string, modify func(any) (any, error)) (any, error) {
					log.Append(NewObject(Property{Name: "modify", Value: id}, Property{Name: "aborted", Value: ctx.Err() != nil}))
					if failure == "modify" {
						return nil, errors.New("modify failed")
					}
					return memory.Modify(ctx, id, modify)
				}, Delete: func(ctx context.Context, id string) error {
					log.Append(NewObject(Property{Name: "delete", Value: id}, Property{Name: "aborted", Value: ctx.Err() != nil}))
					if failure == "delete" {
						return errors.New("delete failed")
					}
					return memory.Delete(ctx, id)
				},
			}})
			options := &OAuthLoginOptions{GetDeviceID: func() string { log.Append("device"); return "device-id" }}
			interaction := ProviderAuthInteraction{Context: ctx, Prompt: func(_ context.Context, prompt *Object) (any, error) {
				log.Append(NewObject(Property{Name: "prompt", Value: prompt}))
				return "reply", nil
			}, Notify: func(event *Object) { log.Append(NewObject(Property{Name: "notify", Value: event})) }}
			var issued any = Undefined
			login := func(i ProviderAuthInteraction, settings *OAuthLoginOptions) (any, error) {
				log.Append(NewObject(Property{Name: "login", Value: true}, Property{Name: "aborted", Value: i.Context.Err() != nil}, Property{Name: "promptIdentity", Value: reflect.ValueOf(i.Prompt).Pointer() == reflect.ValueOf(interaction.Prompt).Pointer()}, Property{Name: "notifyIdentity", Value: reflect.ValueOf(i.Notify).Pointer() == reflect.ValueOf(interaction.Notify).Pointer()}, Property{Name: "optionsIdentity", Value: settings == options}))
				i.Notify(NewObject(Property{Name: "type", Value: "progress"}, Property{Name: "message", Value: "login"}))
				if _, err := i.Prompt(i.Context, NewObject(Property{Name: "type", Value: "text"}, Property{Name: "message", Value: "confirm"})); err != nil {
					return nil, err
				}
				if input.Get("type") == "oauth" {
					settings.GetDeviceID()
				}
				if failure == "login" {
					return nil, errors.New("login failed")
				}
				boxed := input.Get("credential").(*Object)
				if boxed.Get("absent") != true {
					issued = boxed.Get("value")
				}
				return issued, nil
			}
			provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{}, Name: "Provider"}
			switch input.Get("handlers") {
			case "api", "both":
				provider.Auth.APIKey = &APIKeyAuth{Login: func(i ProviderAuthInteraction, settings ...*OAuthLoginOptions) (any, error) {
					return login(i, settings[0])
				}}
			case "ambient":
				provider.Auth.APIKey = &APIKeyAuth{}
			}
			if input.Get("handlers") == "oauth" || input.Get("handlers") == "both" {
				provider.Auth.OAuth = &OAuthAuth{Login: login}
			}
			if input.Get("provider") == true {
				models.SetProvider(provider)
			}
			var result any
			if input.Get("operation") == "logout" {
				result = authResolveCapture(Undefined, models.Logout(ctx, "p"))
			} else {
				result = authResolveCapture(models.Login("p", input.Get("type").(string), interaction, options))
			}
			catalogCompare(t, result, tc.Result)
			catalogCompare(t, log, tc.Log)
			stored, err := memory.Read(context.Background(), "p")
			if err != nil {
				t.Fatal(err)
			}
			catalogCompare(t, authResolveCapture(stored, nil), tc.Stored)
			if same := !jsonjs.IsUndefined(issued) && catalogStrictEqual(stored, issued); same != tc.SameStored {
				t.Fatal("credential identity differs")
			}
		})
	}
}

func TestPiModelsLoginCancellationBoundary(t *testing.T) {
	for _, tc := range readModelsLoginFixture(t).Races {
		t.Run(fmt.Sprintf("%s/fail=%v", tc.Stage, tc.Fail), func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			started, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var once sync.Once
			defer once.Do(func() { close(release) })
			var stored any = NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "old"})
			issued := NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "new"})
			models := NewModels(ModelsOptions{Credentials: &CredentialPersistence{
				Modify: func(_ context.Context, _ string, modify func(any) (any, error)) (any, error) {
					defer close(finished)
					if tc.Stage == "queued" {
						close(started)
						<-release
					}
					next, err := modify(stored)
					if err != nil {
						return nil, err
					}
					if tc.Stage == "active" {
						close(started)
						<-release
					}
					if tc.Fail {
						return nil, errors.New("store failed")
					}
					stored = next
					return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "different-return"}), nil
				}, Delete: func(context.Context, string) error {
					defer close(finished)
					close(started)
					<-release
					if tc.Fail {
						return errors.New("store failed")
					}
					stored = Undefined
					return nil
				},
			}})
			models.SetProvider(&ModelProvider{ID: "p", Name: "Provider", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Login: func(ProviderAuthInteraction, ...*OAuthLoginOptions) (any, error) {
				if tc.Stage == "login" {
					defer close(finished)
					close(started)
					<-release
					if tc.Fail {
						return nil, errors.New("login failed")
					}
				}
				return issued, nil
			}}}})
			pending := make(chan any, 1)
			go func() {
				if tc.Stage == "delete" {
					pending <- authResolveCapture(Undefined, models.Logout(ctx, "p"))
				} else {
					pending <- authResolveCapture(models.Login("p", "api_key", ProviderAuthInteraction{Context: ctx}))
				}
			}()
			publicationAwait(t, started)
			cancel(errors.New("cancelled"))
			var result any
			if tc.SettledBeforeRelease {
				result = publicationAwait(t, pending)
			} else {
				select {
				case early := <-pending:
					t.Fatalf("returned before store settled: %v", early)
				case <-time.After(30 * time.Millisecond):
				}
			}
			once.Do(func() { close(release) })
			if result == nil {
				result = publicationAwait(t, pending)
			}
			publicationAwait(t, finished)
			catalogCompare(t, result, tc.Result)
			catalogCompare(t, authResolveCapture(stored, nil), tc.Stored)
		})
	}
}

func TestPiModelsLoginQueuedStoreAndLogout(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	memory := NewInMemoryCredentialStore()
	if _, err := memory.Modify(context.Background(), "p", func(any) (any, error) {
		return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "old"}), nil
	}); err != nil {
		t.Fatal(err)
	}
	started, release, admitted := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	blocker := make(chan error, 1)
	go func() {
		_, err := memory.Modify(context.Background(), "p", func(current any) (any, error) { close(started); <-release; return current, nil })
		blocker <- err
	}()
	publicationAwait(t, started)
	callbacks := 0
	models := NewModels(ModelsOptions{Credentials: &CredentialPersistence{Modify: func(ctx context.Context, id string, modify func(any) (any, error)) (any, error) {
		close(admitted)
		return memory.Modify(ctx, id, func(current any) (any, error) { callbacks++; return modify(current) })
	}, Delete: memory.Delete}})
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Login: func(ProviderAuthInteraction, ...*OAuthLoginOptions) (any, error) {
		return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "new"}), nil
	}}}})
	login := make(chan any, 1)
	go func() {
		login <- authResolveCapture(models.Login("p", "api_key", ProviderAuthInteraction{Context: ctx}))
	}()
	publicationAwait(t, admitted)
	logout := make(chan any, 1)
	go func() { logout <- authResolveCapture(Undefined, models.Logout(context.Background(), "p")) }()
	cancel(errors.New("cancelled"))
	trace := map[string]any{"login": publicationAwait(t, login)}
	once.Do(func() { close(release) })
	if err := publicationAwait(t, blocker); err != nil {
		t.Fatal(err)
	}
	trace["logout"] = publicationAwait(t, logout)
	// A subsequent queue operation joins any cancelled login left behind logout.
	if _, err := memory.Modify(context.Background(), "p", func(any) (any, error) { return Undefined, nil }); err != nil {
		t.Fatal(err)
	}
	trace["callbacks"] = callbacks
	trace["stored"] = authResolveCapture(memory.Read(context.Background(), "p"))
	catalogCompare(t, trace, readModelsLoginFixture(t).QueuedStore)
}

func TestModelsLoginLogoutRetainsErrorCause(t *testing.T) {
	for _, stage := range []string{"login", "modify", "delete"} {
		t.Run(stage, func(t *testing.T) {
			cause := errors.New("original failure")
			models := NewModels(ModelsOptions{Credentials: &CredentialPersistence{Modify: func(context.Context, string, func(any) (any, error)) (any, error) { return nil, cause }, Delete: func(context.Context, string) error { return cause }}})
			models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Login: func(i ProviderAuthInteraction, _ ...*OAuthLoginOptions) (any, error) {
				if i.Context == nil {
					t.Error("missing normalized context")
				}
				if stage == "login" {
					return nil, cause
				}
				return NewObject(Property{Name: "type", Value: "api_key"}), nil
			}}}})
			var err error
			if stage == "delete" {
				err = models.Logout(context.Background(), "p")
			} else {
				_, err = models.Login("p", "api_key", ProviderAuthInteraction{})
			}
			if !errors.Is(err, cause) {
				t.Fatalf("lost original cause: %v", err)
			}
			if stage == "login" {
				if err != cause {
					t.Fatal("provider login error was wrapped")
				}
			} else {
				var typed *ModelsError
				if !errors.As(err, &typed) || typed.Code != "auth" {
					t.Fatalf("missing storage error category: %v", err)
				}
			}
		})
	}
}
