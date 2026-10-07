package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

type credentialFixture struct {
	UpstreamCommit    string
	StoreCases        []struct{ Initial, Next, Before, Current, Result, After, Deleted json.RawMessage }
	Live, Order, Race json.RawMessage
	DeleteRaces       []struct {
		Mode                             string
		Before, Refreshed, Result, After json.RawMessage
	}
	AuthCases  []struct{ Credential, Env, Calls, Result json.RawMessage }
	LoginCases []struct {
		Key, Prompts, Result json.RawMessage
		Mode                 string
	}
	EnvFailures []struct {
		Mode          string
		Calls, Result json.RawMessage
	}
}

func readCredentialFixture(t *testing.T) credentialFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-credential-store.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture credentialFixture
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.StoreCases) != 100 || len(fixture.AuthCases) != 60 || len(fixture.LoginCases) != 32 || len(fixture.EnvFailures) != 5 {
		t.Fatal("unexpected credential oracle coverage")
	}
	return fixture
}
func credentialUnbox(t *testing.T, raw json.RawMessage) any {
	t.Helper()
	box := catalogDecode(t, raw).(*Object)
	if box.Get("absent") == true {
		return Undefined
	}
	return box.Get("value")
}
func TestPiInMemoryCredentialStore(t *testing.T) {
	ctx := context.Background()
	for i, tc := range readCredentialFixture(t).StoreCases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			store := NewInMemoryCredentialStore()
			initial, next := credentialUnbox(t, tc.Initial), credentialUnbox(t, tc.Next)
			if !jsonjs.IsUndefined(initial) {
				if _, err := store.Modify(ctx, "p", func(any) (any, error) { return initial, nil }); err != nil {
					t.Fatal(err)
				}
			}
			snapshot := func() any {
				read, readErr := store.Read(ctx, "p")
				list, listErr := store.List(ctx)
				return map[string]any{"read": registryCapture(read, readErr), "list": registryCapture(list, listErr)}
			}
			catalogCompare(t, snapshot(), tc.Before)
			var current any
			result, err := store.Modify(ctx, "p", func(value any) (any, error) { current = registryCapture(value, nil); return next, nil })
			catalogCompare(t, current, tc.Current)
			catalogCompare(t, registryCapture(result, err), tc.Result)
			catalogCompare(t, snapshot(), tc.After)
			if err = store.Delete(ctx, "p"); err != nil {
				t.Fatal(err)
			}
			read, err := store.Read(ctx, "p")
			catalogCompare(t, registryCapture(read, err), tc.Deleted)
		})
	}
}
func TestPiCredentialStoreLiveReferencesAndOrder(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryCredentialStore()
	extra := NewObject(Property{Name: "v", Value: 1})
	credential := NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "fixture"}, Property{Name: "extra", Value: extra})
	written, err := store.Modify(ctx, "p", func(any) (any, error) { return credential, nil })
	if err != nil {
		t.Fatal(err)
	}
	read, err := store.Read(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	trace := map[string]any{"writeIdentity": written == credential, "readIdentity": read == credential}
	read.(*Object).Get("extra").(*Object).Set("v", 2)
	value, _ := store.Read(ctx, "p")
	trace["mutation"] = value.(*Object).Get("extra").(*Object).Get("v")
	unchanged, err := store.Modify(ctx, "p", func(any) (any, error) { return Undefined, nil })
	if err != nil {
		t.Fatal(err)
	}
	trace["noChangeIdentity"] = unchanged == credential
	sentinel := errors.New("modify failed")
	_, err = store.Modify(ctx, "p", func(value any) (any, error) { value.(*Object).Get("extra").(*Object).Set("v", 3); return nil, sentinel })
	if !errors.Is(err, sentinel) {
		t.Fatalf("callback identity: %v", err)
	}
	trace["error"] = err.Error()
	value, _ = store.Read(ctx, "p")
	trace["afterError"] = value.(*Object).Get("extra").(*Object).Get("v")
	metadata, err := store.List(ctx)
	if err != nil {
		t.Fatal(err)
	}
	trace["metadata"] = metadata
	again, _ := store.List(ctx)
	trace["listFresh"] = again != metadata
	result, err := store.Modify(ctx, "p", func(any) (any, error) { return Null, nil })
	if err != nil {
		t.Fatal(err)
	}
	trace["nullResultIsPrevious"] = result == credential
	trace["afterNull"], err = store.Read(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	list, err := store.List(ctx)
	trace["nullList"] = registryCapture(list, err)
	catalogCompare(t, trace, readCredentialFixture(t).Live)
	store = NewInMemoryCredentialStore()
	order := NewArray()
	for _, id := range []string{"b", "a", "b", "__proto__", "0"} {
		_, err = store.Modify(ctx, id, func(any) (any, error) {
			return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: "fixture"}), nil
		})
		if err != nil {
			t.Fatal(err)
		}
		list, err = store.List(ctx)
		if err != nil {
			t.Fatal(err)
		}
		order.Append(list)
	}
	if err = store.Delete(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	_, err = store.Modify(ctx, "a", func(any) (any, error) { return NewObject(Property{Name: "type", Value: "oauth"}), nil })
	if err != nil {
		t.Fatal(err)
	}
	list, _ = store.List(ctx)
	order.Append(list)
	catalogCompare(t, order, readCredentialFixture(t).Order)
}
func TestPiCredentialStoreCanceledModifyKeepsQueue(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryCredentialStore()
	_, err := store.Modify(ctx, "p", func(any) (any, error) {
		return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "version", Value: "initial"}), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	oldCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	log := NewArray()
	var logMu sync.Mutex
	appendLog := func(value any) { logMu.Lock(); defer logMu.Unlock(); log.Append(value) }
	oldDone := make(chan any, 1)
	go func() {
		value, err := store.Modify(oldCtx, "p", func(current any) (any, error) {
			current.(*Object).Set("version", "live-mutation")
			appendLog("old-start")
			close(entered)
			<-release
			appendLog("old-end")
			return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "version", Value: "old"}), nil
		})
		oldDone <- registryCapture(value, err)
	}()
	publicationAwait(t, entered)
	pending, err := store.Read(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	var cloner jsonjs.ValueCloner
	trace := map[string]any{"whilePending": cloner.Clone(pending)}
	store.queue.mu.Lock()
	oldTail := store.queue.tails["p"]
	store.queue.mu.Unlock()
	cancel(errors.New("cancelled by caller"))
	trace["old"] = publicationAwait(t, oldDone)
	nextDone := make(chan any, 1)
	go func() {
		value, err := store.Modify(ctx, "p", func(current any) (any, error) {
			appendLog(NewArray("new-sees", current.(*Object).Get("version")))
			return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "version", Value: "new"}), nil
		})
		if err != nil {
			nextDone <- registryCapture(value, err)
		} else {
			nextDone <- value
		}
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		store.queue.mu.Lock()
		queued := store.queue.tails["p"] != oldTail
		store.queue.mu.Unlock()
		if queued {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("next modification not queued")
		}
		runtime.Gosched()
	}
	if _, err = store.Modify(ctx, "q", func(any) (any, error) {
		appendLog("other")
		return NewObject(Property{Name: "type", Value: "api_key"}), nil
	}); err != nil {
		t.Fatal(err)
	}
	once.Do(func() { close(release) })
	trace["next"] = publicationAwait(t, nextDone)
	trace["final"], err = store.Read(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	trace["log"] = log
	catalogCompare(t, trace, readCredentialFixture(t).Race)
}
func TestCredentialStoreCancellationAndReentrantRead(t *testing.T) {
	var store InMemoryCredentialStore
	ctx := context.Background()
	initial := NewObject(Property{Name: "type", Value: "api_key"})
	if _, err := store.Modify(ctx, "p", func(any) (any, error) { return initial, nil }); err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancelCause(ctx)
	cause := errors.New("cancelled by caller")
	cancel(cause)
	operations := []func() error{func() error { _, err := store.Read(canceled, "p"); return err }, func() error { _, err := store.List(canceled); return err }, func() error { return store.Delete(canceled, "p") }, func() error {
		_, err := store.Modify(canceled, "p", func(any) (any, error) { t.Error("aborted callback executed"); return Undefined, nil })
		return err
	}}
	for _, operation := range operations {
		if err := operation(); !errors.Is(err, cause) {
			t.Fatalf("cancellation identity: %v", err)
		}
	}
	done := make(chan error, 1)
	go func() {
		_, err := store.Modify(ctx, "p", func(current any) (any, error) {
			read, err := store.Read(ctx, "p")
			if err != nil {
				return nil, err
			}
			if read != current {
				return nil, errors.New("read must inspect current state without joining the queue")
			}
			return Undefined, nil
		})
		done <- err
	}()
	if err := publicationAwait(t, done); err != nil {
		t.Fatal(err)
	}
	value, err := store.Read(ctx, "p")
	if err != nil || value != initial {
		t.Fatal("canceled delete changed state")
	}
}

func TestPiCredentialStoreDeleteDuringRefresh(t *testing.T) {
	fixture := readCredentialFixture(t)
	if len(fixture.DeleteRaces) != 2 {
		t.Fatal("missing delete race fixtures")
	}
	for _, tc := range fixture.DeleteRaces {
		t.Run(tc.Mode, func(t *testing.T) {
			ctx := context.Background()
			store := NewInMemoryCredentialStore()
			entry := func(key string) *Object {
				return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: key})
			}
			if _, err := store.Modify(ctx, "p", func(any) (any, error) { return entry("initial"), nil }); err != nil {
				t.Fatal(err)
			}
			entered := make(chan struct{})
			release := make(chan struct{})
			var once sync.Once
			defer once.Do(func() { close(release) })
			refreshed := make(chan any, 1)
			go func() {
				value, err := store.Modify(ctx, "p", func(any) (any, error) { close(entered); <-release; return entry("rotated"), nil })
				refreshed <- registryCapture(value, err)
			}()
			publicationAwait(t, entered)
			store.queue.mu.Lock()
			previous := store.queue.tails["p"]
			store.queue.mu.Unlock()
			deleteCtx, cancel := context.WithCancelCause(ctx)
			defer cancel(nil)
			deleted := make(chan any, 1)
			go func() { err := store.Delete(deleteCtx, "p"); deleted <- registryCapture(Undefined, err) }()
			deadline := time.Now().Add(5 * time.Second)
			for {
				store.queue.mu.Lock()
				queued := store.queue.tails["p"] != previous
				store.queue.mu.Unlock()
				if queued {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("delete not queued")
				}
				runtime.Gosched()
			}
			if tc.Mode == "cancel-delete" {
				cancel(errors.New("cancelled by caller"))
			}
			value, err := store.Read(ctx, "p")
			catalogCompare(t, registryCapture(value, err), tc.Before)
			once.Do(func() { close(release) })
			catalogCompare(t, publicationAwait(t, refreshed), tc.Refreshed)
			catalogCompare(t, publicationAwait(t, deleted), tc.Result)
			if _, err = store.Modify(ctx, "p", func(any) (any, error) { return Undefined, nil }); err != nil {
				t.Fatal(err)
			}
			value, err = store.Read(ctx, "p")
			catalogCompare(t, registryCapture(value, err), tc.After)
		})
	}
}
