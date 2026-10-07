package ai

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

func TestPiInMemoryModelsStore(t *testing.T) {
	ctx := context.Background()
	for i, tc := range readCatalogFixture(t).StoreCases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			store := NewInMemoryModelsStore()
			steps := NewArray()
			read := func(provider string) {
				value, err := store.Read(ctx, provider)
				if err != nil {
					t.Fatal(err)
				}
				if jsonjs.IsUndefined(value) {
					steps.Append(NewObject(Property{Name: "absent", Value: true}))
				} else {
					steps.Append(NewObject(Property{Name: "value", Value: value}))
				}
			}
			read("x")
			if err := store.Write(ctx, "x", catalogDecode(t, tc.Entry)); err != nil {
				t.Fatal(err)
			}
			read("x")
			read("other")
			if err := store.Delete(ctx, "x"); err != nil {
				t.Fatal(err)
			}
			read("x")
			if got, want := catalogStringify(t, steps), catalogStringify(t, catalogDecode(t, tc.Steps)); got != want {
				t.Fatalf("Go %s, Pi %s", got, want)
			}
		})
	}
}

func TestPiModelsStoreSnapshotAndFailure(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryModelsStore()
	child := NewObject(Property{Name: "value", Value: 1})
	array := NewArray(child)
	array.Set(2, Undefined)
	array.Set(3, Null)
	array.SetProperty("extra", child)
	entry := NewObject(Property{Name: "models", Value: array}, Property{Name: "shared", Value: child}, Property{Name: "etag", Value: `"quoted"`}, Property{Name: "checkedAt", Value: Undefined})
	entry.Set("self", entry)
	if err := store.Write(ctx, "p", entry); err != nil {
		t.Fatal(err)
	}
	child.Set("value", 9)
	array.Append("late")
	read := func() *Object {
		value, err := store.Read(ctx, "p")
		if err != nil {
			t.Fatal(err)
		}
		return value.(*Object)
	}
	first := read()
	firstModels := first.Get("models").(*Array)
	firstShared := first.Get("shared").(*Object)
	firstExtra, _ := firstModels.GetProperty("extra")
	_, hasCheckedAt := first.Lookup("checkedAt")
	trace := map[string]any{
		"first": map[string]any{
			"value": firstShared.Get("value"), "length": firstModels.Len(), "keys": firstModels.PropertyKeys(),
			"hasCheckedAt": hasCheckedAt, "hasUndefined": firstModels.Has(2),
			"alias": firstShared == firstModels.Get(0) && firstShared == firstExtra,
			"cycle": first.Get("self") == first,
		},
	}
	firstShared.Set("value", 7)
	again := read()
	trace["next"] = map[string]any{"value": again.Get("shared").(*Object).Get("value"), "detached": again != first && again.Get("shared") != firstShared}
	called := 0
	err := store.Write(ctx, "p", NewObject(Property{Name: "toJSON", Value: JSONMethod(func(any, string) (any, error) {
		called++
		return NewObject(Property{Name: "models", Value: NewArray()}), nil
	})}))
	var cloneError *DataCloneError
	if !errors.As(err, &cloneError) {
		t.Fatalf("expected clone error, got %v", err)
	}
	trace["cloneError"] = map[string]any{"name": "DataCloneError", "message": err.Error(), "called": called}
	trace["afterCloneError"] = read().Get("shared").(*Object).Get("value")
	cancelled, cancel := context.WithCancelCause(ctx)
	cause := errors.New("cancelled by caller")
	cancel(cause)
	abortErrors := []string{}
	for _, operation := range []func() error{
		func() error { _, err := store.Read(cancelled, "p"); return err },
		func() error {
			return store.Write(cancelled, "p", NewObject(Property{Name: "models", Value: NewArray()}))
		},
		func() error { return store.Delete(cancelled, "p") },
	} {
		err := operation()
		if !errors.Is(err, cause) {
			t.Fatalf("cancellation cause changed: %v", err)
		}
		abortErrors = append(abortErrors, err.Error())
	}
	trace["abortErrors"] = abortErrors
	trace["afterAbort"] = read().Get("shared").(*Object).Get("value")
	if err := store.Delete(ctx, "p"); err != nil {
		t.Fatal(err)
	}
	deleted, err := store.Read(ctx, "p")
	if err != nil {
		t.Fatal(err)
	}
	trace["deleted"] = jsonjs.IsUndefined(deleted)
	// Compare parsed graphs: property ordering of the trace is not observable,
	// while the explicitly captured catalog/array key lists are ordered.
	actual := diagnosticValue(t, []byte(catalogStringify(t, trace)))
	expected := diagnosticValue(t, readCatalogFixture(t).StoreTrace)
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("Go trace %s, Pi %s", catalogStringify(t, trace), readCatalogFixture(t).StoreTrace)
	}
}

func TestModelsStoreConcurrentProviderIsolation(t *testing.T) {
	var store InMemoryModelsStore // Zero value is usable.
	var workers sync.WaitGroup
	for worker := range 16 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			provider := fmt.Sprint(worker)
			for version := range 40 {
				entry := NewObject(Property{Name: "models", Value: NewArray(NewObject(Property{Name: "id", Value: provider}))}, Property{Name: "version", Value: version})
				if err := store.Write(context.Background(), provider, entry); err != nil {
					t.Error(err)
					return
				}
				value, err := store.Read(context.Background(), provider)
				if err != nil {
					t.Error(err)
					return
				}
				if value.(*Object).Get("version") != version || value.(*Object).Get("models").(*Array).Get(0).(*Object).Get("id") != provider {
					t.Errorf("provider %s received another snapshot", provider)
					return
				}
				if err := store.Delete(context.Background(), provider); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	workers.Wait()
}
