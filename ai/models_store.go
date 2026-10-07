package ai

import (
	"context"
	"math"
	"reflect"
	"sync"

	"github.com/2found/2ai/jsonjs"
)

// InMemoryModelsStore keeps provider-scoped catalog snapshots. Entries use
// Object/Array values so optional fields, unknown metadata, shared references,
// sparse arrays and cycles survive cloning. Read and Write detach the entire
// graph; callers own their inputs and returned values. It is safe for concurrent
// store operations, but callers must not mutate an input during Write.
// This is Pi's catalog store, separate from the workspace's live model list.
type InMemoryModelsStore struct {
	mu      sync.RWMutex
	entries map[string]any
}

func NewInMemoryModelsStore() *InMemoryModelsStore { return &InMemoryModelsStore{} }

// DataCloneError identifies values that cannot be stored by structured clone.
type DataCloneError struct{}

func (*DataCloneError) Error() string { return "The object can not be cloned." }

// Read returns Undefined when the provider has no truthy stored entry.
func (s *InMemoryModelsStore) Read(ctx context.Context, providerID string) (any, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	entry := s.entries[providerID]
	if !catalogEntryTruthy(entry) {
		return Undefined, nil
	}
	var cloner jsonjs.ValueCloner
	return cloner.Clone(entry), nil
}

func (s *InMemoryModelsStore) Write(ctx context.Context, providerID string, entry any) error {
	if err := context.Cause(ctx); err != nil {
		return err
	}
	if err := validateCatalogClone(entry, make(map[any]bool)); err != nil {
		return err
	}
	var cloner jsonjs.ValueCloner
	copied := cloner.Clone(entry)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.entries == nil {
		s.entries = make(map[string]any)
	}
	s.entries[providerID] = copied
	return nil
}

func (s *InMemoryModelsStore) Delete(ctx context.Context, providerID string) error {
	if err := context.Cause(ctx); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.entries, providerID)
	return nil
}

func catalogEntryTruthy(value any) bool {
	if jsonjs.IsNullish(value) {
		return false
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Bool:
		return v.Bool()
	case reflect.String:
		return v.Len() != 0
	case reflect.Float32, reflect.Float64:
		return v.Float() != 0 && !math.IsNaN(v.Float())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() != 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return v.Uint() != 0
	case reflect.Pointer:
		return !v.IsNil()
	}
	return true
}

// structuredClone rejects functions even when they are serializer hooks. Do
// not use JSON export here: it would invoke hooks, discard undefined properties,
// lose repeated references, and fail on otherwise valid cyclic snapshots.
func validateCatalogClone(value any, seen map[any]bool) error {
	if jsonjs.IsNullish(value) {
		return nil
	}
	var children []any
	switch value := value.(type) {
	case *Object:
		if seen[value] {
			return nil
		}
		seen[value] = true
		for _, property := range value.Entries() {
			children = append(children, property.Value)
		}
	case *Array:
		if seen[value] {
			return nil
		}
		seen[value] = true
		for _, key := range value.PropertyKeys() {
			child, _ := value.GetProperty(key)
			children = append(children, child)
		}
	default:
		switch reflect.TypeOf(value).Kind() {
		case reflect.Bool, reflect.String, reflect.Float32, reflect.Float64,
			reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return nil
		default:
			return &DataCloneError{}
		}
	}
	for _, child := range children {
		if err := validateCatalogClone(child, seen); err != nil {
			return err
		}
	}
	return nil
}
