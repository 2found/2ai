package ai

import (
	"context"
	"slices"
	"sync"

	"github.com/2found/2ai/jsonjs"
)

// CredentialPersistence is the app-owned callback boundary used by Models.
// Undefined means absent/no replacement; Null is an explicitly stored null.
// Credentials and metadata use Object/Array values to preserve field presence
// and identity. No new provider interface is required.
type CredentialPersistence struct {
	Read   func(context.Context, string) (any, error)
	List   func(context.Context) (*Array, error)
	Modify func(context.Context, string, func(any) (any, error)) (any, error)
	Delete func(context.Context, string) error
}

// InMemoryCredentialStore follows Pi's live-reference credential storage. Read
// and List inspect committed state immediately, without waiting for queued
// writes. Modify/Delete serialize per provider; other providers are independent.
// Returned credentials are not cloned. Callers synchronize mutations of shared
// credential objects, including mutations from a Modify callback.
type InMemoryCredentialStore struct {
	mu          sync.RWMutex
	credentials map[string]any
	order       []string
	queue       providerOperationQueue
}

func NewInMemoryCredentialStore() *InMemoryCredentialStore { return &InMemoryCredentialStore{} }

func (s *InMemoryCredentialStore) Read(ctx context.Context, providerID string) (any, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if value, exists := s.credentials[providerID]; exists {
		return value, nil
	}
	return Undefined, nil
}

func (s *InMemoryCredentialStore) List(ctx context.Context) (*Array, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := NewArray()
	for _, id := range s.order {
		credential := s.credentials[id]
		if jsonjs.IsNullish(credential) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(credential), "credential.type")
		}
		result.Append(NewObject(Property{Name: "providerId", Value: id}, Property{Name: "type", Value: catalogProperty(credential, "type")}))
	}
	return result, nil
}

func (s *InMemoryCredentialStore) Modify(ctx context.Context, providerID string, modify func(any) (any, error)) (any, error) {
	return s.queue.run(ctx, providerID, func() (any, error) {
		current, err := s.Read(ctx, providerID)
		if err != nil {
			return nil, err
		}
		next, err := modify(current)
		if err != nil {
			return nil, err
		}
		if err := context.Cause(ctx); err != nil {
			return nil, err
		}
		if !jsonjs.IsUndefined(next) {
			s.mu.Lock()
			if s.credentials == nil {
				s.credentials = make(map[string]any)
			}
			if _, exists := s.credentials[providerID]; !exists {
				s.order = append(s.order, providerID)
			}
			s.credentials[providerID] = next
			s.mu.Unlock()
		}
		if jsonjs.IsNullish(next) {
			return current, nil
		}
		return next, nil
	})
}

func (s *InMemoryCredentialStore) Delete(ctx context.Context, providerID string) error {
	_, err := s.queue.run(ctx, providerID, func() (any, error) {
		s.mu.Lock()
		defer s.mu.Unlock()
		if _, exists := s.credentials[providerID]; exists {
			delete(s.credentials, providerID)
			index := slices.Index(s.order, providerID)
			s.order = slices.Delete(s.order, index, index+1)
		}
		return Undefined, nil
	})
	return err
}
