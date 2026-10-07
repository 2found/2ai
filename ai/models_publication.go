package ai

import (
	"context"
	"sync"

	"github.com/2found/2ai/jsonjs"
)

// ModelsPersistence is the concrete callback boundary for provider snapshots.
// Blocking callbacks must honor ctx; a canceled publication returns promptly,
// while its per-provider queue remains held until the callback actually settles.
type ModelsPersistence struct {
	Read   func(context.Context, string) (any, error)
	Write  func(context.Context, string, any) error
	Delete func(context.Context, string) error
}

// ModelsPublisher implements the generation check and publication chain used
// by Pi's Models collection. Provider catalog updates run only after successful
// persistence and a second generation check. Independent providers do not block
// each other. Construct it with the same persistence used for refresh reads.
type ModelsPublisher struct {
	mu          sync.Mutex
	persistence ModelsPersistence
	current     map[string]*ModelsPublicationScope
	queue       providerOperationQueue
}

type ModelsPublicationScope struct {
	Context context.Context
	owner   *ModelsPublisher
	id      string
	cancel  context.CancelFunc
	active  bool
}

func NewModelsPublisher(persistence ModelsPersistence) *ModelsPublisher {
	return &ModelsPublisher{persistence: persistence, current: make(map[string]*ModelsPublicationScope)}
}

// Begin supersedes any earlier refresh for the provider, including cancellation
// of its network/persistence context. Other provider scopes are unaffected.
func (p *ModelsPublisher) Begin(ctx context.Context, providerID string) *ModelsPublicationScope {
	p.mu.Lock()
	defer p.mu.Unlock()
	if old := p.current[providerID]; old != nil && old.active {
		old.cancel()
	}
	signal, cancel := context.WithCancel(ctx)
	scope := &ModelsPublicationScope{Context: signal, owner: p, id: providerID, cancel: cancel, active: true}
	p.current[providerID] = scope
	return scope
}

// Invalidate is the publication boundary for deleting/replacing a provider.
func (p *ModelsPublisher) Invalidate(providerID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if scope := p.current[providerID]; scope != nil && scope.active {
		scope.cancel()
	}
	delete(p.current, providerID)
}

func (p *ModelsPublisher) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for id, scope := range p.current {
		if scope.active {
			scope.cancel()
		}
		delete(p.current, id)
	}
}

// Finish retires only this scope's active controller. Publications retained by
// a provider may still run until its generation is superseded. Finishing an old
// refresh must not retire a replacement refresh's controller.
func (s *ModelsPublicationScope) Finish() {
	s.owner.mu.Lock()
	defer s.owner.mu.Unlock()
	if s.owner.current[s.id] == s {
		s.active = false
	}
}

func (p *ModelsPublisher) clearProviders(ids map[string]bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for id, scope := range p.current {
		if scope.active || ids[id] {
			if scope.active {
				scope.cancel()
			}
			delete(p.current, id)
		}
	}
}

func (s *ModelsPublicationScope) current() bool {
	s.owner.mu.Lock()
	defer s.owner.mu.Unlock()
	return s.Context.Err() == nil && s.owner.current[s.id] == s
}

// Publish queues the entire transaction, including update, before waiting. Even
// when cancellation wins the caller's wait, the queue cannot be released early:
// a store that ignores cancellation could otherwise overwrite a newer snapshot.
func (s *ModelsPublicationScope) Publish(publication ModelsPublication) (bool, error) {
	p := s.owner
	value, err := p.queue.run(s.Context, s.id, func() (any, error) {
		if !s.current() {
			return false, nil
		}
		var err error
		if jsonjs.IsNull(publication.Persist) {
			err = p.persistence.Delete(s.Context, s.id)
		} else if publication.Persist != nil && !jsonjs.IsUndefined(publication.Persist) {
			if err = validateCatalogClone(publication.Persist, make(map[any]bool)); err == nil {
				var cloner jsonjs.ValueCloner
				err = p.persistence.Write(s.Context, s.id, cloner.Clone(publication.Persist))
			}
		}
		if err != nil {
			return false, err
		}
		// No publisher mutex crosses user code: an update may synchronously
		// begin/invalidate a generation. The queue keeps subsequent updates
		// behind it; cancellation may still win the caller's wait.
		accepted := s.current()
		if accepted && publication.Update != nil {
			err = publication.Update()
		}
		return accepted && err == nil, err
	})
	accepted, _ := value.(bool)
	return accepted && err == nil, err
}
