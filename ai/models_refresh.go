package ai

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/2found/2ai/jsonjs"
)

type ModelsRefreshOptions struct {
	AllowNetwork *bool
	// nil selects all providers; an empty non-nil slice selects none.
	Providers []string
	Force     *bool
	Now       func() int64
}

type ModelsRefreshFailure struct {
	ProviderID string
	Error      error
}

type ModelsRefreshResult struct {
	Aborted bool
	// Errors retain completion order, like Pi's Map, and original error identity.
	Errors []ModelsRefreshFailure
}

// Refresh restores cached catalogs before resolving auth or fetching models.
// Providers run independently. Cancellation stops waiting without releasing an
// unfinished persistence transaction or admitting a superseded publication.
func (m *Models) Refresh(ctx context.Context, options ...ModelsRefreshOptions) ModelsRefreshResult {
	result := ModelsRefreshResult{Errors: make([]ModelsRefreshFailure, 0)}
	if ctx.Err() != nil {
		result.Aborted = true
		return result
	}
	var settings ModelsRefreshOptions
	if len(options) > 0 {
		settings = options[0]
	}
	selected := make(map[string]bool, len(settings.Providers))
	for _, id := range settings.Providers {
		selected[id] = true
	}
	var failuresMu sync.Mutex
	var pending sync.WaitGroup
	refreshable := make([]*ModelProvider, 0)
	for _, provider := range m.GetProviders() {
		if provider.RefreshModels == nil || (settings.Providers != nil && !selected[provider.ID]) {
			continue
		}
		refreshable = append(refreshable, provider)
	}
	for _, provider := range refreshable {
		scope := m.publisher.Begin(ctx, provider.ID)
		pending.Add(1)
		go func(provider *ModelProvider, scope *ModelsPublicationScope) {
			defer pending.Done()
			defer scope.Finish()
			settled := make(chan error, 1)
			go func() {
				_, err := invokeAuth(func() (any, error) {
					credential, credentialError := m.readCredential(scope.Context, provider.ID)
					if err := m.runProviderRefreshPhase(provider, credential, false, nil, scope); err != nil {
						return nil, err
					}
					if credentialError != nil {
						return nil, credentialError
					}
					if (settings.AllowNetwork != nil && !*settings.AllowNetwork) || scope.Context.Err() != nil {
						return nil, nil
					}
					credential, err := m.resolveRefreshCredential(scope.Context, provider, credential, settings.Now)
					if err != nil || !catalogEntryTruthy(credential) {
						return nil, err
					}
					return nil, m.runProviderRefreshPhase(provider, credential, true, settings.Force, scope)
				})
				settled <- err
			}()
			var err error
			select {
			case err = <-settled:
			case <-scope.Context.Done():
			}
			if err != nil && scope.Context.Err() == nil {
				failuresMu.Lock()
				defer failuresMu.Unlock()
				for index := range result.Errors {
					if result.Errors[index].ProviderID == provider.ID {
						result.Errors[index].Error = err
						return
					}
				}
				result.Errors = append(result.Errors, ModelsRefreshFailure{provider.ID, err})
			}
		}(provider, scope)
	}
	finished := make(chan struct{})
	go func() { pending.Wait(); close(finished) }()
	select {
	case <-finished:
	case <-ctx.Done():
	}
	failuresMu.Lock()
	defer failuresMu.Unlock()
	return ModelsRefreshResult{Aborted: ctx.Err() != nil, Errors: append([]ModelsRefreshFailure{}, result.Errors...)}
}

func (m *Models) readCredential(ctx context.Context, providerID string) (any, error) {
	credential, err := invokeAuth(func() (any, error) { return m.credentials.Read(ctx, providerID) })
	if err != nil {
		return Undefined, NewModelsError("auth", "Credential store read failed for "+providerID, err)
	}
	return credential, nil
}

func (m *Models) runProviderRefreshPhase(provider *ModelProvider, credential any, network bool, force *bool, scope *ModelsPublicationScope) error {
	stored, err := m.persistence.Read(scope.Context, provider.ID)
	if err != nil {
		return err
	}
	var snapshot *Object
	if catalogEntryTruthy(stored) {
		if err := validateCatalogClone(stored, make(map[any]bool)); err != nil {
			return err
		}
		var cloner jsonjs.ValueCloner
		entry := cloner.Clone(stored)
		models := catalogProperty(entry, "models")
		list, ok := models.(*Array)
		if !ok {
			if jsonjs.IsNullish(models) {
				return jsonjs.PropertyReadError(!jsonjs.IsUndefined(models), "entry.models.filter")
			}
			return errors.New("entry.models.filter is not a function. (In 'entry.models.filter(hasKnownModelType)', 'entry.models.filter' is undefined)")
		}
		known, err := filterKnownCatalogModels(list)
		if err != nil {
			return err
		}
		snapshot = authSpread(entry)
		snapshot.Set("models", known)
	}
	return provider.RefreshModels(ModelRefreshContext{Context: scope.Context, Credential: credential, Stored: snapshot, Publish: scope.Publish, AllowNetwork: network, Force: force})
}

func (m *Models) resolveRefreshCredential(ctx context.Context, provider *ModelProvider, stored any, now func() int64) (any, error) {
	if now == nil {
		now = func() int64 { return time.Now().UnixMilli() }
	}
	unexpired := func(credential any) (bool, error) {
		expires, err := authNumber(catalogProperty(credential, "expires"))
		return float64(now()) < expires, err
	}
	if catalogStrictEqual(catalogProperty(stored, "type"), "oauth") {
		oauth := provider.authConfig().OAuth
		if oauth == nil {
			return Undefined, nil
		}
		valid, err := unexpired(stored)
		if valid || err != nil {
			return stored, err
		}
		if ctx.Err() != nil {
			return Undefined, nil
		}
		post, err := m.credentials.Modify(ctx, provider.ID, func(current any) (any, error) {
			if !catalogStrictEqual(catalogProperty(current, "type"), "oauth") {
				return Undefined, nil
			}
			valid, err := unexpired(current)
			if valid || err != nil {
				return Undefined, err
			}
			return oauth.Refresh(ctx, current)
		})
		if err != nil || !catalogStrictEqual(catalogProperty(post, "type"), "oauth") {
			return Undefined, err
		}
		return post, nil
	}
	apiKey := provider.authConfig().APIKey
	if apiKey == nil {
		return Undefined, nil
	}
	credential := any(Undefined)
	if catalogStrictEqual(catalogProperty(stored, "type"), "api_key") {
		credential = stored
	}
	result, err := apiKey.Resolve(APIKeyAuthInput{Context: ctx, AuthContext: m.authContext, Credential: credential})
	if err != nil || !catalogEntryTruthy(result) {
		return Undefined, err
	}
	auth := catalogProperty(result, "auth")
	if jsonjs.IsNullish(auth) {
		return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(auth), "result.auth.apiKey")
	}
	return NewObject(Property{Name: "type", Value: "api_key"}, Property{Name: "key", Value: catalogProperty(auth, "apiKey")}, Property{Name: "env", Value: catalogProperty(result, "env")}), nil
}

func filterKnownCatalogModels(models *Array) (*Array, error) {
	known := NewArray()
	for _, index := range models.Keys() {
		model := models.Get(index)
		kind, err := GetModelType(model)
		if err != nil {
			return nil, err
		}
		key, err := catalogKey(kind, make(map[*Array]bool))
		if err != nil {
			return nil, err
		}
		if key == "chat" || key == "image" || key == "classifier" {
			known.Append(model)
		}
	}
	return known, nil
}
