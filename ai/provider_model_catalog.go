package ai

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"time"

	"github.com/2found/2ai/jsonjs"
)

// ModelsPublication applies persistence before the provider's in-memory update.
// nil/Undefined leave persistence unchanged; Null deletes the stored entry.
type ModelsPublication struct {
	Persist any
	Update  func() error
}

type ModelRefreshContext struct {
	Context      context.Context
	Credential   any
	Stored       *Object
	Publish      func(ModelsPublication) (bool, error)
	AllowNetwork bool
	Force        *bool
}

// ProviderModelCatalog is the catalog state owned by Pi's createProvider.
// The baseline and model objects retain identity; callers synchronize mutations
// of those objects. Refresh publication and catalog list construction are safe
// to call concurrently. Static catalogs have no RefreshModels callback.
type ProviderModelCatalog struct {
	mu            sync.RWMutex
	baseline      *Array
	dynamic       *Array
	RefreshModels func(ModelRefreshContext) error
}

func NewProviderModelCatalog(providerID string, baseline *Array, fetch func(ModelRefreshContext) (*Array, error), now func() int64) *ProviderModelCatalog {
	return newProviderModelCatalog(func() string { return providerID }, baseline, fetch, now)
}

func newProviderModelCatalog(providerID func() string, baseline *Array, fetch func(ModelRefreshContext) (*Array, error), now func() int64) *ProviderModelCatalog {
	if now == nil {
		now = func() int64 { return time.Now().UnixMilli() }
	}
	catalog := &ProviderModelCatalog{baseline: baseline, dynamic: NewArray()}
	if fetch == nil {
		return catalog
	}
	catalog.RefreshModels = func(refresh ModelRefreshContext) error {
		update := func(models *Array) func() error {
			return func() error {
				catalog.mu.Lock()
				defer catalog.mu.Unlock()
				catalog.dynamic = models
				return nil
			}
		}
		if refresh.Stored != nil {
			models, ok := refresh.Stored.Get("models").(*Array)
			if !ok {
				return errors.New("stored catalog models must be an Array")
			}
			restored := NewArray()
			for _, index := range models.Keys() {
				model := models.Get(index)
				if jsonjs.IsNullish(model) {
					return jsonjs.PropertyReadError(!jsonjs.IsUndefined(model), "model.provider")
				}
				if catalogStrictEqual(catalogProperty(model, "provider"), providerID()) {
					restored.Append(model)
				}
			}
			accepted, err := refresh.Publish(ModelsPublication{Update: update(restored)})
			if err != nil || !accepted {
				return err
			}
		}
		if !refresh.AllowNetwork || refresh.Context.Err() != nil {
			return nil
		}
		fetched, err := fetch(refresh)
		if err != nil {
			return err
		}
		if refresh.Context.Err() != nil {
			return nil
		}
		if fetched == nil {
			return jsonjs.PropertyReadError(true, "fetched.filter")
		}
		refreshed, err := filterKnownCatalogModels(fetched)
		if err != nil {
			return err
		}
		_, err = refresh.Publish(ModelsPublication{
			Persist: NewObject(Property{Name: "models", Value: refreshed}, Property{Name: "checkedAt", Value: now()}),
			Update:  update(refreshed),
		})
		return err
	}
	return catalog
}

func (c *ProviderModelCatalog) GetAllModels() (*Array, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	merged := NewArray()
	// Spread fills baseline holes with explicit undefined; model objects remain
	// shared. Only the first matching baseline slot is replaced by an overlay.
	for index := 0; index < c.baseline.Len(); index++ {
		merged.Append(c.baseline.Get(index))
	}
	for index := 0; index < c.dynamic.Len(); index++ {
		model := c.dynamic.Get(index)
		match := -1
		for slot := 0; slot < merged.Len(); slot++ {
			entry := merged.Get(slot)
			entryType, err := GetModelType(entry)
			if err != nil {
				return nil, err
			}
			modelType, err := GetModelType(model)
			if err != nil {
				return nil, err
			}
			if catalogStrictEqual(entryType, modelType) && catalogStrictEqual(catalogProperty(entry, "id"), catalogProperty(model, "id")) {
				match = slot
				break
			}
		}
		if match < 0 {
			merged.Append(model)
		} else {
			merged.Set(match, model)
		}
	}
	return merged, nil
}

func (c *ProviderModelCatalog) GetModels() (*Array, error) {
	models, err := c.GetAllModels()
	if err != nil {
		return nil, err
	}
	chat := NewArray()
	for index := 0; index < models.Len(); index++ {
		model := models.Get(index)
		matches, err := IsModelType(model, "chat")
		if err != nil {
			return nil, err
		}
		if matches {
			chat.Append(model)
		}
	}
	return chat, nil
}

// GetModelType uses nullish fallback, retaining unknown and malformed explicit
// type values. Unlike catalog flattening, an absent/null type means chat.
func GetModelType(model any) (any, error) {
	if jsonjs.IsNullish(model) {
		return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(model), "model.type")
	}
	kind := catalogProperty(model, "type")
	if jsonjs.IsNullish(kind) {
		return "chat", nil
	}
	return kind, nil
}

func IsModelType(model, kind any) (bool, error) {
	actual, err := GetModelType(model)
	return err == nil && catalogStrictEqual(actual, kind), err
}

func HasAPI(model, api any) (bool, error) {
	chat, err := IsModelType(model, "chat")
	return chat && catalogStrictEqual(catalogProperty(model, "api"), api), err
}

func ModelsAreEqual(a, b any) (bool, error) {
	if !catalogEntryTruthy(a) || !catalogEntryTruthy(b) {
		return false, nil
	}
	aType, err := GetModelType(a)
	if err != nil {
		return false, err
	}
	bType, err := GetModelType(b)
	return err == nil && catalogStrictEqual(aType, bType) && catalogStrictEqual(catalogProperty(a, "id"), catalogProperty(b, "id")) && catalogStrictEqual(catalogProperty(a, "provider"), catalogProperty(b, "provider")), err
}

func catalogStrictEqual(a, b any) bool {
	if jsonjs.IsUndefined(a) || jsonjs.IsUndefined(b) {
		return jsonjs.IsUndefined(a) && jsonjs.IsUndefined(b)
	}
	if a == nil || b == nil || jsonjs.IsNull(a) || jsonjs.IsNull(b) {
		return (a == nil || jsonjs.IsNull(a)) && (b == nil || jsonjs.IsNull(b))
	}
	number := func(value any) (float64, bool) {
		v := reflect.ValueOf(value)
		switch v.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return float64(v.Int()), true
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return float64(v.Uint()), true
		case reflect.Float32, reflect.Float64:
			return v.Float(), true
		}
		return 0, false
	}
	if an, ok := number(a); ok {
		bn, ok := number(b)
		return ok && an == bn
	}
	return reflect.TypeOf(a) == reflect.TypeOf(b) && reflect.TypeOf(a).Comparable() && a == b
}
