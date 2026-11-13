package ai

import (
	"context"

	"github.com/2found/2ai/core/ai/jscompat"

	"github.com/2found/2ai/core/jsonjs"
)

// GetAuth accepts a provider ID or a model Object. Model headers override auth
// headers case-insensitively without changing the credential or auth result.
// An unknown provider returns Undefined even when the caller is already aborted.
func (m *Models) GetAuth(ctx context.Context, providerOrModel any, options ...AuthResolutionOverrides) (any, error) {
	providerID, byID := providerOrModel.(string)
	if !byID {
		if jsonjs.IsNullish(providerOrModel) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(providerOrModel), "providerOrModel.provider")
		}
		var ok bool
		providerID, ok = jscompat.ReadProperty(providerOrModel, "provider").(string)
		if !ok {
			return Undefined, nil
		}
	}
	provider := m.GetProvider(providerID)
	if provider == nil {
		return Undefined, nil
	}
	var overrides AuthResolutionOverrides
	if len(options) > 0 {
		overrides = options[0]
	}
	return m.authForProvider(ctx, provider, providerOrModel, byID, overrides)
}

func (m *Models) authForProvider(ctx context.Context, provider *ModelProvider, providerOrModel any, byID bool, overrides AuthResolutionOverrides) (any, error) {
	result, err := ResolveProviderAuth(ctx, provider, &m.credentials, m.authContext, overrides)
	if err != nil {
		return nil, err
	}
	headers := jscompat.ReadProperty(providerOrModel, "headers")
	if !jscompat.Truthy(result) || byID || !jscompat.Truthy(headers) {
		return result, nil
	}
	merged := jscompat.Spread(result)
	resolvedAuth := jscompat.ReadProperty(result, "auth")
	if jsonjs.IsNullish(resolvedAuth) {
		return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(resolvedAuth), "result.auth.headers")
	}
	auth := jscompat.Spread(resolvedAuth)
	auth.Set("headers", mergeAuthHeaders(jscompat.ReadProperty(auth, "headers"), headers))
	merged.Set("auth", auth)
	return merged, nil
}

func mergeAuthHeaders(base, override any) any {
	if !jscompat.Truthy(base) && !jscompat.Truthy(override) {
		return Undefined
	}
	merged := jscompat.Spread(base)
	for _, property := range jscompat.Spread(override).Entries() {
		lower := jscompat.LowerHeaderName(property.Name)
		for _, existing := range merged.Entries() {
			if jscompat.LowerHeaderName(existing.Name) == lower {
				merged.Delete(existing.Name)
			}
		}
		// Pi assigns onto a fresh ordinary object after deleting case aliases.
		// Its inherited __proto__ setter does not create a serialized header.
		if property.Name != "__proto__" {
			merged.Set(property.Name, property.Value)
		}
	}
	return merged
}
