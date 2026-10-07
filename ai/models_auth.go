package ai

import (
	"context"
	"strings"

	"github.com/2found/2ai/jsonjs"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
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
		providerID, ok = catalogProperty(providerOrModel, "provider").(string)
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
	headers := catalogProperty(providerOrModel, "headers")
	if !catalogEntryTruthy(result) || byID || !catalogEntryTruthy(headers) {
		return result, nil
	}
	merged := authSpread(result)
	resolvedAuth := catalogProperty(result, "auth")
	if jsonjs.IsNullish(resolvedAuth) {
		return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(resolvedAuth), "result.auth.headers")
	}
	auth := authSpread(resolvedAuth)
	auth.Set("headers", mergeAuthHeaders(catalogProperty(auth, "headers"), headers))
	merged.Set("auth", auth)
	return merged, nil
}

func mergeAuthHeaders(base, override any) any {
	if !catalogEntryTruthy(base) && !catalogEntryTruthy(override) {
		return Undefined
	}
	merged := authSpread(base)
	for _, property := range authSpread(override).Entries() {
		lower := authHeaderName(property.Name)
		for _, existing := range merged.Entries() {
			if authHeaderName(existing.Name) == lower {
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

func authHeaderName(name string) string {
	var result, segment strings.Builder
	flush := func() {
		result.WriteString(cases.Lower(language.Und).String(segment.String()))
		segment.Reset()
	}
	for _, point := range jsonjs.StringCodePoints(name) {
		if point >= 0xd800 && point <= 0xdfff {
			flush()
			result.WriteString(catalogCodeUnit(point))
		} else {
			segment.WriteRune(point)
		}
	}
	flush()
	return result.String()
}
