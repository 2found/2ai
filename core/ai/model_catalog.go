package ai

import (
	"github.com/2found/2ai/core/ai/jscompat"
	"github.com/2found/2ai/core/jsonjs"
)

// FlattenChatModelCatalog projects Pi's grouped catalog onto model IDs. The
// provider argument is type metadata in Pi; it never rewrites model fields.
// Use Object/Array containers to retain enumeration order and model identity.
// Missing type is excluded here, unlike the model collection's legacy-chat rule.
func FlattenChatModelCatalog(_ string, groups any) (*Object, error) {
	return flattenModelCatalog(groups, "chat")
}

func FlattenImageModelCatalog(_ string, groups any) (*Object, error) {
	return flattenModelCatalog(groups, "image")
}

func FlattenClassifierModelCatalog(_ string, groups any) (*Object, error) {
	return flattenModelCatalog(groups, "classifier")
}

func flattenModelCatalog(groups any, kind string) (*Object, error) {
	groupValues, err := jscompat.Values(groups)
	if err != nil {
		return nil, err
	}
	// Pi finishes flatMap before filtering, then finishes filtering before
	// coercing IDs in Object.fromEntries. Preserve those error boundaries.
	models := []any{}
	for _, group := range groupValues {
		values, err := jscompat.Values(group)
		if err != nil {
			return nil, err
		}
		models = append(models, values...)
	}
	selected := []any{}
	for _, model := range models {
		if jsonjs.IsNullish(model) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(model), "model.type")
		}
		if modelType, ok := jscompat.ReadProperty(model, "type").(string); ok && modelType == kind {
			selected = append(selected, model)
		}
	}
	result := NewObject()
	for _, model := range selected {
		id, err := jscompat.ToPropertyKey(jscompat.ReadProperty(model, "id"), make(map[*Array]bool))
		if err != nil {
			return nil, err
		}
		// Set defines an own property, including __proto__. Repeated IDs replace
		// the value without moving the key's first insertion position.
		result.Set(id, model)
	}
	return result, nil
}
