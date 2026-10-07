package ai

import (
	"errors"
	"math"
	"strings"

	"github.com/2found/2ai/jsonjs"
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
	groupValues, err := catalogValues(groups)
	if err != nil {
		return nil, err
	}
	// Pi finishes flatMap before filtering, then finishes filtering before
	// coercing IDs in Object.fromEntries. Preserve those error boundaries.
	models := []any{}
	for _, group := range groupValues {
		values, err := catalogValues(group)
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
		if modelType, ok := catalogProperty(model, "type").(string); ok && modelType == kind {
			selected = append(selected, model)
		}
	}
	result := NewObject()
	for _, model := range selected {
		id, err := catalogKey(catalogProperty(model, "id"), make(map[*Array]bool))
		if err != nil {
			return nil, err
		}
		// Set defines an own property, including __proto__. Repeated IDs replace
		// the value without moving the key's first insertion position.
		result.Set(id, model)
	}
	return result, nil
}

func catalogValues(value any) ([]any, error) {
	if jsonjs.IsNullish(value) {
		return nil, errors.New("Object.values requires that input parameter not be null or undefined")
	}
	values := []any{}
	switch value := value.(type) {
	case *Object:
		for _, property := range value.Entries() {
			values = append(values, property.Value)
		}
	case *Array:
		for _, key := range value.PropertyKeys() {
			entry, _ := value.GetProperty(key)
			values = append(values, entry)
		}
	case string:
		for _, point := range jsonjs.StringCodePoints(value) {
			if point > 0xffff {
				point -= 0x10000
				values = append(values, catalogCodeUnit(0xd800+point>>10), catalogCodeUnit(0xdc00+(point&1023)))
			} else {
				values = append(values, catalogCodeUnit(point))
			}
		}
	}
	return values, nil
}

func catalogCodeUnit(point rune) string {
	if point >= 0xd800 && point <= 0xdfff {
		return string([]byte{byte(0xe0 | point>>12), byte(0x80 | point>>6&63), byte(0x80 | point&63)})
	}
	return string(point)
}

func catalogProperty(value any, name string) any {
	switch value := value.(type) {
	case *Object:
		if property, exists := value.Lookup(name); exists {
			return property
		}
	case *Array:
		if property, exists := value.GetProperty(name); exists {
			return property
		}
	}
	return Undefined
}

// Catalog IDs normally are strings. These conversions retain the observable
// behavior of JSON-shaped malformed catalogs without invoking serializers.
func catalogKey(value any, visiting map[*Array]bool) (string, error) {
	if jsonjs.IsUndefined(value) {
		return "undefined", nil
	}
	if value == nil || jsonjs.IsNull(value) {
		return "null", nil
	}
	switch value := value.(type) {
	case string:
		return value, nil
	case *Object:
		if _, exists := value.Lookup("toString"); exists {
			return "", errors.New("No default value")
		}
		return "[object Object]", nil
	case *Array:
		if _, exists := value.GetProperty("toString"); exists {
			return "", errors.New("No default value")
		}
		if _, exists := value.GetProperty("join"); exists {
			return "[object Array]", nil
		}
		if visiting[value] {
			return "", nil
		}
		visiting[value] = true
		defer delete(visiting, value)
		parts := make([]string, value.Len())
		for i := range parts {
			item := value.Get(i)
			if jsonjs.IsNullish(item) {
				continue
			}
			var err error
			parts[i], err = catalogKey(item, visiting)
			if err != nil {
				return "", err
			}
		}
		return strings.Join(parts, ","), nil
	case float64:
		if math.IsNaN(value) {
			return "NaN", nil
		}
		if math.IsInf(value, 1) {
			return "Infinity", nil
		}
		if math.IsInf(value, -1) {
			return "-Infinity", nil
		}
	}
	raw, err := jsonjs.StringifyValue(value)
	return string(raw), err
}
