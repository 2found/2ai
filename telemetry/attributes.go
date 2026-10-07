package telemetry

import (
	"reflect"

	"github.com/2found/2ai/jsonjs"
)

func copyAttributes(attributes Attributes) (Attributes, bool) {
	copied := NewAttributes()
	for _, property := range attributes.Entries() {
		name, value := property.Name, property.Value
		// The inherited __proto__ setter never creates an own attribute.
		if name == "__proto__" {
			continue
		}
		if jsonjs.IsUndefined(value) {
			continue
		}
		v := reflect.ValueOf(value)
		if !jsonjs.SupportedValue(value) {
			return Attributes{}, false
		}
		if array, ok := value.(*Array); ok {
			copied.Set(name, NewArray(array.Values()...))
		} else if v.Kind() == reflect.Slice {
			cloned := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
			reflect.Copy(cloned, v)
			copied.Set(name, cloned.Interface())
		} else {
			copied.Set(name, value)
		}
	}
	return copied, true
}
