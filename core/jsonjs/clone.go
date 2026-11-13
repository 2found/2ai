package jsonjs

import "reflect"

// ValueCloner detaches mutable JSON-shaped containers without serialization.
// One cloner preserves repeated references and cycles across multiple roots.
// Callers synchronize access to the source graph and to the cloner.
type ValueCloner struct {
	objects map[*Object]*Object
	arrays  map[*Array]*Array
	native  map[valueIdentity]reflect.Value
}

func (c *ValueCloner) Clone(value any) any {
	v := c.clone(reflect.ValueOf(value))
	if !v.IsValid() {
		return nil
	}
	return v.Interface()
}

func (c *ValueCloner) clone(value reflect.Value) reflect.Value {
	if !value.IsValid() {
		return value
	}
	if value.Kind() == reflect.Interface {
		if value.IsNil() {
			return value
		}
		result := reflect.New(value.Type()).Elem()
		result.Set(c.clone(value.Elem()))
		return result
	}
	if object, ok := objectValue(value); ok {
		if object == nil {
			return value
		}
		if c.objects == nil {
			c.objects = make(map[*Object]*Object)
		}
		copied, exists := c.objects[object]
		if !exists {
			copied = NewObject()
			c.objects[object] = copied
			for _, property := range object.Entries() {
				copied.Set(property.Name, c.Clone(property.Value))
			}
		}
		if value.Type() == reflect.TypeFor[ObjectValue]() {
			return reflect.ValueOf(ObjectValue{object: copied})
		}
		return reflect.ValueOf(copied)
	}
	if array, ok := referenceArray(value); ok {
		if array == nil {
			return value
		}
		if c.arrays == nil {
			c.arrays = make(map[*Array]*Array)
		}
		if copied, exists := c.arrays[array]; exists {
			return reflect.ValueOf(copied)
		}
		copied := NewArray()
		copied.length = array.length
		c.arrays[array] = copied
		for index, child := range array.values {
			copied.values[index] = c.Clone(child)
		}
		for _, property := range array.named.Entries() {
			copied.SetProperty(property.Name, c.Clone(property.Value))
		}
		return reflect.ValueOf(copied)
	}
	switch value.Kind() {
	case reflect.Map, reflect.Slice:
		if value.IsNil() {
			return value
		}
		identity := valueReference(value)
		if c.native == nil {
			c.native = make(map[valueIdentity]reflect.Value)
		}
		if copied, exists := c.native[identity]; exists {
			return copied
		}
		var copied reflect.Value
		if value.Kind() == reflect.Map {
			copied = reflect.MakeMapWithSize(value.Type(), value.Len())
			c.native[identity] = copied
			iter := value.MapRange()
			for iter.Next() {
				copied.SetMapIndex(iter.Key(), c.clone(iter.Value()))
			}
		} else {
			copied = reflect.MakeSlice(value.Type(), value.Len(), value.Len())
			c.native[identity] = copied
			for i := 0; i < value.Len(); i++ {
				copied.Index(i).Set(c.clone(value.Index(i)))
			}
		}
		return copied
	case reflect.Array:
		copied := reflect.New(value.Type()).Elem()
		for i := 0; i < value.Len(); i++ {
			copied.Index(i).Set(c.clone(value.Index(i)))
		}
		return copied
	default:
		// Scalars/functions are immutable. Unsupported arbitrary Go pointers and
		// structs remain unsupported JSON values; cloning does not invoke methods.
		return value
	}
}
