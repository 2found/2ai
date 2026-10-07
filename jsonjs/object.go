package jsonjs

import (
	"slices"
)

// Object retains own-property insertion order and shared nested identity.
// Set overwrites in place; Delete followed by Set appends a new property.
// Entries enumerates numeric index keys first, as Object.entries does in Pi.
// Callers must synchronize concurrent access. Recording and JSON export never
// invoke arbitrary user serializers.
type Object struct {
	keys   []string
	values map[string]any
}

func NewObject(properties ...Property) *Object {
	object := &Object{values: map[string]any{}}
	for _, property := range properties {
		object.Set(property.Name, property.Value)
	}
	return object
}

func (o *Object) Get(name string) any { value, _ := o.Lookup(name); return value }
func (o *Object) Lookup(name string) (any, bool) {
	if o == nil {
		return nil, false
	}
	value, exists := o.values[name]
	return value, exists
}
func (o *Object) Len() int {
	if o == nil {
		return 0
	}
	return len(o.values)
}
func (o *Object) Set(name string, value any) {
	if o.values == nil {
		o.values = map[string]any{}
	}
	if _, exists := o.values[name]; !exists {
		o.keys = append(o.keys, name)
	}
	o.values[name] = value
}
func (o *Object) Delete(name string) {
	if o == nil {
		return
	}
	if _, exists := o.values[name]; !exists {
		return
	}
	delete(o.values, name)
	index := slices.Index(o.keys, name)
	o.keys = slices.Delete(o.keys, index, index+1)
}
func (o *Object) Entries() []Property {
	if o == nil {
		return []Property{}
	}
	entries := make([]Property, 0, len(o.keys))
	for _, name := range o.keys {
		entries = append(entries, Property{Name: name, Value: o.values[name]})
	}
	slices.SortStableFunc(entries, func(a, b Property) int {
		ai, ax := ArrayIndex(a.Name)
		bi, bx := ArrayIndex(b.Name)
		if ax && bx {
			if ai < bi {
				return -1
			}
			if ai > bi {
				return 1
			}
			return 0
		}
		if ax {
			return -1
		}
		if bx {
			return 1
		}
		return 0
	})
	return entries
}
