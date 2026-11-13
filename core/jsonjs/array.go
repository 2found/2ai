package jsonjs

import (
	"reflect"
	"slices"
	"strconv"
)

// Array retains JavaScript-style identity and length across shared references.
// Absent indices (holes) differ from present Undefined values in Has/Keys, but
// both read as Undefined and export as null. Enumerable own data properties
// outside array indices retain insertion order and do not affect length or JSON.
// An explicit JSONMethod stored as toJSON can replace the array during export.
// Custom prototypes and accessors are not represented.
// Callers must synchronize concurrent reads and writes.
type Array struct {
	length int
	values map[int]any
	named  *Object
}

func NewArray(values ...any) *Array {
	a := &Array{length: len(values), values: make(map[int]any, len(values))}
	for index, value := range values {
		a.values[index] = value
	}
	return a
}

func (a *Array) Len() int {
	if a == nil {
		return 0
	}
	return a.length
}
func (a *Array) Has(index int) bool {
	if a == nil || index < 0 || index >= a.length {
		return false
	}
	_, exists := a.values[index]
	return exists
}
func (a *Array) Get(index int) any {
	if !a.Has(index) {
		return Undefined
	}
	return a.values[index]
}
func (a *Array) Set(index int, value any) {
	if index < 0 || uint64(index) >= 4294967295 {
		panic("telemetry: array index out of range")
	}
	if a.values == nil {
		a.values = map[int]any{}
	}
	a.values[index] = value
	if index >= a.length {
		a.length = index + 1
	}
}
func (a *Array) Delete(index int) {
	if a != nil {
		delete(a.values, index)
	}
}

// GetProperty reads an indexed or named own data property. Inherited properties
// and the reserved length property are excluded; use Len for length.
func (a *Array) GetProperty(name string) (any, bool) {
	if a == nil {
		return Undefined, false
	}
	if index, ok := ArrayIndex(name); ok {
		return a.Get(int(index)), a.Has(int(index))
	}
	if value, found := a.named.Lookup(name); found {
		return value, true
	}
	return Undefined, false
}

// SetProperty defines an enumerable own data property. It bypasses prototype
// setters (including __proto__) and accessors. Use SetLength to change length.
func (a *Array) SetProperty(name string, value any) {
	if name == "length" {
		panic("use SetLength to set array length")
	}
	if index, ok := ArrayIndex(name); ok {
		a.Set(int(index), value)
		return
	}
	if a.named == nil {
		a.named = NewObject()
	}
	a.named.Set(name, value)
}

func (a *Array) DeleteProperty(name string) {
	if a == nil {
		return
	}
	if index, ok := ArrayIndex(name); ok {
		a.Delete(int(index))
		return
	}
	a.named.Delete(name)
}

// PropertyKeys follows Object.keys order: numeric indices, then named own data
// properties in insertion order. The non-enumerable length property is excluded.
func (a *Array) PropertyKeys() []string {
	keys := []string{}
	if a == nil {
		return keys
	}
	for _, index := range a.Keys() {
		keys = append(keys, strconv.Itoa(index))
	}
	for _, property := range a.named.Entries() {
		keys = append(keys, property.Name)
	}
	return keys
}
func (a *Array) SetLength(length int) {
	if length < 0 || uint64(length) > 4294967295 {
		panic("Invalid array length")
	}
	if length < a.length {
		for index := range a.values {
			if index >= length {
				delete(a.values, index)
			}
		}
	}
	a.length = length
}
func (a *Array) Append(values ...any) int {
	length := uint64(a.length) + uint64(len(values))
	if length > 4294967295 {
		panic("Invalid array length")
	}
	for _, value := range values {
		a.Set(a.length, value)
	}
	return a.length
}
func (a *Array) Pop() any {
	if a.Len() == 0 {
		return Undefined
	}
	index := a.length - 1
	value := a.Get(index)
	delete(a.values, index)
	a.length = index
	return value
}
func (a *Array) Keys() []int {
	keys := []int{}
	if a != nil {
		for index := range a.values {
			keys = append(keys, index)
		}
	}
	slices.Sort(keys)
	return keys
}

// Values returns a detached, dense outer slice. Holes become Undefined, as in
// JavaScript array spread; nested objects and arrays retain their identity.
func (a *Array) Values() []any {
	values := make([]any, a.Len())
	for i := range values {
		values[i] = a.Get(i)
	}
	return values
}
func (a *Array) MarshalJSON() ([]byte, error) { return MarshalValue(a) }

func referenceArray(value reflect.Value) (*Array, bool) {
	if value.IsValid() && value.Type() == reflect.TypeFor[*Array]() {
		return value.Interface().(*Array), true
	}
	return nil, false
}
func arrayIdentity(array *Array) valueIdentity {
	return valueIdentity{kind: reflect.Pointer, pointer: reflect.ValueOf(array).Pointer(), valueType: reflect.TypeFor[*Array]()}
}
