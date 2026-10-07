package engine

import (
	"encoding/json"
	"reflect"
	"slices"
	"strconv"
	"sync"

	"github.com/2found/2ai/ai"
)

// List preserves collection identity when entries or length change. Container
// operations are synchronized; callers still synchronize edits to shared items.
// Values returns a detached, dense slice; Has distinguishes holes from zero T.
type List[T any] struct {
	mu     sync.RWMutex
	length int
	values map[int]T
	named  map[string]T
	order  []string
}

type MessageList = List[*ai.Message]
type ToolList = List[*Tool]

func NewList[T any](values ...T) *List[T] {
	list := &List[T]{length: len(values), values: make(map[int]T, len(values))}
	for i, value := range values {
		list.values[i] = value
	}
	return list
}

func (l *List[T]) Len() int {
	if l == nil {
		return 0
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.length
}

func (l *List[T]) Get(index int) T {
	if l == nil {
		var zero T
		return zero
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.values[index]
}

func (l *List[T]) Has(index int) bool {
	if l == nil {
		return false
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	_, ok := l.values[index]
	return ok
}

func (l *List[T]) Set(index int, value T) {
	if index < 0 || uint64(index) >= 4294967295 {
		panic("Invalid array index")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.values == nil {
		l.values = make(map[int]T)
	}
	l.values[index] = value
	if index >= l.length {
		l.length = index + 1
	}
}

func (l *List[T]) Delete(index int) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.values, index)
}

// Array properties outside canonical indices do not affect length or JSON.
// The reserved length property is exposed through Len and SetLength instead.
func listPropertyIndex(name string) (int, bool) {
	index, err := strconv.ParseUint(name, 10, 32)
	return int(index), err == nil && index < 4294967295 && strconv.FormatUint(index, 10) == name
}

// GetProperty reads indexed or ordinary enumerable data properties, excluding
// inherited properties and the reserved length property.
func (l *List[T]) GetProperty(name string) (T, bool) {
	var zero T
	if l == nil {
		return zero, false
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	if index, ok := listPropertyIndex(name); ok {
		value, exists := l.values[index]
		return value, exists
	}
	value, exists := l.named[name]
	return value, exists
}

// SetProperty defines an enumerable own data property, including names such as
// __proto__; it does not invoke JavaScript prototype setters or accessors.
func (l *List[T]) SetProperty(name string, value T) {
	if name == "length" {
		panic("use SetLength to set array length")
	}
	if index, ok := listPropertyIndex(name); ok {
		l.Set(index, value)
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.setNamed(name, value)
}

// Caller holds mu.
func (l *List[T]) setNamed(name string, value T) {
	if l.named == nil {
		l.named = make(map[string]T)
	}
	if _, exists := l.named[name]; !exists {
		l.order = append(l.order, name)
	}
	l.named[name] = value
}

func (l *List[T]) DeleteProperty(name string) {
	if l == nil {
		return
	}
	if index, ok := listPropertyIndex(name); ok {
		l.Delete(index)
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.named, name)
	if index := slices.Index(l.order, name); index >= 0 {
		l.order = slices.Delete(l.order, index, index+1)
	}
}

// PropertyKeys includes enumerable own data properties in JavaScript key order.
// Keys returns only array indices. Neither method includes the length property.
func (l *List[T]) PropertyKeys() []string {
	keys := []string{}
	if l == nil {
		return keys
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	indices := make([]int, 0, len(l.values))
	for index := range l.values {
		indices = append(indices, index)
	}
	slices.Sort(indices)
	for _, index := range indices {
		keys = append(keys, strconv.Itoa(index))
	}
	return append(keys, l.order...)
}

// Assignment to array[array.length - 1] writes the ordinary "-1" property
// when a callback emptied the list during an assistant stream.
func (l *List[T]) setLast(value T) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.length == 0 {
		l.setNamed("-1", value)
		return
	}
	if l.values == nil {
		l.values = make(map[int]T)
	}
	l.values[l.length-1] = value
}

func (l *List[T]) SetLength(length int) {
	if length < 0 || uint64(length) > 4294967295 {
		panic("Invalid array length")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	for index := range l.values {
		if index >= length {
			delete(l.values, index)
		}
	}
	l.length = length
}

func (l *List[T]) Append(values ...T) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	if uint64(l.length)+uint64(len(values)) > 4294967295 {
		panic("Invalid array length")
	}
	if l.values == nil {
		l.values = make(map[int]T)
	}
	for _, value := range values {
		l.values[l.length] = value
		l.length++
	}
	return l.length
}

func (l *List[T]) Keys() []int {
	keys := []int{}
	if l == nil {
		return keys
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	for index := range l.values {
		keys = append(keys, index)
	}
	slices.Sort(keys)
	return keys
}

func (l *List[T]) Values() []T {
	if l == nil {
		return []T{}
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	values := make([]T, l.length)
	for index, value := range l.values {
		values[index] = value
	}
	return values
}

func (l *List[T]) MarshalJSON() ([]byte, error) {
	// Release the list lock before serializing items; their graph is user-owned.
	values := []any{}
	if l != nil {
		l.mu.RLock()
		values = make([]any, l.length)
		for index, value := range l.values {
			values[index] = value
		}
		l.mu.RUnlock()
	}
	return json.Marshal(values)
}

func (l *List[T]) UnmarshalJSON(raw []byte) error {
	var values []T
	if err := json.Unmarshal(raw, &values); err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.length, l.values = len(values), make(map[int]T, len(values))
	l.named, l.order = nil, nil
	for index, value := range values {
		l.values[index] = value
	}
	return nil
}

// Clone copies indexed entries, retaining item references and sparse slots.
// Like Array.slice, it drops ordinary named properties.
func (l *List[T]) Clone() *List[T] {
	if l == nil {
		return NewList[T]()
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	if constructor, exists := l.named["constructor"]; exists && !listDefaultConstructor(constructor) {
		panic("Species construction did not get a valid constructor")
	}
	return l.copyIndexedLocked()
}

// Internal atomic read of indexed membership. Reading an array does not invoke
// its constructor/species; only public Clone follows Array.slice construction.
func (l *List[T]) indexedSnapshot() *List[T] {
	if l == nil {
		return NewList[T]()
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.copyIndexedLocked()
}

// Caller holds mu. The snapshot keeps holes distinct from present zero values.
func (l *List[T]) copyIndexedLocked() *List[T] {
	copied := NewList[T]()
	copied.length = l.length
	for index, value := range l.values {
		copied.values[index] = value
	}
	return copied
}

// JSON-shaped objects have no Symbol.species, so slice uses the default array
// constructor. An own primitive constructor (other than undefined) rejects.
// Custom JavaScript constructor/species functions have no native representation.
func listDefaultConstructor(value any) bool {
	if value == Undefined {
		return true
	}
	v := reflect.ValueOf(value)
	seen := map[uintptr]bool{}
	for v.IsValid() && (v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface) {
		if v.IsNil() {
			return false
		}
		if v.Kind() == reflect.Pointer {
			address := v.Pointer()
			if seen[address] {
				return true
			}
			seen[address] = true
		}
		v = v.Elem()
	}
	if !v.IsValid() {
		return false
	}
	switch v.Kind() {
	case reflect.Map, reflect.Slice:
		return !v.IsNil()
	case reflect.Array, reflect.Struct:
		return true
	default:
		return false
	}
}
