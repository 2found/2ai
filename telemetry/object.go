package telemetry

import "github.com/2found/2ai/jsonjs"

// Property, Object and Array share the native JSON value implementation used
// by agent arguments. No JavaScript runtime or interface hierarchy is involved.
type Property = jsonjs.Property
type Object = jsonjs.Object
type Array = jsonjs.Array
type JSONMethod = jsonjs.JSONMethod

// StringifyValue preserves JSON.stringify's undefined root as an empty result.
// Only explicit JSONMethod values stored as toJSON hooks can execute on export.
func StringifyValue(value any) ([]byte, error) { return jsonjs.StringifyValue(value) }

func NewObject(properties ...Property) *Object        { return jsonjs.NewObject(properties...) }
func NewArray(values ...any) *Array                   { return jsonjs.NewArray(values...) }
func NewAttributes(properties ...Property) Attributes { return jsonjs.NewObjectValue(properties...) }
