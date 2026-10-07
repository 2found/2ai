package engine

import "github.com/2found/2ai/jsonjs"

// Object and Array retain nested identity across argument hooks and execution.
// Callers must synchronize concurrent access to shared values.
type Object = jsonjs.Object
type Array = jsonjs.Array
type Property = jsonjs.Property
type JSONMethod = jsonjs.JSONMethod

const Undefined = jsonjs.Undefined
const Null = jsonjs.Null

func NewObject(properties ...Property) *Object { return jsonjs.NewObject(properties...) }
func NewArray(values ...any) *Array            { return jsonjs.NewArray(values...) }
