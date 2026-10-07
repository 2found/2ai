package ai

import "github.com/2found/2ai/jsonjs"

// Object and Array carry mutable metadata without serializing nested values.
type Object = jsonjs.Object
type Array = jsonjs.Array
type Property = jsonjs.Property
type JSONMethod = jsonjs.JSONMethod

const Undefined = jsonjs.Undefined
const Null = jsonjs.Null

func NewObject(properties ...Property) *Object { return jsonjs.NewObject(properties...) }
func NewArray(values ...any) *Array            { return jsonjs.NewArray(values...) }
