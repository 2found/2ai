// Package jscompat holds the JavaScript semantics that the core/ai port of Pi
// emulates: Number() and parseFloat parsing, ECMAScript whitespace, JSON
// repair and streaming parse, JSON.stringify, URLSearchParams encoding, lone
// surrogate sanitising, and the Object/property helpers (Object.values,
// property reads, strict equality, SameValueZero, spread) used by the model
// catalog and auth code.
//
// It is the leaf of the core/ai tree: it depends only on core/jsonjs and the
// standard library and never imports core/ai.
package jscompat
