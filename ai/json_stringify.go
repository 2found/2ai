package ai

import "github.com/2found/2ai/jsonjs"

// StringifyJSON normalizes serialized JSON using JavaScript value semantics,
// retaining numeric index-key ordering, binary64 numbers and UTF-16 surrogates.
func StringifyJSON(raw []byte) ([]byte, error) {
	return jsonjs.StringifyJSON(raw)
}
