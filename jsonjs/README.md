# jsonjs

Shared JSON value semantics for native AI/agent transcripts. Preserve JavaScript
number rounding, numeric-key ordering, duplicate keys, missing versus null,
object identity and lone UTF-16 surrogates; this package is not a generic
replacement for encoding/json.

Raw-property decoding scans validated value boundaries without constructing
unused nested values. Unescaped valid UTF-8 strings use a direct path; escaped
strings, invalid UTF-8 and WTF-8 use the lossless UTF-16 path. Serialization
preallocates from known field sizes and preserves the existing escaping rules.
No parsed transcript cache or shared mutable value is introduced.

Verify with `go test -race ./jsonjs` from the module root. The fixtures include
source-runtime hashes for all 65,536 UTF-16 units and 1,048,576 surrogate pairs.
Run `go test ./jsonjs -run '^$' -bench 'DecodeObjectProperties|QuoteStringUTF8' -benchmem`
for allocation measurements. Integration callers in `ai` and `agentcore` must
also pass when these semantics change.
