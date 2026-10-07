# telemetry

Native Go port of Pi's callback telemetry runtime at
`eeac84ca92498ac18b6832754d01aef1d3c5f654`. Production code uses only the Go
standard library. There is no subprocess, Bun, Node, or TypeScript dependency.

[Safe settled producer telemetry](SAFE.md) defines the separate typed bounded
pull observer, provenance rules and frozen C4 schema/example. The private
recorder/export contract below is preserved. Settled wall timestamps and
monotonic durations are also available through `InMemory.GetSpanTimings` and
private export `Batch.SpanTimings`, without changing Pi span JSON.

This module uses concrete `Context`, `Span`, and `InMemory` types. The zero
`Context` is a no-op. `NewInMemory()` owns an isolated recorder; a span supplies
the explicit context for its children. Callbacks run in the calling goroutine.
`StartSpan[T]` preserves a typed return value and error; the `StartSpan` methods
accept callbacks returning only an error. Panics retain their original value.

```go
recorder := telemetry.NewInMemory()
answer, err := telemetry.StartSpan(recorder.Context,
    telemetry.SpanOptions{Name: "operation"},
    func(span *telemetry.Span) (string, error) {
        span.AddEvent("ready", telemetry.NewAttributes(telemetry.Property{Name: "cached", Value: false}))
        return "answer", nil
    })
spans := recorder.GetSpans()
```

`Attributes` is an ordered object, constructed with `NewAttributes(Property{...},
...)`; use `Get`, `Lookup`, `Set`, `Delete`, `Len` and `Entries` instead of map
indexing. Its zero value is empty. Copies share the input object, while recorder
admission and snapshots copy the outer object and arrays. `NewObject` constructs
shared nested objects with the same property operations. JSON decoding also
creates these nested `*Object` values. Replacing a value keeps its original key
position; deleting and re-adding a key moves it to the end. Numeric index keys
enumerate first in numeric order. `Entries` returns a detached entry list whose
nested values remain shared. All native host call sites use this ordered API.

`NewArray(values...)` supplies shared array identity. JSON decoding creates
`*Array` values too. `Get`, `Has`, `Set`, `Delete`, `Append`, `Pop`, `Len`,
`SetLength` and `Keys` operate on that same array even after growth or truncation.
Missing elements differ from present `Undefined` values in `Has`/`Keys`; both
read as `Undefined` and export as `null`. `Values` returns a detached dense slice
with holes filled by `Undefined`. Recording and snapshotting spread-copy direct
array attributes, including filling holes, while arrays nested inside another
array or object remain shared. Sparse indexed storage avoids allocating all
intervening slots during a distant `Set` or length growth. `GetProperty`,
`SetProperty`, `DeleteProperty` and `PropertyKeys` expose indexed and named
enumerable own data properties. Numeric indices enumerate first; other names
retain insertion order. Named properties survive length changes and remain live
on nested arrays and status fields. Direct attribute spread copies drop them;
JSON excludes them, even when they contain cycles or functions. `SetProperty`
defines an own property for `__proto__`, without invoking a prototype setter.
Length remains exposed through `Len`/`SetLength`. Custom prototypes and accessors
are not represented. An explicit `JSONMethod` stored under `toJSON` can replace
the array during JSON export.
Plain Go slices remain accepted and retain their Go types; use `Array` when
shared growth/truncation must behave like JavaScript.

`StartSpanFrom` accepts an options reader on both `Context` and `Span`; the
package-level generic helper also preserves a typed callback result. No-op
contexts and settled parents skip reading. If a reader panics or supplies an
unsupported Go payload, the callback still executes once with a no-op span and
keeps its original result, error or panic. Failed admission consumes no span ID.
Readers run outside the recorder lock; a parent that settles before reading
finishes prevents child admission. Spans created by the reader retain their own
IDs, parentage and settlement independently of that outer admission.

Recorded spans retain Pi's JSON field names, start order, explicit parent IDs,
settlement order, attribute merging, ordered events, and explicit status
precedence. Only the exact status `"ok"` normalizes to success and drops error
metadata; all other readable names, including the Go zero value, normalize to
`"error"` and preserve supplied error details. A status set before settlement
suppresses automatic error inspection.
Once inspection starts, its computed error status overwrites a status set
reentrantly by the error reader, matching Pi's assignment order. Error inspection
runs outside the recorder lock, so snapshots and mutations remain available.
`SetStatusFrom(func() SpanStatus)` models a status read that can fail or reenter
recording. No-op or settled spans do not invoke the reader. A reader panic leaves
the outer status and explicit-status flag untouched; mutations made by the
reader itself remain. Successful reads apply their result after those mutations.
The reader runs outside the recorder lock. If another goroutine settles the span
while a reader is blocked, the completed read cannot overwrite that settlement.
Attribute objects and their outer arrays are copied. Mutations after settlement are
inert; late child callbacks still execute with no-op telemetry. A child admitted
before its parent settles can finish independently. The recorder supports
concurrent Go callers. Callers must synchronize concurrent mutation of their
shared input and snapshot maps and slices.

Attribute values distinguish null from absence: `nil` records JSON `null`,
including when replacing an existing value. `telemetry.Undefined` omits an
attribute and leaves an existing value unchanged during merges. Inside nested
objects, undefined fields remain present in memory but disappear from JSON;
undefined array elements export as `null`. This also preserves null attributes
decoded from JSON. Callers previously using nil for omission must use
`telemetry.Undefined` instead. Function values remain present in snapshots;
JSON omits them from objects and exports them as `null` in arrays. Recording
and export never invoke ordinary functions. An explicit `JSONMethod` stored as an
own `toJSON` property is the exception at export: it receives the object/array
and containing key, and its return value replaces that position. Recording,
snapshotting and graph cloning never call it. Export propagates callback errors
and panics, reads later property values after earlier hooks, and captures object
keys/array length before visiting children. It does not call a returned object's
hook again at the same position. `StringifyValue` preserves an undefined root as
no bytes; Go's `json.Marshaler` adapter must instead return `null`. Go's outer
`encoding/json` encoder may also apply HTML escaping to the returned bytes.

`SetAttributesFrom` and `AddEventFrom` accept attribute readers. Active spans
read once; no-op and settled spans skip reading. A panic suppresses the outer
operation while preserving events, status and attribute edits already made by
the reader. Successful attribute merges use the attributes captured before the
read, matching Pi's `mergeAttributes` evaluation order: a reentrant attribute
update may be overwritten by the outer merge. Nested events remain before the
outer event. Readers run without holding the recorder lock, and a span that
settles before the reader returns ignores the pending operation.

Go adaptations: ordinary Go errors produce `{name: "Error", message: err.Error()}`; other panic values
produce an error status without details. This includes `panic(nil)`: Go's runtime
wrapper is excluded from recorded error details while the original recovered
panic is propagated to the caller. Unsupported Go attribute payloads (channels, arbitrary pointers/structs and maps with
non-string keys) in JSON-visible positions are ignored atomically. Named array
properties are not visited by array export or spread copying. These adaptations
do not invoke user serialization methods.
An explicit `*ErrorDetails` preserves a serialized failure's original name and
message while retaining normal Go error identity; the recorder copies its fields.

Decoded status JSON follows Pi's property reads. A null status is ignored without
setting explicit status; other primitive inputs are readable and normalize to
error. Falsy `error` values are omitted, while truthy primitive/array values
produce empty error details. Only exact `"ok"` skips error inspection entirely.
`ErrorDetails` retains missing, null and non-string name/message fields from JSON
without inventing empty strings or keeping unrelated properties. Native string
edits override their decoded values; ordinary Go literals retain both string
fields. `NameValue` and `MessageValue` expose the actual fields, including
`Undefined`, null, nonfinite numbers and shared `*Object`/`*Array` values.
`SetNameValue` and `SetMessageValue` replace a field on just that error details
value, including explicit empty strings, null and `Undefined`. The field
descriptors remain immutable across shallow status copies, but nested objects
and arrays retain their references. Edits through those references affect other
copies and settled recorder snapshots, matching Pi's shallow error-field copying.
Callers must synchronize shared edits. JSON export retains the attribute value
domain and only invokes explicitly registered `JSONMethod` hooks, never arbitrary
field functions or custom Go serializers; cycles and
unsupported Go field payloads in JSON-visible positions return an export error.

The shared `internal/jsonjs` normalizer preserves numeric index-key ordering,
duplicate-key semantics, binary64 rounding/overflow and UTF-16 surrogate escapes
for these decoded fields. AI partial parsing, declarations and proxy encoding use
the same Go implementation through `ai.StringifyJSON`.

Attribute JSON follows Pi's number domain: Go integer/float widths share the
numeric category, including within `[]any`; byte slices serialize as number
arrays. JSON export converts numbers to binary64, rounds integers beyond its
exact range, emits nonfinite values as `null`, and emits negative zero as `0`.
The recorder and detached snapshots retain their original Go types and values.
Export extracts primitives without invoking user-defined serialization methods.

Decoding an attribute object uses binary64 values too: valid numeric overflow
becomes infinity in memory, and underflow retains signed zero. Nested objects
and arrays use the same conversion; duplicate keys keep their final value.
A successful decode replaces the object, while a failed decode leaves the previous
object and its aliases untouched. Null decodes to the empty zero value. Non-object top-level
inputs remain outside the typed `Attributes` boundary. The shared
`internal/jsonjs.DecodeJSON` parser preserves lone UTF-16 surrogates as WTF-8
bytes inside Go strings, keeping distinct string values and map keys separate.
Valid surrogate pairs use ordinary UTF-8. Attribute export restores the original
surrogate escapes, including in nested objects and arrays. Preserve these strings
as opaque values or serialize them through `Attributes`; ordinary Go rune
iteration and `encoding/json` on a plain string do not preserve lone surrogates.
Span options and recorded span/event names use the same UTF-16 decoding and
export rules. Ten pinned-source cases verify names through admission, active
and settled snapshots, and importing serialized records again.
Ordered attributes and decoded nested objects preserve source key insertion
order. Plain Go maps remain accepted as nested values, but their insertion order
cannot be recovered; those values export deterministically. Use `NewObject` or
decoded objects when order matters.

`SchemaDefinition` and its span/event/attribute definitions preserve the
serializable metadata, including enum/example arrays, sensitivity, cardinality,
parent vocabularies, and explicit empty events. `DefineSchema` returns the same
pointer. `CreateTypedSpanStarter` binds a context; its `StartSpan` callback
receives the span and a starter bound to that span for children. `StartTypedSpan`
preserves a generic result. Schema values are never inspected at runtime:
duplicate or unknown names do not introduce runtime validation absent from Pi.
Go does not infer literal attribute types from a schema value. TypeScript's
schema-derived overloads/exact attribute constraints have no direct Go equivalent;
the Go API exposes metadata and runtime behavior, not those compile-time checks.

Other backends use `NewContextWithCallbacks` with `ContextCallbacks` and `NewSpan`
with `SpanCallbacks`. These concrete callbacks provide the same explicit-parent
extension point without an interface hierarchy. `NewContext` still binds an
eager-only `StartSpan` callback; use `NewContextWithCallbacks` to additionally
bind `StartSpanFrom`. Missing context operations run the callback with no-op
telemetry and never evaluate a deferred reader. The backend owns callback admission, settlement, passive recording,
and concurrency; the constructors do not hide backend failures or change results.
Backends can forward `SpanCallbacks.SetStatusFrom`, `SetAttributesFrom` and
`AddEventFrom` to preserve deferred read admission and passivity; opaque
contexts leave absent callbacks inert.
Supplied status callbacks receive explicit null status inputs and retain their
own failure behavior; only the in-memory recorder ignores unreadable null status.
Thirty-two pinned-source cases compare direct and typed backend delegation,
nested/retained child starters, callback results and thrown-value identity.
An opaque span can use an in-memory root context for its children without owning
a recorded parent span. Its missing mutation callbacks remain inert, including
deferred readers; creating children still records through that context. Six Go
regressions cover this boundary for events, attributes and status.
`telemetry/testing.CreateAdapterConformance` supplies ten runner-independent
cases. Each case creates and closes its own `AdapterFixture`, checks normalized
snapshots, and returns an error on failure. Run the suite for any new backend.

## Verification and remaining work

The `pi-tojson.json` fixture contains 128 pinned-source placement/return cases
and 13 source-runtime mutation cases. These verify export-only hook execution,
receiver identity, containing keys, array spread behavior, root/property omission,
replacement values, errors, cycles, later-field mutations, and captured array
length. Native tests also cover map mutation during cycle detection, passive
cloning, and Go error/panic identity. These tests run entirely in Go.

`go test -race ./telemetry/...` runs without a JavaScript runtime. Thirty-nine recorded
fixtures were generated by the unchanged Pi implementation, covering nested
parentage, active snapshots, settlement, errors, explicit status, attributes,
events, late child callbacks, and reentrant/unreadable error inspection.
Additional Go tests cover result/error/panic
identity, detached snapshots, passive error inspection, and concurrent callers.
Another 24 source-generated cases verify thrown null, booleans, numbers, strings,
arrays, plain error-shaped objects and named errors, with automatic or explicit
status. These compare settled snapshots and check that recording preserves the
original failure, including map/slice/error identity in Go.
Seven additional schema/starter oracle scenarios check metadata round trips,
cross-schema children, root reuse, late children, no-op execution and passive
schema inputs. The Go conformance suite runs all nine mapped source cases plus a
supplemental normalization case against both the in-memory recorder and a backend
composed entirely through public callbacks. The mapped names are checked against
the original nine-case export. Throwing options, attribute, event and status readers
model failed property access; readable unknown status names remain a separate
case. Inertness checks also verify that settled spans do not invoke readers.
Forty-eight additional original-source getter cases cover active/no-op/settled
spans, ordinary/null throws, reentrant status mutations and snapshot reads, with
callback success/failure. Each runs against native and callback-forwarded Go
contexts. A gated Go concurrency check verifies that a blocked reader neither
holds the recorder lock nor changes a span that settled meanwhile. This is an
explicit Go callback boundary, not support for arbitrary JavaScript proxies.
Sixty-four original-source attribute getter cases cover successful/failed reads,
null/undefined values, nested attribute/event/status writes and snapshots across
active/no-op/settled spans. Both native and callback-forwarded contexts run every
case. Gated Go checks cover concurrent settlement; an additional Go-only case
checks unsupported payload rejection after reentrant writes.
Sixty-four more original-source option getter cases cover failures at name,
attributes and attribute-value reads, changes made during reading, reader-created
spans, root/child/no-op/late admission, and callback result/error identity. Native
and callback-forwarded contexts each run every case. Go checks also cover callback
panic identity, nil readers, concurrent parent settlement and opaque backends
without deferred admission. The conformance suite now uses throwing readers for
all unreadable-input cases. These are explicit callback boundaries, not a JavaScript
proxy implementation. Go errors/goroutines adapt failures and promise scheduling.
Failure-path tests verify adapter rejection and fixture cleanup. Twenty-four additional span fixtures compare empty, missing,
unknown and case-sensitive status names, dropped/retained error details, success,
callback failures and suppression of error inspection against the actual source.
Another 69 source-generated attribute cases compare active/settled JSON,
including byte arrays, mixed numeric widths, nonfinite values, signed zero,
large integers, and merge/event behavior. Comparisons retain JSON number text
so float decoding cannot conceal an integer-rounding difference. Go tests cover
passive export, preserved snapshots, and atomic rejection of unsupported Go values.
Sixty additional source-generated cases parse external attribute JSON before
span admission, merging and event recording. They compare active/settled JSON
and the exact binary64 bits of decoded and recorded numbers, covering overflow,
signed underflow, subnormal rounding, nested values, duplicate keys and special
property names. Go checks also cover reused receivers, atomic rejection, nested
options/event decoding and importing exported nulls into recorded spans.
Another 60 source cases exercise Unicode values and property names during span
admission, merging and event recording. Separate memory and JSON projections keep
surrogate escapes as text so the fixture decoder cannot hide replacement or key
collisions. They cover lone/pair/reversed surrogates, distinct surrogate keys,
duplicate escaped keys, replacement characters, nested arrays/objects and control
characters. The shared decoder/string exporter also matches source-runtime hashes
for all 65,536 single UTF-16 code units and 1,048,576 valid surrogate pairs.
Another 64 source cases compare the exact complete span JSON before/after
mutations and at settlement, through both decoding and native constructors.
They cover overwrite position, delete/reinsert, numeric index keys, duplicate
JSON keys, prototype-related keys, nested shared edits, detached snapshot edits,
undefined merges and reentrant attribute readers. Go checks cover object aliases,
cyclic ordered graphs and passive export after settlement.
Another 112 source cases compare complete JSON, array lengths, own index keys,
undefined/null/hole values and reference identity across direct, object-nested,
array-nested and repeated references. Each runs through both native construction
and decoded arrays. Input/snapshot edits cover append, distant assignment,
deletion, length growth/truncation and pop. Go checks cover cyclic arrays,
passive function export, unsupported payload rejection, detached dense views
and sparse storage at JavaScript's maximum array index.
Another 240 original-source cases compare named properties on direct, nested,
shared and status arrays, including input/snapshot edits before and after
settlement. They compare complete JSON, property order and values, reference
identity, constructor/prototype-related names, undefined values and named cycles
or functions. Seven source-runtime graph-clone cases verify that detached copies
preserve sparse membership, named fields, shared children and cycles. A native
provider-stream snapshot test verifies the same graph preservation and producer
isolation at the AI observation boundary. Native checks also cover the maximum
index versus an ordinary numeric-looking name and ignored opaque Go payloads.
The recorder does not enforce homogeneous arrays or reject readable nested
objects merely because they lie outside the TypeScript attribute declaration.
Twenty-four additional value cases cover mixed arrays, nested arrays/objects,
null elements, nested special keys, merges, and recursive number conversion.
Four graph fixtures check shallow copying and shared nested identity through
input/snapshot edits. Cyclic graphs remain recordable; JSON export returns an
error until the cycle is removed. Its diagnostic uses Go wording. No export
calls user-defined serialization methods. Nested object/array edits can affect
later snapshots, including settled spans, matching Pi's shallow-copy behavior;
this is distinct from late Span mutation methods, which remain inert.
Attribute copying also matches Pi's plain-object behavior: `__proto__` is not
recorded as an own attribute; `constructor`, `toString` and `prototype` remain
ordinary attribute names. Source fixtures cover admission, updates and events.
Eleven of the attribute cases cover null versus undefined at admission, during
merges and in nested objects/arrays. The original span-action fixtures now encode
undefined attributes explicitly instead of using null as a test-only marker.
Eight cases cover function values at admission, merging, events and nested
positions; memory inspection verifies that JSON omission does not discard them.
Additional Go checks verify own-property presence, JSON round trips, marker
retention in snapshots and shared nested edits after settlement.
Another 228 original-source cases verify status JSON with no prior status,
explicit success/error, and callback success/failure. They compare active/settled
spans, complete serialized status JSON and individual error property encodings.
Cases cover integer rounding, overflow, negative zero, nested key order,
name/message order and distinct UTF-16 surrogates. Go checks verify detached
status copies, native edits, JSON round trips and
atomic rejection of malformed JSON; malformed JSON diagnostic wording remains Go's.
Another 48 source cases compare complete status JSON and field reference identity
for explicit status and automatically inspected failures. Distinct/shared objects
and arrays are edited through the input or retained snapshot, replaced at the
outer error field, or mutated after settlement. They verify that field replacement
stays detached while nested edits remain shared, and automatic failures retain
the original error identity. Go checks cover explicit field absence/null/empty
strings, live overflow and signed zero, cycles and passive function export.

Run `go test -race ./telemetry/...` against the recorded source fixtures.
The TypeScript reference and generators have been removed. Source provenance
is retained in `third_party/pi/UPSTREAM.json`; the original MIT notice is in
`LICENSE.pi`. The native host records request spans and settled trace packets
through this package and is the server default.

The ordered value containers and JSON codec are shared with the native
agent engine through `internal/jsonjs`. Telemetry exports concrete aliases;
recorder-specific attribute admission, outer-array copying and status handling
stay in this package. Serializing an input object preserves its own properties;
the recorder applies its attribute filtering when accepting that input.
