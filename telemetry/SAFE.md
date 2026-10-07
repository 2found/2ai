# Safe settled producer telemetry

`2ai.safe.v1` is a separate, opt-in projection for operational accounting. A
host attaches a `SafeObserver` to the execution context and pulls detached
`SafeRecord` values. The observer has no sink callback, dispatcher goroutine,
delivery retry or dedup store. The existing private recorder and synchronous
private export sink keep their behavior; private export is not a safe collector.

```go
observer, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{
    Capacity: 2048, RecordBytes: 4096,
})
if err != nil { return err }
ctx, err = telemetry.WithSafeObserver(ctx, observer, telemetry.SafeExecution{
    ExecutionID: "opaque-execution-token", Category: telemetry.CategoryMain,
})
if err != nil { return err }
labels, err := telemetry.ApproveModelLabels("configured-provider", "configured-model")
if err != nil { return err }
// Bind labels on ai.FallbackCandidate.SafeLabels or FallbackRequest.SafeLabels.
// Pass ctx to the producer/AgentCore. In a host-owned collector:
records, health := observer.Drain(256), observer.Health()
// Persist records and health using trusted host scope.
_ = labels
_ = records
_ = health
```

Approval is a host promise: labels are configured public identifiers, never
secrets, response text, arbitrary model metadata, endpoint/account names or user
identity. The library checks syntax, not meaning. IDs must be 1–128 ASCII bytes
from `[A-Za-z0-9._-]`. Invalid/oversize correlation or labels return an error
without truncation. Zero labels serialize as null with `labels_state: unavailable`.
A response provider/model selection that does not match the approved candidate
omits both labels, including internally selected response models. Tool labels
require `ApproveToolLabel` and explicit `NativeRun.SafeToolLabels` bindings.
Private names/attributes are not read to infer labels.

## Records and identity

The closed JSON schema lists every field/enum and recursively rejects additional
properties. Record kinds are:

| Kind | Meaning | Additive accounting |
| --- | --- | --- |
| `attempt_settled` | Attempted native producer open, settled after the actual stream ends; `producer_admitted` distinguishes open failure from a returned stream | Only this kind, using observed tokens/known cost |
| `span_settled` | Live eager span settlement with its own timing and structural outcome | Never |
| `reconciliation` | Existing reported host totals preserved separately for comparison | Never; `reconciliation.additive` is false |

An observer creates IDs from a random 128-bit prefix and monotonic sequence.
Execution correlation is host-approved and opaque. Event/span/call/attempt IDs
are assigned once when observation begins; drain never recreates them. Retry
and fallback rungs share a call ID. Ordinals may restart on fallback, so they
are not dedup keys. Independent private recorders reusing span ID 1 have unrelated
safe IDs. Parent references include their execution ID when known. Delivery
retries must retain the returned `producer_event_id`.

Native fallback emits one record per `StreamFn` producer open, including failed,
withheld and aborted attempts. OAuth rotations emit inner producer records and
suppress the outer composition record. Internal HTTP/socket retries within one
`StreamFn` remain one producer attempt. Completed durable reattachment without
a new producer emits no new attempts; new calls get new IDs. This is in-memory
settlement evidence, not crash-durable exactly-once delivery. Custom aggregate-only
callbacks have unavailable physical coverage and create no fictitious attempts.

Eager `Context.StartSpan`/generic `StartSpan` compose safe settlement using
`SafeContext(private, ctx)`. Propagate `span.Context()` with `WithContext` to
attach model attempts to their span. The legacy deferred `StartSpanFrom`
reader/admission contract is unchanged and does not independently create safe
span records. Built-in span kinds are `agent`, `model`, `tool`; other private
names become `unknown`. Categories are `main`, `compaction`, `advisor`, `memory`,
`child`; built-in auxiliary spans override the inherited category. Structural
status/failure enums never contain error text. Safe classification does not call
arbitrary error readers/serializers. `SetSafeOutcome` supplies bounded outcomes
without changing private status precedence.

`started_at`/`settled_at` are UTC wall timestamps; clock skew can reverse them.
`duration_ns` uses the observation's own monotonic clock, independently of root
duration. Terminal publication alone is not settlement: producers are joined
before attempt emission, including early host failure/cancellation paths. Retry
backoff belongs to enclosing spans, not physical attempt duration.

## Usage and pricing evidence

`accounting_schema: 2ai.disjoint-tokens.v1` preserves existing uncached input,
output, cache-read/cache-write buckets and pricing arithmetic. Usage/pricing
observation, attempt coverage and collector loss are independent. Unknown
numerical values are null, not trusted zero.

| Evidence | Result |
| --- | --- |
| Missing/default/JSON-decoded usage without provenance | Usage unavailable, tokens null, price unknown |
| Explicitly observed zero usage and known zero price | Observed/priced, zero tokens and known cost 0 |
| Observed usage without known price | Unpriced, tokens retained, cost null |
| Valid usage and explicit/native known rates | Existing cost result retained, priced |
| Negative/nonfinite accounting | Invalid source, affected observation unavailable; never clamp into trusted billing |
| Admission failure | Usage/pricing unavailable, `producer_admitted: false` |

Wire parsers track presence/validity of required usage counters. Native model
pricing is known only when all four disjoint rates (and applicable tiers) are
explicit finite nonnegative values; empty initialized metadata is unknown.
Explicit zero rates are known. There is no subscription/OAuth/free-price
inference or new price table. Custom producers supply `ai.Usage.Observation`
with `explicit` evidence. Transcript JSON excludes it; decoding a checkpoint
does not recreate observation. Sources are `wire`/`explicit` for usage,
`model`/`explicit` for pricing, or `unknown`/`invalid` when unavailable.

Per-record `complete` requires observed usage, priced cost and observed attempt
coverage. `attempt_coverage: unavailable` prevents summaries from claiming
complete physical coverage. Root/parent totals can include children and remain
comparison-only: never add them to attempt sums. `Health.Complete` separately
reports admission loss, even for individually complete records.

## Bounds, privacy and lifetime

Defaults/maxima: 2048 queued records, 4096 serialized JSON bytes per record,
128 bytes per identifier, 256 records per drain. Hosts may lower queue/record
caps. Nonpositive drain limits consume nothing. Admission uses bounded scalar
serialization, a nonblocking lock attempt and a nonblocking queue send. It
invokes no arbitrary callbacks and never waits for a consumer. Overflow,
oversize, closed admission and contention have separate monotonic counters;
any loss makes collection incomplete. Health snapshots are approximate until
producers quiesce. Dropped records are not automatically replayed; queues never
grow and admission creates no goroutine per record.

Safe records contain no prompts, messages, response/thinking/signatures, raw
errors, credentials, headers, endpoint/account data, tool arguments/results,
private attributes/events or serializer hooks. Scalar pointer fields are copied
before admission. Consumer edits cannot mutate provider/native transcripts or
private recorder objects. A consumer may block/error/panic independently; the
host owns delivery recovery, storage and health persistence.

Keep intake open across detached work: root settlement does not close it.
`DetachSafeExecution` creates an independent child execution with known origin
and cross-execution parent references. `CaptureSafeOrigin`/`BindSafeOrigin`
transfer safe metadata only, without cancellation/private recorder state.
Coalesced memory work with multiple/unknown origins omits the single origin;
it never chooses the last run's identity or grows an origin list. Late safe
children do not reopen a settled private recorder. Shut down/join host-owned
background work, then `Close` and drain pending records. Post-close records
are loss-counted; closing does not cancel work or destroy pending records.

## Frozen C4 fixture and T6 boundary

The schema/example are original synthetic 2ai fixtures with no live provider
data or changes to upstream fixture provenance. Eight attempts comprise seven
observed, six priced, one unpriced and one missing. Additive totals are
input/output/cache-read/cache-write `58/22/9/6` and known estimated cost `.028`
USD. Reconciliation is non-additive.

| Artifact | SHA-256 |
| --- | --- |
| [safe-c4-v1.schema.json](testdata/safe-c4-v1.schema.json) | `dbe8abf2638933231dfa416a63afad0f8cde0a03853f41785524b3400fe92521` |
| [safe-c4-v1.example.json](testdata/safe-c4-v1.example.json) | `5516930e7dec659c98a1cfa8520789c159a0cdab8d185772b8428d81b2645a23` |

T6 owns trusted runtime/execution-slice/tenant identity, authorization, durable
dedup (for example `(runtime_id, execution_slice_id, producer_event_id)`),
writer/store/API bounds, retention and the Soot dependency pin. 2ai implements
none of those stores/product scopes. These fingerprints identify a local
contract; a local commit is not a release receipt. Independent Review/QA and
explicit authorized versioned publication remain separate gates before
downstream release consumption.
