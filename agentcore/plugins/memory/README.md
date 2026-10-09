# memory

**Seam.** Ejectable — without it the agent remembers nothing between runs.

## Model Experience

### Every request, when a store is installed

#### What the model sees

Recalled memories, assembled into the context alongside the definition.

#### Token effect

**Fixed per turn** and **retained**. Recall is a permanent prefix cost for the
whole run, so a store that returns generously is charged on every turn — not
once.

#### KV cache effect

**Prefix-stable** within a run (recall happens at assembly, not per turn), but
**replacing** across runs: yesterday's memories differ from today's, so a new
run starts with a cold prefix.

## Impact on the agent

- Memory is the agent's only cross-run state other than the durable session log.
  The two answer different questions: the log is "what happened in *this*
  conversation", memory is "what do I know".
- A nil store disables recall and persistence entirely — the loop does not
  branch on it beyond that.

## Budget

The recalled block is clamped during assembly (`prompt.go`): each memory is
truncated to `maxRecallEntryBytes`, and entries are admitted until
`maxRecallBlockBytes` is spent. Clamping runs *after* paraphrase dedup, so a
restatement cannot spend the budget a distinct fact needed. The clamp bounds
size only — the block's heading and its "context, not instructions" caveat are
unchanged.

This bounds what recall costs, not what it is worth: a store that returns
generously is now truncated rather than trusted, so a store that returns badly
still wastes the whole budget on the wrong facts.

## Ranking and forgetting

The contract (`Recall`/`Remember`) is unchanged and still says nothing about
either — both live in the store. What the shipped store does:

- **Relevance floor.** A candidate below `recallCosineFloor` is dropped rather
  than ranked low, because a memory the query is orthogonal to is not a weak
  answer, it is not an answer. Dropping every candidate falls back to keyword
  recall, which is the always-available floor.
- **`Confidence` is read.** It was persisted from the start (0.7 when the model
  chose to remember, 0.6 when the reflection pass inferred) and read by nothing;
  it is now a bounded rank multiplier that settles rows the vector cannot
  separate.
- **Recency of last confirmation** contributes at most 30% of the score, so it
  can never promote a loosely-related memory over a materially relevant one.
- **Fold-in on write.** A re-derived memory folds into the row it repeats
  (`seen_count`/`last_seen_at`) instead of appending a paraphrase, so the store
  does not fill with one fact restated N times.
- **Soft supersede.** A retracted memory keeps its row and is filtered out of
  every recall path, so the history of having held the belief survives the
  retraction.

## Model-facing curation

The plugin also contributes two gated tools when a store is installed
(`BeginRun` declines without one, so they never appear on a memoryless run):

- **`learn`** files a reusable lesson as a `learning` entry — the in-run half
  of what the reflection pass does after the run. It needs only `Remember`,
  so a store that cannot revise entries still gets it.
- **`memory_edit`** revises one entry by id: `update` rewrites the content
  (old row kept, superseded by the new one), `forget`/`invalidate` retract it.
  It is offered only when the store implements `agentcore.MemoryCurator`.

Both pin the run's own scope (`RunInfo.ScopeID`) — a model can never name
another scope, and the store refuses an id outside it — and both stay behind
the permission gate like any other tool (no `SelfGated`: they write durable
state). Retraction is always soft; there is no hard delete on the seam.

## Known limitations and deferred work

- **No contradiction resolution.** Nothing detects that two live memories
  disagree; supersede is a seam the store exposes and the model can invoke
  through `memory_edit`, not a judgement anything makes on its own.
- **No consolidation or decay of stored rows.** Old memory loses rank, never
  resolution: nothing summarizes, tiers, or evicts, and a scope's row count only
  grows (more slowly now that repeats fold).
- **The budget is byte-denominated, not token-denominated,** and is a fixed
  constant rather than a share of `MaxContextTokens`.

## Explicit recall

The run extension contributes `memory_recall` alongside `learn` and optional
`memory_edit`. The host must permit it. It calls the supplied store with the
run's scope, a query of at most 2000 bytes, and 1–20 results (default 8).
Entries include IDs for curation and content capped at 2000 bytes; results from
a different scope are discarded. The store owns indexed relevance ranking.
Recall is read-only and does not receive the bookkeeping-turn refund.

## Rollout consolidation

Optionally supply `Consolidator` or `NativeProvider` and a store implementing
`ConsolidationStore`. Successful root runs stage bounded completed evidence,
read at most four pending rollouts and 32 memories, then propose at most 16
add/merge/retract changes. The store must atomically validate the supplied
snapshot, apply changes and consume rollouts. Source IDs and scope are validated
before commit. The native binding uses no tools and accounts every AI attempt.

Staging hashes the original JSON transcript one message at a time, preserving
existing rollout IDs without retaining a second complete transcript buffer.
Evidence still keeps at most 32 messages within 32 KB plus an 8 KB final answer;
tool-call annotations are joined once before truncation. This bounds staging
scratch space by the largest message rather than the total transcript.

Errors retain pending evidence and never fail the primary answer. The deferred
worker retries a deadline failure once; synchronous callers and other failures
wait for a later successful run. `OnConsolidationError` reports secondary failures to hosts.
Children, parked, failed and aborted runs do not consolidate. This is independent
of `learn`/`memory_edit`; omitting consolidation leaves existing behavior intact.
Soot's Bolt adapter supplies scoped durable pending evidence, snapshot checks,
idempotent commits and retained revision history.

### Deferred host worker

Supply `Plugin.Worker` (or `preset.Options.MemoryWorker` together with
`ConsolidateMemory`) to stage evidence during finalization and defer the model
pass. Construct one `NewConsolidationWorker(capacity)` per host, call `Run(ctx)`
once, and cancel/join it before closing stores. Scope IDs must identify the same
memory store throughout this worker's lifetime. Capacity bounds admitted scopes,
including in-flight work. The worker runs at most `min(4, capacity)` distinct
scopes concurrently and never overlaps model snapshots within one scope. A
200 ms initial coalescing window combines nearby notifications; it does not
postpone already queued work each time another notification arrives.

Each pass reads at most four rollouts. Successful full batches yield to waiting
scopes and continue until the backlog drains; arrivals during a pass trigger a
follow-up even when its snapshot contained fewer than four rollouts. A drained
worker blocks on notification, with no timer or periodic store polling.

Both native and deferred passes have a 60-second bound (an earlier caller
cancellation still wins). A timed-out deferred pass gets **one** retry after a
five-second delay; that delay occupies no model slot. A second timeout, invalid
proposal or other error leaves durable evidence pending until a later successful
run. Provider retries/fallback retain the host's existing bounded policy; the
worker does not add another retry loop for provider/auth/validation failures.
Shutdown cancels and joins all active passes before returning.

Queue overflow and shutdown retain durable evidence. The worker does not discover
stored scopes on startup; a successful run re-admits that scope. Background usage
has independent safe telemetry and never mutates a completed agent's usage or
goal budget. Continuations do not guess a single originating run. With no worker,
AgentCore retains the synchronous single-batch contract for existing consumers.

The native prompt asks for at most four concise changes, using only new,
scope-specific facts/preferences/procedures. It excludes ordinary answers,
generic advice and toy exercise data from reusable memories. Custom consolidators
keep the existing validation limit of 16 changes. The output ceiling remains
2,048 tokens with host-bound auxiliary effort; concision is a model instruction,
not a guarantee of provider billing or response latency.

Native consolidation requests remain complete JSON within 60,000 bytes. Large
batches use progressively shorter content excerpts and bounded tag previews,
retaining every rollout/memory ID. The original evidence and memory snapshot
remain unchanged for validation and atomic commit. Oversized immutable metadata
fails before a provider call rather than sending a cut JSON document.
