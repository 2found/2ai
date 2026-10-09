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

## Store ownership and lifecycle

The foundation supplies interfaces and tools, not a durable database or ranking
policy. The host implements `MemoryStore` and owns identity, retrieval indexes,
capacity, expiry and retention. There is no built-in vector store or guaranteed
semantic duplicate/contradiction detector.

`MemoryEntry` separates kind (fact, preference, procedure, learning, episode,
context, outcome) from ownership and lifetime. Optional metadata includes a fact
key, expiry, last confirmation, observation count and host-assigned conversation /
run / rollout sources. A source proves where evidence came from, not that its
content is true. Model-authored classifications remain inferred evidence.

## Model-facing tools

All tools remain behind the normal permission gate. A nil store disables them.

- `learn` persists a learning through the original `Remember` seam. Stores
  implementing optional `ScopedMemory` also advertise host-approved target aliases,
  typed entries, keys, TTL and explicit replacement IDs. The model never supplies
  a tenant/user identifier. Read permission does not imply write permission.
- `memory_recall` retrieves 1–20 previews (default 8), with a query of at most
  2000 bytes and content previews of at most 2000 bytes. Results must belong to
  the current scope or a host-declared readable target. Optional `MemoryReader`
  enables `id` instead of `query` for a complete entry before curation.
- `memory_edit` requires `MemoryCurator`: update content, or soft-retract with
  forget/invalidate. Optional `MemoryEraser` adds explicit erase of retained memory
  revisions. Conversation transcripts have a separate host retention policy.

Credential-pattern redaction runs before explicit retention and staged evidence.
It handles common token formats, authorization headers, credential assignments,
private keys and credentials embedded in URLs. It is a defense in depth filter,
not general PII recognition or a guarantee against every secret representation.
Hosts must also sanitize their persistence boundary and enforce every read/write
capability, including edits by ID. Shared publication guidance in tool descriptions
is not an approval workflow; the host remains the authority.

Recall is assembled once per run and bounded in bytes, not tokens. Lifetime and
ranking depend on the host; merely implementing `MemoryStore` does not add TTL,
deduplication or erasure semantics to an existing store.

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

Consolidation errors retain already-staged evidence and never fail the primary
answer. A staging failure has no durability guarantee; hosts must observe
`OnConsolidationError` rather than infer memory retention from reply success. The deferred
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

Queue overflow and shutdown retain durable evidence. The host can page its durable
pending index at startup and call `Recover(ctx, plugin, scope)` after rechecking
current authority and provider policy. Recovery waits for capacity notifications
without polling or replaying a primary run. The worker itself does not enumerate
host storage; a later successful run also re-admits pending work. Background usage
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
