# spill

**Extension.** Ejectable — the loop never names it. Without it, an oversized
tool result is head+tail truncated and the omitted middle is gone from the run
for good.

## Model Experience

### A tool returns more bytes than the inline cap

#### What the model sees

A bounded preview — head and tail, because the end of a long result usually
carries the signal — followed by an omission notice, and a new tool it can call
to read any part of what was cut.

##### Verbatim text for this field

```
(Omitted 41827 of 42394 bytes. Full result saved at: spill_9f3c1a…. Call
read_spill with this locator (and an offset/limit in bytes) to read any part of it.)
```

A backend may replace the trailing sentence with its own `RetrievalHint`.

#### Token effect

**Capped.** The replacement is provably `<= MaxInlineBytes`: the notice's byte
cost is reserved out of the cap before the preview is sized. Since the original
was above the cap, spilling can never make a result larger. What it changes is
not the size but the *recoverability* — the same budget now buys a pointer to
everything instead of a hole.

#### KV cache effect

**Append-only.** The replacement is written once, at the position the raw result
would have occupied, and never rewritten. Nothing earlier in the prefix moves.

### The model calls `read_spill`

#### What the model sees

A byte window with an explicit position header, so paging is unambiguous:

```
[bytes 0-8192 of 42394]
<content>
```

The tail read appends `, end of artifact`.

#### Token effect

**Capped.** One call returns at most 8 KB (`defaultSpillReadLimit`), regardless
of what the model asks for. A large artifact is paged, never dumped.

#### KV cache effect

**Append-only.**

### The run has no spill store, or the save fails

#### What the model sees

Exactly what it would see without this plugin installed: the loop's own
head+tail truncation. The plugin returns "no opinion" rather than an error.

#### Token effect

**Zero-direct.** No change against the baseline.

#### KV cache effect

**Independent.**

## Impact on the agent

- Adds one tool (`read_spill`) that bypasses the permission gate. It is
  `SelfGated`: it can only return text **this session** produced and had
  truncated away. The store is fenced by session id, and a store implementing
  `SpillOwner` is asked directly before every read, so a locator minted by
  another run reads back as not-found — deliberately indistinguishable from a
  locator that never existed.
- Takes over result bounding by declaring `Replace`, which is why interceptors
  receive the **raw** result: a lossless bound cannot be built on top of an
  already-truncated string.
- A failed tool call is left alone — its "result" is an error string the loop is
  about to overwrite, so persisting it would spend storage on nothing.

## Known limitations and deferred work

- **No eviction.** `MemorySpillStore` holds every artifact for the lifetime of
  the store. A long-lived process that spills heavily grows without bound;
  supply a store with a retention policy.
- **No cross-run retrieval.** A locator is fenced to its session, so a resumed
  run under the same session id can read its own spills, but a *different* run
  investigating the same task cannot. There is no operation to grant access.
- **The preview split is fixed** at two-thirds head / one-third tail. A result
  whose signal sits in the middle is previewed badly, and there is no way for a
  tool to declare a better shape.
- **The provider still sees a text notice.** The opaque locator also rides
  `ToolTrace.SpillLocator` and durable `Message.ResultRef`, so context pruning
  and fallback compaction can keep it reachable, but provider wire formats do
  not gain a vendor-specific artifact block; the active model reads the notice.
