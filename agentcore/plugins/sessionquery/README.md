# sessionquery

**Extension.** Ejectable — the loop never names it. Without it, compaction is
lossy in practice: once the loop summarizes an older span, the detail is gone
from the model's context even though it is sitting intact in the append-only
log.

This is the read path that makes compaction safe. With it, the context can
shrink freely, because anything dropped is one query away.

## Model Experience

### The model searches its own history

#### What the model sees

One tool, `session_query`, taking a free-text query and returning ranked
matches from this session's durable log — **including spans compaction has
already summarized away**.

##### Verbatim text for this field

```
3 matches (searched 412 entries):

[seq 118, turn 9, tool] score 2.0
run_sql returned 1,284 rows; top account by spend was acct_9931 at $41,208.55…

[seq 61, turn 4, assistant] score 1.0
…
```

An empty query returns the most recent entries, which is how a model browses
rather than searches.

#### Token effect

**Capped.** `Limit` matches (default 10, hard cap 50) × `ExcerptBytes` per
excerpt (default 600). A query cannot return more than roughly 30 KB regardless
of how large the log is.

#### KV cache effect

**Append-only.** The result enters as an ordinary tool result.

### The run is not durable and no provider was supplied

#### What the model sees

Nothing. The plugin **declines the run**, so the tool is never advertised.

That is deliberate: a tool that can only ever answer "nothing" is worse than no
tool, because the model will call it, wait, and learn nothing.

#### Token effect

**Zero-direct.**

#### KV cache effect

**Independent.**

## Impact on the agent

- Adds one tool that bypasses the permission gate. It is `SelfGated`: the search
  is pinned to the run's own `SessionID` before every call, so a model-supplied
  value cannot widen the scope — it reads only text this agent already produced
  and was already shown.
- Changes the economics of compaction. An agent with this plugin can be given a
  much more aggressive `KeepRecentTokens` without losing the ability to answer
  questions about its own earlier work.
- The built-in provider **scans one session's log**. That is a bounded,
  already-scoped set, which is the only reason a scan is acceptable here.

## Known limitations and deferred work

- **A cross-session `Provider` is the consumer's responsibility to index.** The
  repo's standing search rule applies: it must be typo/accent tolerant and
  index-backed (pg_trgm GIN over a normalized key, queried with `%` /
  `similarity`) — never a bare `ILIKE '%q%'` and never a full scan. Nothing in
  this package enforces that.
- **Scoring is token-count then recency.** There is no phrase matching, no field
  weighting, and no relevance feedback. A query whose terms are common in the
  log returns near-arbitrary ordering.
- **Messages only by default.** Bookkeeping entries (compaction brackets, goal,
  tool-disable) are reachable only by naming `Kinds` explicitly, and the model is
  not told they exist.
- **No pagination.** Past `MaxLimit` there is no cursor; the model must
  re-query with narrower terms.
