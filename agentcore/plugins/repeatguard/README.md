# repeatguard

**Extension.** Ejectable — the loop never names it. Without it, a model that
re-issues the same call with identical arguments twenty turns running trips
nothing and burns the entire turn budget.

This fills a real hole. The other loop breakers all fire at the wrong altitude:
the goal gate catches a verbatim-repeated **answer**, compaction summarizes away a
stale duplicate result **after the fact**, and the circuit breaker only counts
tool **failures**. A tool that succeeds every time, called with identical
arguments forever, trips none of them.

Ported from deepseek-harness's `dsh-repeat-tool-reminder` (MIT).

## Model Experience

### The chain of identical calls reaches the first threshold (default: 3)

#### What the model sees

A synthetic user message, appended after **every** tool result in the batch — so
the model reads its own repeated output and the reminder together.

##### Verbatim text for this field

```
You are repeating the exact same tool call with identical arguments. Carefully
analyze the previous result before calling again: if the task is not complete,
try a different approach or different arguments instead of repeating the call.
```

#### Token effect

**Fixed**, ~50 tokens, at most once per threshold per chain.

#### KV cache effect

**Append-only.**

### The chain reaches a later threshold (default: 5, then 8)

#### What the model sees

The detailed form, naming what is actually happening:

```
Repeated tool call detected:
- tool: run_sql
- consecutive_calls: 5
- arguments: {"limit":100,"q":"select count(*) from events"}
The repeated calls are not making progress. Do not call this tool with these
exact arguments again. Inspect the latest result and choose a different action,
different arguments, or finish the task if enough evidence has been gathered.
```

#### Token effect

**Capped.** The quoted arguments are clamped to `ArgumentsPreviewChars`
(default 500 runes) with an explicit `… (+N more chars)` marker, so a looping
write-shaped payload cannot ride unbounded into every subsequent request. The
chain key always compares the **full** canonical string; only the reminder text
is bounded.

#### KV cache effect

**Append-only.**

### New material enters the conversation

A steer, a follow-up, or another extension's injection resets the chain. The
model has been given information it did not have, so its earlier repetition is
no longer the same chain of reasoning — otherwise the steer that fixes the loop
still eats the next reminder.

#### Token effect

**Zero-direct.**

## Impact on the agent

- **Never blocks a call.** A legitimately repeated poll is delayed by nothing.
  At the call site a real loop is indistinguishable from a legitimate retry;
  only the model knows which it is, so the plugin supplies the observation and
  leaves the judgment alone.
- Denied and disabled calls are counted too — a model hammering a blocked call
  is precisely the loop worth breaking.
- Arguments are compared **canonically** (nested keys sorted), because models
  re-emit the same object with keys shuffled constantly and treating those as
  different calls would miss most real loops.
- `update_plan` is excluded by default, and exclusion is *transparent*:
  `run_sql X → update_plan → run_sql X` still counts as two consecutive
  `run_sql X`, so bookkeeping cannot launder a loop.
- Thresholds are validated at composition. A value below 2 or a duplicate fails
  the build rather than producing a guard that silently never fires.

## Known limitations and deferred work

- **Consecutive only.** Alternating `A B A B A B` is a loop the plugin cannot
  see; it tracks one chain, not a cycle detector.
- **No cross-run memory.** Counters are per-run by design, so an agent that
  loops, is resumed, and loops again identically starts from zero.
- **Advisory only.** A model that ignores the reminder is unaffected; the run
  still relies on `MaxTurns` / `MaxToolCalls` as the real backstop.
- **The reminder is a user-role message.** On providers that weight the last
  user turn heavily it can pull focus away from the actual task.
