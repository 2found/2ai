# finishguard

**Extension.** Ejectable — the loop never names it. Without it, the first
answer the model produces is the answer the run returns.

Generalized from hermes-agent's turn-end verification guard.

## Model Experience

### The model produces a final answer and the guard is not satisfied

#### What the model sees

The guard's text, injected as a synthetic user message, and the run continues
instead of returning. The message is whatever the guard returns — this plugin
supplies the mechanism and the cap, not the wording.

AgentRay's evidence guard (`internal/runtime/evidence_guard.go`) is the shipped
example:

```
[System: your answer states figures, but no data tool was executed in this run.
If the figures come from data-tool results earlier in this conversation, keep
them and briefly note that provenance. Otherwise either verify them now with a
granted read tool (run_sql, explore_events, activity_summary, ...) and correct
the answer, or restate the answer saying explicitly that the figures are
estimates not read from project data.]
```

The user also sees a curated progress note (`verifying before finishing`) —
never the raw injection, which is internal scaffolding.

#### Token effect

**Fixed** per nudge, bounded by `MaxNudges` (default 2) per run. The real cost
is not the message but the extra turn it buys.

#### KV cache effect

**Append-only.** The nudge is appended after the rejected answer; nothing
earlier moves.

### The guard is satisfied, or the cap is spent

#### What the model sees

Nothing. The run returns the answer it produced.

#### Token effect

**Zero-direct.**

## Impact on the agent

- Consulted only on a **normal** finish — the model produced a text answer with
  no tool calls. A budget wrap-up, a tool-budget stop, a `MaxTurns` stop, an
  abort, or a terminal tool never re-opens the run: those are stops the run did
  not choose, and nudging them would wedge it.
- Runs **before** the follow-up drain. A stop interceptor is about the answer
  just produced; follow-ups are new input for after an accepted finish.
- The nudge is persisted to the durable log like a steer, so a resumed run
  replays the conversation the model actually saw.
- The cap is enforced against `StopInfo.Attempt`, a count the **loop** keeps —
  so a guard that forgets to bound itself is still bounded.
- A panicking guard accepts the finish rather than taking the run down.
- The injection is reported as external input, so a repeat detector does not
  treat the turn after a nudge as a continuation of the one before it.

## Known limitations and deferred work

- **The guard sees only passive evidence** — the answer text and the tool trace.
  It cannot run a check of its own, call a tool, or consult a model.
- **`Continue` with nothing to inject is treated as accepting the finish**,
  because re-running the same turn against an unchanged conversation would spin.
  A guard that wants to stop the run must say so with `StopReason`.
- **First interceptor to say `Continue` wins.** Two stop interceptors in one
  composition do not compose their nudges; the later one is not consulted that
  turn.
- **No cross-run budget.** `MaxNudges` is per run, so a resumed run gets a fresh
  allowance.
