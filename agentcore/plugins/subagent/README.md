# subagent

**Extension.** Ejectable — the loop never names it. Without it the agent is
solo: every step of every sub-task happens in its own context window.

What delegation buys the parent is **context, not compute**. A child explores in
its own isolated history and returns only its final answer, so the tool churn,
dead ends, and large intermediate results never enter the parent's window.

## Model Experience

### Delegation is available (run below `MaxDepth`)

#### What the model sees

One tool, `spawn_subagent`, with `task` and `context` parameters — plus an
`agent` parameter listing named teammates when a roster is configured, and an
optional `output_schema` (JSON Schema object) that turns the final answer into
validated JSON.

### Guidance belongs to the capability

The run extension owns the orchestration instructions in the `spawn_subagent`
**tool schema**. Applications supply the authorized roster and execution callbacks;
they do not need to append a second generic orchestrator system prompt.

The schema tells the model to handle simple work directly, delegate bounded tasks
when useful, supply only authorized context, collect required evidence, reconcile
gaps and return one coherent result. Partial results and uncertain effects are
not proof of success. Delegation does not grant authority.

Additional guidance follows the actual surface:

- **Self-fork:** fresh history, inherited capabilities subject to host restrictions.
- **Named roster:** choose only a destination whose stated purpose covers the
  task/project/resource. Missing access is a concrete blocker, not a reason to
  probe unrelated targets. Each callback owns its target's identity, inputs,
  history and permissions; descriptions are selection hints, never grants.
- **Async enabled, ephemeral run:** a job receipt is not completion; collect or
  cancel it with the run's job controls before reporting success.
- **Durable run:** neither the async parameter nor its instructions are offered.
- **Permission denied or depth cap:** the tool and all its guidance disappear
  through the existing policy/depth advertisement path.

This uses the existing `ToolContributor` extension; it adds no kernel hook,
router model, orchestrator persona, system-prompt block or application concept.
The schema is rebuilt for the current run and roster after checkpoint resume,
so old destination hints are not accumulated in the definition. Tool guidance is
present on native model requests alongside the advertised capability, including
following context compaction. Selection guidance improves model behavior; only
the host's grants and callbacks enforce resource access.

Host-specific boundaries (for example original-input-only forwarding, data
isolation and publication) belong in the consuming application. Domain workflows
and expertise belong in its skills or definition, not this plugin.

### `output_schema` — typed final answers

When `output_schema` is set, the child's task gains an instruction to end with
a single bare JSON value matching the schema. The plugin validates the final
answer itself — a forked child cannot carry the provider's structured-output
seam — and on a violation re-opens the child exactly once with the error. The
parent receives the validated JSON; if the retry also fails it receives the
raw answer suffixed with `[validation failed: …]`, so a bad shape is visible
rather than silent.

An ephemeral correction resumes the child's native checkpoint with a new
validation-error message. For durable execution, the host runs the correction
at `<childSession>/retry`, seeded from the original native checkpoint. A
completed original child remains immutable. The deterministic id keeps the spawn replay-safe — a re-issued spawn
reattaches to whichever child log completed. A delegate has no transcript to
re-open (`Delegate.Run` is an opaque closure), so its retry is a single
re-invocation carrying the error and the rejected answer in the task.

For native durable runs, a human question from either the original child or
its corrective retry parks the parent. Answering the parent forwards to the
child's recorded question. Resume reattaches the same child and re-applies
schema validation and output limits before delivering its answer. Additional
questions keep the same delegation and receive distinct workflow IDs.

#### Token effect

**Fixed schema overhead plus isolated child context.** The common orchestration
guidance is attached once to the advertised tool; named/async guidance appears
only with those capabilities. The child's entire run — every tool
call, every intermediate result — is replaced in the parent's context by one
answer capped at `MaxOutputBytes` (default 48 KB). This reduces parent-context
pressure for substantial child work; it does not promise fewer total model tokens
or lower latency for small tasks.

#### KV cache effect

**Append-only** for the parent. The child runs against its own prefix entirely.

### The run is already at `MaxDepth`

#### What the model sees

Nothing. The plugin **declines the run**, so the tool is never advertised.

Enforcing the cap by absence rather than by refusal matters: a tool that is
offered and then always refuses is something the model must read, reason about,
and work around.

#### Token effect

**Zero-direct.**

## Impact on the agent

- The tool **is** gated. Unlike `read_skill` / `read_spill` / `job_*` /
  `session_query`, spawning reaches capability the agent has not already
  exercised, so the consumer must permit `spawn_subagent` in its policy.
- A child is built by `Agent.Fork`, which lives in core precisely so scope can
  only **narrow**: the child inherits every capability-bearing field verbatim —
  provider, ladder, tools, policy (including the installed permission gate),
  memory, definition, limits, env, compaction, retry, caching, extensions — and
  drops the run-control seams (durable session by default, steering, follow-up,
  step gate, `PrepareNextTurn`). A child is one bounded task, not a
  conversation.
- Depth rides the **context**, not a field, so an A → B → A cycle is bounded by
  the same `MaxDepth` as a straight chain, even when the two agents are composed
  with different plugin sets.
- A durable parent gives the child a durable session at
  `parentSession + "/" + invocationKey`. The key identifies the recorded
  physical invocation, even when the provider repeats a tool-call ID. With a
  native session host installed, self-delegation is retry-safe: a replayed spawn **reattaches** —
  a completed child returns its recorded answer without re-running (no duplicate
  spend or side effects), an interrupted one resumes from its own log.
- A **delegate**-routed spawn is not retry-safe. `Delegate.Run` is an opaque
  closure under the target agent's identity with no reattach wiring, so
  replaying it would re-run the teammate's entire task; those calls are left
  dangling for the model to decide.
- Child usage — including a **failed** child's — is folded into the parent via
  `AddChildUsage`, so the parent's budget gate accounts for what its children
  spent.
- Spawns are `Parallel`, so a fan-out turn ("spawn three, then synthesize") runs
  them concurrently.
- **A cancelled child fails the spawn; it does not answer it.** A cancelled run
  is not an error to its caller — the loop stops between turns and returns what
  it has with `StopReason: "aborted"` and a nil error, which is right for a
  viewer that walked away and wrong here. `res.Final` is then whatever the child
  last happened to say, mid-task, and returning it hands the parent a killed
  child's partial state as its *answer*: the shard is recorded as reconciled, the
  batch looks complete, and the interrupted work is never redone. The spawn fails
  instead, which puts something the model can act on in the transcript and leaves
  the call replayable.

## When a fan-out is interrupted

The native session host owns child journals and recovery. Completed children
reattach without new model calls or usage; interrupted children resume only
from recorded native state and settled tool receipts. Missing or ambiguous
completion receipts fail closed instead of repeating effects or treating a
partial answer as completion. Cancellation reaches all in-flight children.

`internal/runtime/pi_children_integration_test.go` exercises native isolation,
reattachment, interrupted-child recovery, corrective retry reattachment,
cancellation, parallel sessions and parked questions. Plugin tests separately
verify the fork callback receives the inherited agent, invocation identity and
original native checkpoint.

## Known limitations and deferred work

- **The parent sees only the final answer.** `output_schema` can make that
  answer validated JSON, but there are no partial results and no way for a
  child to hand back an artifact reference instead of prose.
- **No per-child budget.** `MaxPerRun` counts spawns, not tokens or cost; one
  expensive child can consume the whole run's budget.
- **A child cannot be steered.** No steering queue, no step gate — once started
  it runs to completion or cancellation.
- **`Delegate` is fully opaque.** agentcore cannot verify that a delegate's
  scope is narrower than the caller's; that guarantee is the consumer's.
- **Depth is the only recursion bound.** There is no detection of a semantic
  cycle below `MaxDepth` (A asks B the same question A was asked).

## Native execution

Async children publish tool-name and lifecycle progress through the run-owned
`agentcore.WithRunProgress` observer, automatically bound by `RunNative`.
They never retain the spawn tool's streaming callback after it returns. Native
delivery is serialized and fenced when the run ends; cancelled jobs suppress
further notes. Synchronous children still use the spawn tool's emitter. These
notes are display activity, not partial answers or child transcript content.
Other native session hosts may bind the same run-owned observer while preserving
their own cancellation, serialization and end-of-run fence.

`Plugin.RunFork` is an optional consumer hook for self-delegation. The plugin
still creates the child with `Agent.Fork`, applies its depth and spawn limits,
validates output, and folds usage into the parent. Without the hook, ephemeral
children use `RunNative` directly. Durable children require the hook and a
recorded invocation key; there is no legacy driver or provider-call-ID fallback.

The hook receives `ForkRequest` with the durable child session ID, prompt,
recall task, and (for a corrective attempt) the previous `RunResult`. Native
adapters must seed retries from `Previous.NativeState`, never from the display
projection in `Previous.Messages`. The adapter owns native persistence,
completion verification, reattachment, and cancellation. A completed reattach
must report zero new usage. Native self-forks derive session IDs from the
persisted invocation key so reused provider call IDs remain distinct.

The native adapter distinguishes a parked child from a completed child. Its
structured error identifies the durable child question, and reattaching while
that question is pending performs no model work. Answering the child session
allows that same fork request to resume. The parent plugin still returns this
as a delegation error; forwarding the question and answer through the parent's
human-input workflow requires consumer integration.
