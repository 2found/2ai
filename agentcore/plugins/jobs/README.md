# jobs

**Extension.** Ejectable — the loop never names it. Without it every tool is
synchronous: a long crawl, build, render, or export blocks the run until it
finishes, and a run that has to wait cannot also think.

## Model Experience

### A tool goes asynchronous

The tool calls `agentcore.LaunchBackground(ctx, tool, label, run)` and returns
immediately with the JSON receipt (`id`, `tool`, `label`, `state`,
`started_at`). What the model sees in the tool result is whatever that tool
chose to say; the convention is to include the receipt.

#### Token effect

**Zero-direct** at the moment of the call. The plugin's cost arrives later, in
the completion notice.

#### KV cache effect

**Append-only.**

### Jobs finished since the last turn

#### What the model sees

A synthetic user message at the top of the next turn, before the model reasons.

##### Verbatim text for this field

```
Background job update:
- job_7f2a (crawl_site, succeeded, 4.21s):
<up to 2 KB of the result inline>
- job_91bc (render_pdf, succeeded, 61.4s): 184320 bytes of output — call job_status with id "job_91bc" to read it.
- job_c3d1 (crawl_site, failed, 0.98s): dial tcp: connection refused
```

A small successful result rides inline so the common case costs no extra tool
call; a large one is named and left for `job_status`.

#### Token effect

**Capped.** Inline output is bounded at 2 KB per job (`noticeInlineLimit`),
and a job's stored result is bounded at the run's `MaxToolResultLen` (16 KB
when unset) at completion time, before it can be read at all.

#### KV cache effect

**Append-only.** The notice is appended after the previous turn's messages and
never rewritten.

### The model inspects or stops work

#### What the model sees

Four tools: `job_list`, `job_status`, `job_wait`, `job_cancel`. `job_wait`
blocks for at most 120s and then returns the job's current state, so the model
cannot park the run forever on work that never finishes.

A running job is not a black box. Its work publishes output as it goes
(`agentcore.PublishJobOutput(ctx, text)`); the board keeps the first 2 KiB and
the last 8 KiB and counts the rest, so memory stays bounded however much the job
prints. `job_status` on a running job returns its state and the newest 2 KiB of
output (`output_bytes` asks for more, never beyond the run's tool-result limit);
`job_list` adds the newest output line under each running job; a `job_wait` that
times out carries the newest 1 KiB. A job that fails or is cancelled keeps up to
4 KiB of its last output beside its error. A job that never publishes renders
exactly as before. The `run_shell` / `computer_use` background mode streams the
command's output into its job, and an async `spawn_subagent` publishes a start
line and its current tool or progress note.

A work function that runs its tool under another context's values (to keep the
call's policy and session) must build it with
`agentcore.BackgroundValues(jobCtx, values)`; a wrapper that hides `jobCtx`'s
values silences `PublishJobOutput` and `HandleJobMessages`.

### Optional: steering a running sub-agent (`Plugin{AllowSend: true}`)

Adds `job_send {id, message}`. The message (at most 4 KiB, at most 8 unread per
child) is queued for a running background sub-agent and reaches it at its next
step boundary as a user message, persisted in the child's own transcript (and so
in its checkpoint); a message that arrives while the child composes its final
answer holds the finish for one more step. It is refused with a reason when the
job is unknown, already finished, or not a sub-agent (any job whose work did not
register a handler with `agentcore.HandleJobMessages`). The child keeps its
depth cap, its policy intersection and its budgets: steering is a user message,
not a capability. Only `spawn_subagent` self-forks register a handler; a named
delegate has no step boundary to deliver to. A message that arrives after the
child's last step is returned in the job result as undelivered. Off by default:
it changes the tool list the model sees.

### Optional: work that outlives the run (`Plugin{Survive: &Survival{...}}`)

Adds `job_keep {id}`. Only a job the model keeps survives; every other job is
cancelled when the run ends, exactly as without the option, and a kept job does
not hold up the final answer. See `Survival` for what the host owns. In short:

```go
hold := jobs.NewHold(func(e jobs.Ended) { /* start a run of e.Conversation carrying e.Notice */ })
plugin := jobs.Plugin{Survive: &jobs.Survival{Hold: hold, MaxAge: time.Hour, MaxJobs: 4}}
ctx := jobs.WithConversation(ctx, conversationID) // per run, passed to RunNative
```

State machine of one job:

```
running --job_keep--> running(kept)
running(kept) --run ends--> detached in the Hold (goroutine keeps running)
detached --next run of the same conversation begins--> running(kept) on that board
running/running(kept) --work returns / job_cancel / MaxAge--> succeeded | failed | cancelled
ended while a run is attached --> announced by that run's next step, once
ended while detached          --> announced through Hold's callback, once (the next run then lists it as announced)
ended and announced, run closes --> leaves the Hold; the run's checkpoint carries the record
announced, no run comes back  --> leaves the Hold MaxAge after it ended
process restart               --> Hold is empty; the restored checkpoint reports it interrupted
```

The Hold keeps only what a later run still needs: running kept jobs, and a
finished one until the run its notice starts adopts it. Nothing finished stays
in process memory for a conversation nobody returns to.

The job's context is detached from the run's cancellation but keeps the run's
values, so it runs under the policy of the run that started it; a later run only
lists, reads and cancels it, and cannot widen it. Kept jobs are found by
conversation identity only: another conversation sees nothing. `MaxAge` (default
1 h, counted from the job's start) cancels a kept job; `MaxJobs` (default 4)
caps the kept jobs running per conversation. **Model-backed work cannot be kept**:
`spawn_subagent` jobs (and any tool named in `Survival.Exclude`) are refused by
`job_keep` with that reason, because a child's token usage is folded into the
Agent that spawned it, which has finished by the time a surviving child would.

#### Token effect

**Capped**, by the same result bound applied at completion.

#### KV cache effect

**Append-only.**

## Impact on the agent

- Adds four tools that bypass the permission gate. They are `SelfGated`: every
  run has its own board, so they observe and cancel only work **this run**
  launched — a capability the agent already exercised by starting the job.
- Binds a launcher onto the run context (`RunContext`) through
  `agentcore.WithBackgroundLauncher`. `agentcore.LaunchBackground` fails when
  the plugin is absent, so a background-capable tool degrades to synchronous.
- The job's context derives from the **run's**, not the calling tool's, so work
  survives the call that started it and still dies with the run.
- `CloseRun` cancels everything this run started. A job outliving its run would
  be unobservable and uncancellable — nothing left to look at it.
- The board holds at most `MaxConcurrent` running jobs (15 by default, `-1` for
  no ceiling) and 16 finished, announced records; older announced records are
  evicted. A record no notice has carried yet is never evicted.
- **The board survives resume** through the native plugin checkpoint
  (`NativeState` / `RestoreNativeState`). Finished results stay readable with
  `job_status`, unannounced completions are announced on the first resumed
  step, and ids keep counting so a new job never reuses an id the transcript
  names.

## Known limitations

- **Running work does not survive the run, unless the host opts in** (see
  `Survival`). Even then only the process's memory keeps it; a goroutine cannot be
  checkpointed. The checkpoint is taken before `CloseRun` cancels the work, so a
  job still running then is recorded as `cancelled` with the error
  "interrupted: the run ended before this job finished; start it again if it is
  still needed", and the resumed run's first step announces it. Durable host
  tasks belong on the consumer's own task queue.
- **No completion notice for a run that has already ended.** Jobs finishing
  after the last turn are cancelled by `CloseRun`; the resumed run learns they
  were interrupted, not what they would have produced. (A kept job's end is
  delivered through the Hold's callback.)
- **The Hold's callback is once per process.** A crash between the callback and
  the host acting on it loses the notice; the restored checkpoint then reports
  the job interrupted.
