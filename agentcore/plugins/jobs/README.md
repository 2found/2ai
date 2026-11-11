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

- **Running work does not survive the run.** A goroutine cannot be
  checkpointed. The checkpoint is taken before `CloseRun` cancels the work, so a
  job still running then is recorded as `cancelled` with the error
  "interrupted: the run ended before this job finished; start it again if it is
  still needed", and the resumed run's first step announces it. Durable host
  tasks belong on the consumer's own task queue.
- **No completion notice for a run that has already ended.** Jobs finishing
  after the last turn are cancelled by `CloseRun`; the resumed run learns they
  were interrupted, not what they would have produced.
