# jobs

**Extension.** Ejectable — the loop never names it. Without it every tool is
synchronous: a long crawl, build, render, or export blocks the run until it
finishes, and a run that has to wait cannot also think.

## Model Experience

### A tool goes asynchronous

The tool calls `jobs.From(ctx).Start(...)` and returns immediately. What the
model sees in the tool result is whatever that tool chose to say — this plugin
does not dictate it; the convention is to return the job id.

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

**Capped.** Inline output is bounded at 2 KB per job (`jobNoticeInlineLimit`),
and a job's stored result is bounded at `MaxResultBytes` (default: the run's
`MaxToolResultLen`) at completion time, before it can be read at all.

#### KV cache effect

**Append-only.** The notice is appended after the previous turn's messages and
never rewritten.

### The model inspects or stops work

#### What the model sees

Four tools: `job_list`, `job_status`, `job_wait`, `job_cancel`. `job_wait`
blocks for at most `MaxWaitSeconds` (default 120s) and then returns the job's
current state, so the model cannot park the run forever on work that never
finishes.

#### Token effect

**Capped**, by the same `MaxResultBytes` bound applied at completion.

#### KV cache effect

**Append-only.**

## Impact on the agent

- Adds four tools that bypass the permission gate. They are `SelfGated`: every
  one is scoped to `RunInfo.Owner`, so they observe and cancel only work **this
  run** launched — a capability the agent already exercised by starting the job.
- Binds a `Launcher` onto the run context (`RunContext`). A tool discovers the
  capability with `jobs.From(ctx)` and the `ok` result **is** the plugin's
  presence, so a job-aware tool degrades to synchronous when the plugin is
  ejected rather than failing.
- The job's context derives from the **run's**, not the calling tool's, so work
  survives the call that started it and still dies with the run.
- `CloseRun` cancels everything this run started. A job outliving its run would
  be unobservable and uncancellable — nothing left to look at it.

## Known limitations and deferred work

- **`LocalJobStore` dies with the process.** A restart loses running jobs and
  their results; a durable store is the consumer's to supply.
- **The owner fence is not authenticated.** It is a map key, not a capability
  token. A custom `JobStore` that ignores `owner` silently removes the fence,
  and nothing detects that.
- **No completion notice for a run that has already ended.** Jobs finishing
  after the last turn are cancelled by `CloseRun`, so late-completing work is
  lost rather than reported on the next run of the same session.
- **`DrainCompleted` / `CancelAll` are discovered by type assertion**, not part
  of `JobStore`. A store that omits them loses notices and run-scoped
  cancellation with no build error.
