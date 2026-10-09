# agentcore

Native runs support the [safe producer observer](../telemetry/SAFE.md) attached
to the execution context. Host-approved model bindings belong to AI candidates;
`NativeRun.SafeToolLabels` binds public tool IDs. Main, compaction, advisor,
memory and child calls preserve their categories. Detached child/deferred
memory work uses independent safe execution IDs with known or explicitly
unknown/coalesced origins. Parent usage is comparison-only reconciliation,
never added again to child/attempt totals. Custom aggregate-only callbacks have
unavailable physical-attempt coverage.

The server uses the native Go engine by default. Execution lives in
[`engine/`](engine/), provider transcripts and transports in [`ai`](../ai/),
and recording in [`telemetry`](../telemetry/). Application composition,
credentials and persistence adapters stay in the consuming application.

The supported completion scope is Codex, Antigravity, Gemini and Claude Code.
TypeScript source, worker transport, fixture generators and Bun build tooling
have been removed. Go tests consume recorded JSON fixtures; source hashes and
licenses remain in [`third_party/pi`](../third_party/pi/README.md).

The root owns host composition, governed tool dispatch and plugins. All run
entry points execute the native engine. Provider contracts live in
`ai/protocol` so the facade can depend on the engine without an import cycle.
Package boundaries follow ownership and dependency direction; no additional
interface hierarchy is needed.


## Native migration and consumer compatibility

The server and Soot use the native engine. The old root loop, configurable
`Driver`, server legacy dispatch and runtime-selection flag are removed.
`Prompt`, `PromptStream`, `Continue` and `ContinueStream` now require
`Config.NativeProvider`. `Continue` accepts newly authored host input; resume
provider history through `RunNative` and `RunResult.NativeState`. Durable
execution uses the native session host. Recorded native fixtures and the integration suite run with ordinary Go tests.
Native integration tests cover memory, compaction, todo and advisor. Native retry/fallback orchestration now lives in
`ai.FallbackProvider`. The runtime binds durable selection and accounting to
its callbacks. `ai.NativeProvider` owns native transport dispatch, and
`engine.StreamFn` aliases the shared `ai.StreamFn` contract.

`engine` owns native scheduling and provider events. The root still supplies
composition, governed tools and plugins. A new
engine does not itself replace the host's context compaction or session policy.

The process bridge (`NewPi`, `PiAgent`) and its root-level configuration types
(`PiConfig`, `PiCallback`, `PiError`) have been removed. Native consumers use
`engine.AgentOptions`/`engine.NewAgent`; the application's JSON callback binding
is owned by the consuming host configuration. Sessions now hold the concrete
native agent, without the temporary worker/native function table or production
string-command dispatcher. Recorded worker commands are decoded only by tests.

The earlier port also removed `LabStepDiff`, `DiffStep`, and the observe helpers
`AggregateRunSummaries`/`AggregateRunCoverage`. Existing `Plugin`, `Tool`,
`LLMProvider`, `SessionStore` and extension interface method sets are unchanged;
no plugin package has been removed. Provider contracts are now owned by
`ai/protocol` and re-exported by the root package while callers migrate. Native integrations use `Hooks.PiContext`
for native transcripts and `subagent.Settings.RunFork` for native child runs.

The sibling Soot consumer uses `New` with `Config.NativeProvider` and
`RunNative` for both case work and conversations. Its provider factory builds
`ai.NativeClient` candidates under `ai.FallbackProvider`; the store atomically
persists `RunResult.NativeState` with the public answer. Legacy context arrays
are imported once as untrusted source facts, never replayed as provider history.

`RunNative` accepts an opaque prior checkpoint and newly authored user/system
input. It composes the existing plugins and governed tools around `engine.Agent`,
with native compaction and explicit-parent telemetry. Display `Messages` are
output only. Durable sessions use the application's native session host for
in-flight effects, leases and parked delegations. Ephemeral native subagents
continue corrections from `NativeState`, preserving their original transcript.

Shared native compaction and transcript-checkpoint utilities live in
[`host/`](host/); application persistence and credentials remain in the consumer.
The same package owns host-input conversion and display projections. Native
request observers receive the final transformed provider view; successful
compaction emits a rebase before the request observation, and settled messages
emit append observations. This keeps advisor and other existing `RunObserver`
plugins informed without giving them control over native history. Integration
tests exercise memory, todo, advisor and compaction through the native engine.

Native context summary progress reports each chunk's settlement, including a
failed chunk. It does not claim that the entire compaction succeeded or that the
next model request is ready. Chunk and model-request spans have individual
timings in `telemetry/export.Batch.SpanTimings`. Async subagent display progress
uses the run-owned observer in `background.go`, fenced on native run teardown.

The existing kernel is one flat package. The native Go replacement lives in
[`engine/`](engine/); the other subdirectories are ejectable
[`plugins/`](plugins/) and black-box [`integration/`](integration/) tests.
The existing root package has a hard dependency rule in both directions:

> **Composition names no plugin and imports only shared runtime modules.**
> Provider contracts live in `ai/protocol`, below the agent facade; the native
> loop lives in `engine`, and reusable host policy lives in `host`. Delete every package under `plugins/` and this package still
> compiles, still runs, and still passes its tests — it just does less.

These boundaries are tested: [`boundary_test.go`](boundary_test.go) reads
the package's own imports and fails on a plugin or application import. That is what makes the kernel publishable on its own and
what makes [`plugins/README.md`](plugins/README.md)'s ejectability claim true.

Everything structural on this page is enforced the same way, because a rule that
only this file knows is a rule that drifts:

| test | holds |
|---|---|
| `TestKernelNamesNoPlugin` | no `plugins/` import from the root package |
| `TestKernelRuntimeDependencies` | only AI, engine, host, telemetry and shared JSON dependencies |
| `TestKernelTreeHoldsOnlyDeclaredBoundaries` | only the declared `engine/`, `host/`, `plugins/`, and `integration/` boundaries sit below the root |
| `TestNativeEngineNamesNoHost` | the native engine directly imports only AI/telemetry modules, the shared JSON value codec, and libraries; it cannot launch a subprocess or import host policy |
| `TestEveryKernelFileJustifiesItself` | every root `.go` file has a row below |
| `TestPluginsDoNotNameEachOther` | no plugin imports a sibling (except `preset`) |
| `TestEveryPluginDocumentsItself` | every plugin folder has a `README.md` |

The API-side version of this page — the boundary rules, the three kinds of
contribution, and the layer map — is [`doc.go`](doc.go), so a reader who arrives
through `go doc` gets it too.

## Why the root is flat

The obvious tidy-up — `agentcore/session/`, `agentcore/context/`,
`agentcore/tools/` — is the wrong move here, because these files are not
independent concerns that happen to sit together. They are one machine:
the loop reads the limits, writes the log, dispatches the extensions, brackets
the compaction, and stamps the idempotency key, all through unexported fields
of one `Agent`. Splitting them into packages would either export that machinery
(making it everyone's to break) or produce packages that can only be imported in
one order — folders pretending to be boundaries.

Flat is not the same as unordered, though, and the distinction is where this
package earns the choice. The layering is real — composition sits above the
contracts, the contracts above the loop, the loop above the log — it is just
carried by *file* rather than by folder, and by a rule about which direction
job: `tooldispatch.go` is the trust boundary, `turn.go` is one turn against the
model, `compose.go` is how an agent gets built. A file earns a split only when
it starts needing "and" to describe a *boundary*, not a subject — the loop's
own retry, schema validation, and idempotency stamps live inside the files that
use them rather than beside them. [`doc.go`](doc.go) lists the layers in order.

The real boundary is `plugins/`, and it is enforced above. So the organizing
question for the root is not "which folder?" but:

> **Why is this file in the kernel and not a plugin?**

There are exactly four answers, and every file below gives one. A file that
cannot give one belongs in a plugin, or belongs nowhere.

| answer | means |
|---|---|
| **loop** | the loop calls it directly on every run; there is no composition in which it is absent |
| **contract** | a type the loop dispatches over, so a plugin outside this repo can implement it |
| **log** | it reads or writes durable run state, and only the log's owner may |
| **seam default** | the built-in behind a replaceable seam, so an agent composed with nothing still works |

## The files

### Composition — how an agent gets built

| file | why core |
|---|---|
| [`doc.go`](doc.go) | **contract** — the package doc: the two boundary rules, the three kinds of plugin contribution, and the layer map below rendered where `go doc` can see it. No code. |
| [`provider.go`](provider.go) | **contract** — `LLMProvider`, `ChatRequest/ChatResponse`, `Usage`, and provider-neutral text/image `ContentPart`s. The wire seam every model call goes through, kept small enough that an implementation is a translation layer and nothing more. |
| [`provider_session.go`](provider_session.go) | **contract + seam default** — provider-private conversation state with selective account-rotation reset, plus the bounded lease-aware in-process registry. The loop carries the session on every request; providers own the concrete records, and a cache miss may cost discovery but never change correctness. |
| [`plugin.go`](plugin.go) | **loop** — `Plugin`, `Registry`, `Priority`. The composition surface itself: seam setters, additive contributions, per-plugin `Unload`. |
| [`compose.go`](compose.go) | **loop** — `Build`, `BuildRegistry`, `ApplyConfig`, and `Limits`/`DefaultLimits`: the run's bounds are chosen at composition, read every turn, and published to extensions through `RunInfo`. There is no composition in which a run is unbounded. |
| [`agent.go`](agent.go) | **loop** — the configured runtime instance (capability-bearing fields unexported on purpose, see `fork.go`) plus `Agent.Describe()`: what the agent is *actually* configured with after every default and override. |
| [`definition.go`](definition.go) | **contract** — `AgentDefinition`, `Skill`, `SkillLoader`, and the always-loaded byte cap that keeps the system prompt bounded. |
| [`seams.go`](seams.go) | **seam default** — `ConfigPlugin`, `ModelPlugin`, `DefinitionPlugin`, `PolicyPlugin`, `ToolsPlugin`, `HooksPlugin`, `BudgetPlugin`, `SessionPlugin`, `CompactionPlugin`, `SteeringPlugin`. Core adapters that claim seams during `Build(...)` without needing fake wrapper packages under `plugins/`. |

### The loop and its extension points

| file | why core |
|---|---|
| `run_policy.go` | Shared native-run ceiling messages and bookkeeping classification. |
| [`turn.go`](turn.go) | **loop** — one turn against the model: capability shaping (including copy-on-write request-wide image budgets), `ProviderError` classification, same-rung retry, then escalation down the ladder, plus the streaming path. Retry lives with the loop, not in a provider, so failure behaviour cannot differ per vendor. That includes reading usage off the stream: a delta's `Usage` is a running total that may arrive at any point (Anthropic states input tokens before the first output token; OpenAI sends a usage-only chunk *after* the terminal one), so the turn keeps the newest non-zero value of each field rather than whatever rode `Done`. Getting it wrong is silent — the answer is still correct and only the number the budget gate meters on is zero. |
| [`tooldispatch.go`](tooldispatch.go) | **loop** — one tool call end to end: lookup → prepare → validate → gate → execute → bound → trace. The trust boundary, applied in exactly one place so it is unskippable rather than usually-called. Also the two context stamps a call carries: the idempotency key derived from `(sessionID, toolCallID)` — stable across crash-resume because both already survive in the log — and the provider-assigned tool-call id a spawn tool derives the child's deterministic session from. |
| [`result.go`](result.go) | **contract** — `RunResult`, `StreamEvent` and the event vocabulary, `ResultCard`. The loop's output side, which consumers render and plugins observe. (`ToolTrace` sits with the code that fills it, in `tooldispatch.go`.) |
| [`background.go`](background.go) | **contract** — run-owned background launcher and progress observer shared by independent plugins; no scheduler or task policy in core. |
| [`extension.go`](extension.go) | **contract** — every extension point (`ToolInterceptor`, `StepInterceptor`, `StopInterceptor`, `RunObserver`, `ToolContributor`, …) and the `extensionSet` the loop dispatches through. The file that forbids naming a plugin. |
| [`run_lifecycle.go`](run_lifecycle.go) | **contract + dispatch** — optional settled-boundary controls, host command handlers and finalization before native checkpoints; no capability-specific state. |
| [`hooks.go`](hooks.go) | **contract** — the lifecycle hook types and their dispatch, including the `BeforeToolCall` shape the permission gate is built from. |
| [`tool.go`](tool.go) | **contract** — `Tool`, optional additive `RichTool`, `ToolSet`, `ArgPreparer`, and the loop's own byte bounding. Ordinary text stays compatible while image parts can cross the neutral provider seam. |
| [`toolbridge.go`](toolbridge.go) | **contract + loop** — the run-owned `ToolInvoker` capability for tools such as eval that need to call another registered tool. Nested calls re-enter the single dispatch trust boundary (schema, policy, credentials, hooks, bounds, tracing, idempotency), share the run's atomic execution budget, reject recursion, and persist as audit metadata without inventing provider-authored tool messages. Directly constructing a tool never grants this capability. |
| [`pi_tools.go`](pi_tools.go) | **host adapter** — exposes a composed Agent's tools to the native Go engine through the existing dispatch boundary. Owns extension resources and the busy slot, supplies native tool definitions/results, and shares the execution budget with nested calls. |
| `native_session.go` | Restores the native transcript from journal entries, retaining opaque messages and rejecting unsettled effects. Consumer workflow validation remains separate. |
| `native_run.go` | Composed plugins and consumer checkpoints over the native engine; AI-owned retry/fallback. |
| `pi_lifecycle.go` | Native-loop host boundary for composed prompt, step, batch, stop, and ceiling policies. Calls extension contracts while Pi owns scheduling and native history. |
| [`image.go`](image.go) | **loop** — the central rich-image trust boundary: decode validation, MIME correction, pixel/input limits, provider-portable resize/recompression, aggregate byte budgeting, and coordinate mapping. Keeping it beside tool dispatch gives every rich tool identical laptop/server behavior without trusting each implementation to normalize correctly. |
| [`permission.go`](permission.go) | **contract + seam default** — `Policy`, `Decision`, and `DenyAll`. Default-deny is the kernel's, so a composition that forgets governance is not ungoverned. |
| [`prompt.go`](prompt.go) | **loop** — system-prompt assembly, recall dedup, and provider-neutral prompt-cache breakpoints placed at the end of the request's *append-only* prefix (the persisted history), not on its final message. A `ContextHook` trailer is re-rendered every turn, so a prefix ending in one can never be read back: measured on a 300-turn run, **7 of 299** cache entries were still a prefix of the next request; anchoring before the trailer makes it **277 of 299** (the rest are the 22 compactions, which legitimately rewrite the window). |
| [`skill_tool.go`](skill_tool.go) | **loop** — the `read_skill` built-in, registered by the loop whenever a definition carries skills. Progressive disclosure is part of the prompt, not an add-on. |
| [`schema.go`](schema.go) | **loop** — compiles `OutputSchema` at composition time and validates final text locally, so providers that ignore structured-output hints cannot silently violate the result contract. |

### Durable state — only the log's owner may touch these

| file | why core |
|---|---|
| [`goal_revision.go`](goal_revision.go) | **contract + log seam** — a consumer-authored completion-condition change and its context-scoped commit boundary. Native tools commit before the plugin publishes the condition; goal policy, validation, and the revision tool stay in `plugins/goal`. |
| [`session.go`](session.go) | **log** — `SessionEntry` kinds, `SessionStore`, reduce/recover, side records (`EntryInbox`/`EntryInboxDone`, `EntryAssistantFrame`, `EntryToolProgress`, `EntryToolOutcome`), idempotent parked-call answers, optional `SessionLeaseStore` ownership, and the goal as a *fact about the run* (written once, recovered on resume; what to DO about an unmet goal is [`plugins/goal`](plugins/goal/)). The log is the source of truth; run state is rebuilt by reducing it, never mutated in place. Chain entries form the session tree; side records capture intent and mid-turn granularity without forking the chain. A completed outcome is journaled before slower siblings finish, then superseded by its canonical source-order result. `SessionBatchStore` upgrades a turn save point from ordered-prefix compatibility to an atomic commit. The windowed read (`SessionWindowStore`, `LoadResumeLog`) lets the fold restart at a checkpoint, so a resume reads a suffix rather than a history — and the rule for when that is safe lives here, once, rather than in each backend. |
| [`session_tree.go`](session_tree.go) | **log** — the log is a tree: parent ids, branches, `EntryLeafMove`, `Rewind`. Reduce and recover walk only the active branch. |
| [`memsession.go`](memsession.go) | **seam default** — in-process append-only `SessionStore` with atomic save-point batches and per-session resume ownership, so a laptop run is resumable and the log/concurrency invariants are checkable with nothing wired. |
| [`compaction.go`](compaction.go) | **log + seam default** — WHEN to compact and the durable bracket around it, plus the goal pin that survives summarization, and `Compactor`/`DefaultCompactor`: WHAT replaces the old span, as a strategy with more than one right answer. The trigger is `min(model window − output headroom, configured ceiling)`, re-derived per turn from the answering rung: the ladder routinely mixes models whose windows differ by 30x, and a ceiling too high for the current one means the loop never compacts before the provider rejects the request. The kernel knows no model's window — the rung carries it. |
| [`contextprune.go`](contextprune.go) | **contract + seam default** — `ContextPruner` is the optional deterministic first-stage contract; the built-in implementation batch-prunes superseded, stale, and bulky old tool results before paying for an LLM summary. It protects deep warm-cache prefixes, requires meaningful aggregate savings for age-only rewrites, and preserves recoverable artifact references. A replacement compactor brings its own policy; the loop still owns the durable bracket around every rewrite. |
| [`fork.go`](fork.go) | **loop** — everything that crosses an agent boundary: building a child from a parent's *unexported* fields (a delegation plugin cannot do this from outside without those fields becoming exported, which is exactly how a child ends up out-scoping its parent), delegation depth (a spawn plugin's recursion cap is only enforceable if the depth survives the hop), and the run's session id (a sub-agent shares its parent's provider and ctx, so without this tag the two are one undifferentiated stream to anything decorating the seam). All three ride ctx — the only thread that survives the hop. |
| [`lab.go`](lab.go) | **log** — the read model: one pure fold from recorded facts to ordered steps, so a live-stepped run and a replayed run read identically — and the same fold one level up as chapters: a run's compaction summaries ARE its table of contents, and dividing the fold at them is what makes a several-thousand-step run navigable rather than merely paginated. |

### Host capabilities — injected, never reached for

| file | why core |
|---|---|
| [`env.go`](env.go) | **execution contract** — host-owned tool infrastructure: `Env`, the `Sandbox`, optional `SessionSandbox` and `ProcessSandbox` capabilities (persistent eval and LSP), session-id plumbing, and `CredentialResolver`. These are not agent plugins. The host binds sandbox backends to tools; the executor resolves credentials after gating. `Env.Sandbox` is diagnostic metadata, not an isolation switch. |
| [`memory.go`](memory.go) | **contract** — `MemoryEntry`, `MemoryStore`, `Embedder`, `Cosine`: cross-run recall as a seam the consumer backs. A nil store is valid; ranking degrades to keyword recall rather than failing. |
| [`faux.go`](faux.go) | **seam default** — the providers that make the loop testable with no network and no key: `FauxProvider` (scripted) and `ReplayProvider` (plays a recorded `[]TurnRecord` and asserts the loop rebuilt the same request, so a transcript is a regression test). omp has no record/replay; this is agentray's. |

## Tests in the root

Root tests exercise **loop-owned** behaviour, not plugin policy. Two that look
like plugin tests and are not, because the state is the kernel's:

- `budget_test.go` — the graceful-stop protocol (one tool-free wrap-up turn,
  then `budget_exhausted`) is the loop's; `BudgetPlugin` in `seams.go`
  only supplies the ceiling.
- `session_test.go`, `compaction_test.go` — the goal's persistence and its
  survival through compaction are the log's;
  [`plugins/goal`](plugins/goal/) owns the completion protocol and is tested
  there.

Black-box tests that import plugins now live in [`integration/`](integration/).
They prove capabilities against the real public loop without mixing consumer
composition into the kernel's package-private test suite. A plugin-policy test
still belongs next to the plugin itself.

### Shared telemetry and native extensions

Supply `NativeRun.Telemetry` or carry an explicit parent with
`telemetry.WithContext`. `telemetry/export` delivers completed span batches;
`telemetry/llm` supplies model trace records and DB/file sink contracts. Model
pricing is in `ai`. The former `observe` plugin, `WrapProvider` and raw provider
response hook are removed. Native checkpoint validation replaces the old
typed-journal invariant observer.

Go plugins use `ExtensionFactory.BeginRun` around the engine's hooks.
`NativeContextContributor` adds a run-scoped request transform without changing
authoritative history. Todo uses it with its per-run tool and checkpoint;
memory contributes explicit `memory_recall`. Pi's TypeScript `ExtensionAPI`
loader belongs to its coding-agent application, not to the ported agent engine.
