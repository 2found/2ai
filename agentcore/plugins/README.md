# agentcore plugins

One folder per capability. Each folder has a `README.md` explaining what it
does to the agent — what the model sees, what it costs in tokens, what it does
to the provider's prefix cache, and what it cannot do.

## The rule

> **The loop names no plugin.** Delete every package in this directory and
> `agentcore` still compiles and runs — it just does less.

Everything below the loop reaches it through the generic interfaces in
[`agentcore/extension.go`](../extension.go). The loop dispatches to
`ToolInterceptor`, `BatchInterceptor`, `StepInterceptor`, `StopInterceptor`,
`RunObserver`, `ToolContributor`, `PromptContributor`,
`ContextContributor`, `NativeContextContributor`, `NativeStateContributor`, `RunController`, `RunCommandHandler`, `RunFinalizer`, `RunCloser`, `SelfGated`, `BookkeepingTool` — never to
`spill`, `jobs`, `todo`, or `subagent`.

Plugins do not name each other either. Where one capability needs to know
something about another — the repeat guard must not count a plan tool's
legitimate repeats — the question is asked generically (`RunInfo.Bookkeeping`),
never by importing the other package.

Adding a capability is **a new folder**. It is never an edit to `agentcore`, and
never an edit to another plugin.

Both halves are enforced in [`../boundary_test.go`](../boundary_test.go), and the
second half is the one that needs it. Go's cycle check already makes the loop
importing a plugin a build error. Nothing stops a plugin importing a *sibling* —
they are peers, so it compiles cleanly — which makes it the cheapest rule here to
break and the most expensive to unwind: the day the repeat guard imports `todo`
to ask whether a call was bookkeeping, ejecting `todo` stops being a composition
change and becomes a build error. `TestPluginsDoNotNameEachOther` fails on that
import; `preset` is exempt because it is the aggregator, not a capability.
`TestEveryPluginDocumentsItself` holds the per-folder `README.md` promise above.

## Three kinds of plugin

They are not the same thing, and calling all three "plugin" without saying which
is how a plugin system quietly becomes a monolith with folders. The kind is
decided by which registry call the plugin makes, so you can read it off the
`Register` body in one line.

### Seams — core loop configuration

A seam has **exactly one provider**. Core seams (`model`, `definition`, `session`,
`compaction`, `budget`, `steering`) and their adapters (`tools`, `hooks`, `policy`)
live in `agentcore` itself (`agentcore/seams.go` and `agentcore.ConfigPlugin`),
not in this directory — they are tightly coupled to the kernel's unexported setters
and have no independent logic.

The capabilities below are real, ejectable plugins:

| Folder | Kind | What it does | Ejecting it means |
|---|---|---|---|
| [`memory`](memory/) | Seam + tools | Cross-run recall + curation (`memory_recall`, `learn`, `memory_edit`) | Forgets between runs |
| [`goal`](goal/) | Hybrid | Completion contract (`STATUS: DONE`) + `update_goal` tool | Runs stop when they like |

The goal hybrid installs a recorded decision and its enforcement together:

- [`goal`](goal/) claims the goal seam (the loop persists the condition to the durable
  log and recovers it on resume) **and** adds the gate as an extension. The state
  is core's because only the loop may write the log; the policy — contract,
  sentinel, nudge, stall breaker — is entirely in the plugin.

Sandbox and credentials are tool execution infrastructure, not agent plugins.
The host binds a sandbox backend to the tools that use it and installs any
argument guard through `Hooks.Before`. `Config.Env.Credentials` supplies the
executor's secret resolver; `runToolCall` invokes it after permission checks and
before execution, retaining placeholders in the argument trace. `Env.Sandbox`
only records the backend for diagnostics; setting it does not isolate a tool.
There are no separate sandbox/credentials registry seams.

Tracing uses the shared `telemetry` context supplied to `RunNative` (or carried
with `telemetry.WithContext`). It is not an ejectable agent capability.
`telemetry/export` delivers completed span batches; `telemetry/llm` adapts native
request traces to storage/file records. Pricing belongs to `ai`.

### Contributions — add to the composition

`r.AddTools(...)` / `r.AddHooks(...)`. Additive,
unkeyed, and resolved **once at compose time** — they carry no per-run state, so
they need no lifecycle. This is the cheapest kind of plugin, and the right one
whenever a capability is fully expressed by "here is a tool", "here is a
listener".

| Folder | Contributes | Ejecting it means |
|---|---|---|
| [`ask`](ask/) | `ask` tool for mid-run human decisions | the agent guesses instead of asking |

### Extensions — add something the loop does not do

`r.AddExtension(...)`. Additive and **unkeyed**: several may intercept the same
point, and they compose in registration order. Unlike a contribution an
extension is **run-scoped** — the loop instantiates it per run via `BeginRun`,
which is what lets it hold state (a repeat chain, a nudge budget) with no
cross-run leakage, and lets it decline a run outright. State shared by parallel
tools still needs synchronization. A
capability is not a slot — two interceptors bounding a tool result is a
waterfall, not a conflict.

These are the ejectable ones. The loop holds them behind
`[]agentcore.ExtensionFactory` and discovers what each can do by **type
assertion** at run start, so the set of things a plugin may do is open and adding
a new kind does not touch core.

| Folder | Adds | Ejecting it means |
|---|---|---|
| [`todo`](todo/) | run-scoped `update_plan`, pinned context and checkpoint | a long run drifts off task |
| [`advisor`](advisor/) | pre-finish reviewer | the agent never gets a second opinion |
| [`finishguard`](finishguard/) | verify-on-stop | the first answer is the answer |
| [`jobs`](jobs/) | async tools + `job_*` | every tool blocks the run |
| [`repeatguard`](repeatguard/) | loop-detection reminder | a repeat loop burns the turn budget |
| [`sessionquery`](sessionquery/) | `session_query` over the log | compaction is lossy in practice |
| [`spill`](spill/) | lossless bounding + `read_spill` | oversized results are truncated for good |
| [`subagent`](subagent/) | `spawn_subagent` | the agent is solo |
## preset

[`preset`](preset/) is none of the above: it composes the rest back into
agentcore's default agent. `preset.Plugins(cfg)` is pinned to
`agentcore.New(cfg)` parity;
`preset.Full(cfg, opts)` is that list plus the capabilities `Config` has no field
for — `spill`, `jobs`, `repeatguard`, `sessionquery`, and opt-in todo, subagent, advisor, ask and finish guards — and is what a deployment actually composes.

## Five things the mechanism guarantees

**Replaceability.** A seam has one provider and the plugin that registers it says
so. `Registry.Describe()` prints which plugin owns which seam; `Agent.Describe()`
prints what an agent actually ended up configured with — for "what is *actually*
running?".

**Reversibility.** Registrations are effects owned by the plugin that made them,
so `Registry.Unload(name)` unwinds one completely: seam released, hooks
withdrawn, extension removed, previous provider restored.

**Ordering is explicit, not positional.** Hooks carry a `Priority`
(`PriorityGate` < `PriorityDefault` < `PriorityLate`), so the permission gate is
consulted before any consumer hook regardless of where `policy` sits in the list.
The governance guarantee is a property of the composition, not a side effect of
construction order.

**The core owns persistence.** An extension returns `AdditionalContexts`; the
**loop** appends them (after every tool result in the batch, never interleaved —
that would break tool-call/result adjacency) and writes them to the durable log.
No plugin touches `appendEntry`. That is what makes "model-visible means logged"
structurally true rather than a convention every new injector has to remember.

**One interface, uniformly.** Every plugin in every table above is a value with
the same two methods (`Name`, `Register`); every extension adds the same
`BeginRun`. `preset_test.go`'s `TestEveryPluginSharesOneInterface` holds them all
in one `[]agentcore.Plugin` and drives each through `BeginRun`, so a capability
that ever needed a bespoke entry point could not be listed there.
`TestSeamsAreNotExtensions` guards the reverse — a seam plugin must not also
pretend to be ejectable.

## Declining a run is not an error

`BeginRun` returning a nil `Extension` means "not this run" — delegation already
at max depth, no store to spill into, no durable log to query. The tool is then
never advertised, which is deliberate: a tool that is offered and can only ever
refuse is something the model must read, reason about, and work around.

## Writing one

```go
package mycap

type Plugin struct{ /* config */ }

func (Plugin) Name() string { return "my_cap" }

func (p Plugin) Register(r *agentcore.Registry) error { r.AddExtension(p); return nil }

func (p Plugin) BeginRun(ctx context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	if !p.canServe(info) {
		return nil, nil // decline; not an error
	}
	return &capRun{ /* state owned by this run */ }, nil
}
```

Use the existing engine bridge; Pi's executable extension loader is in its
`coding-agent` application. AgentCore's `Register` declares a capability and
`BeginRun` binds it to one execution:

| Need | Extension interface |
| --- | --- |
| Model-callable operations | `ToolContributor`; the host separately grants calls |
| Stable system instructions | `PromptContributor` |
| Live reminders after compaction | `NativeContextContributor`; transform detached native JSON, never replay display messages |
| Resume state | `NativeStateContributor`; core checkpoints it, the host commits it |
| Tool/turn/finish policy | `ToolInterceptor`, `BatchInterceptor`, `StepInterceptor`, `StopInterceptor` |
| Run-owned resources | `ContextContributor` and `RunCloser` |

Keep provider fallback in `ai` and tracing in `telemetry`. Soot supplies stores,
credentials, tool callbacks and orchestration; plugins own the agent behavior.
Verify a stateful capability through `RunNative`, including checkpoint resume
and child isolation, rather than only testing its registration.

Then implement whichever optional interfaces apply. Write the README in the
format the others use — model experience, token effect, KV cache effect, known
limitations — because the honest section is the last one.

## What is deliberately not copied from deepseek-harness / Cordis

The dependency-injection framework. No reflection, no service container, no
lifecycle graph. A plugin is a value with a `Register` method, a seam is a typed
field, an extension point is a Go interface, and composition is a function call.
