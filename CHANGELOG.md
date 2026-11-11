# Changelog

## Unreleased

## 0.2.0 — 2026-11-11

Baseline: v0.1.9. This release hardens the harness across security, long runs, prompt caching, telemetry and built-in tools, and removes everything the native run path never reached. It is a breaking release for hosts: see **Migration**.

### Migration

- **Credentials are deny-by-default.** A `{{cred:NAME}}` placeholder resolves only under an `Env.CredentialGrants` entry naming the tool (and, with `Hosts`, only for a tool that declares the hosts it contacts). `CredentialResolver.Resolve` receives one placeholder at a time.
- **Docker network needs an allowlist.** `Network: true` with an empty `NetworkAllow` is refused; pass `sandbox.AnyHost` for open egress. An allowlist is enforced by an internal Docker network and needs a native Linux engine (it fails on Docker Desktop). `ComputerUseLimits` / `BrowserUseLimits` take the allowlist; empty is offline.
- **Tool schemas changed.** `browser_use` takes `args` (an argv array), not `command`. `write_file` overwrites an existing file only with a matching `expected_hash`. `run_shell` / `computer_use` gain `timeout_seconds` and `run_in_background`.
- **`llm.trace` payloads are opt-in.** Pass `telemetry.Limits{Payloads: N}` to `export.New` or `export.Options.Limits` to keep full request and response traces.
- **`Pi*` identifiers are renamed.** `Hooks.PiContext`/`PiContextHook` → `Hooks.Context`/`ContextHook`; `PiToolHost`/`OpenPiTools`/`PiToolOutcome` → `ToolHost`/`OpenTools`/`ToolOutcome`; `PiTurnPreparation`/`PiCompletedTurn`/`PiTurnDecision` → `TurnPreparation`/`CompletedTurn`/`TurnDecision`; `PiArgumentPreparer` → `ArgumentPreparer`; `PiQuestionFromEntry`/`PiChildQuestionFromEntry` → `QuestionFromEntry`/`ChildQuestionFromEntry`; `ToolHost` methods drop `Pi` (`StartPiRun` → `StartRun`, …); `EntryPi*` → `EntryNative*`. Persisted entry values are unchanged, so existing checkpoints load.
- **Removed (never reached by `RunNative`):** `Compactor`, `DefaultCompactor`, `CompactionRequest`, `CompactionResult`, `Registry.SetCompactor`, `CompactionUsing`, `Config.CompactionModel`, `Config.Compactor`, `CompactionSettings.MaxSummaryTokens` and `Prune*`, `ContextPruner`, `PiCompactionPolicy`, `TurnPreparation.DisableTools`, host `Transform` / `TransformWithPolicy` / `SummaryView` / `FoldSummary`, the old `ContextHook` trailer seam, `StreamInterceptor`, `StreamDecision`, `Registry.UsePolicy`, `host.ProjectMessage` / `ProjectContent` (use `ai.ProjectMessage` / `ProjectContent`), `protocol.Message.CacheAnchor` and `ContextTokenAdjustment`, `ai.EstimateTextAndImageContentTokens`, `ai.ClampThinkingLevel`, `todo.PiContextHook`, `ask.Plugin.SelfGated`, `preset.Options.Jobs` / `SessionQuery`, repeatguard `Include` / `Exclude` / `ArgumentsPreviewChars` / `DefaultExclude`, and the jobs store, launcher and option API (only `jobs.Plugin{MaxConcurrent}` remains).
- **Changed signatures.** `Registry.AddTools` returns `error` (a duplicate tool name is an error). `Config.CompactionProvider` / `CompactionPlugin.Provider` / `Registry.SetCompactionModel` take `*ai.FallbackProvider`. The third argument of `host.CompactionPolicy.Summarize` is the summary's instructions, not the revision. `SetPolicy(nil)` installs `DenyAll`.

### Security

- **A broken gate refuses the call.** A panicking before-hook or policy blocks the tool call under every `HookErrorPolicy` and is reported through `OnError`. The policy is enforced inside the executor on every call, even without a `PolicyPlugin`.
- **Credentials are scoped and redacted.** Resolved values are JSON-escaped into arguments and redacted from tool output, partial updates and errors before a hook, trace or the model sees them.
- **Permissions can ask and rewrite.** `Decision.Ask` parks the run on an `ApprovalRequest`; `Decision.Arguments` (`AllowedWith`) rewrites arguments, which are re-validated and re-checked. `Fork(id, narrow...)` narrows a child's policy, and a run started inside a tool call never exceeds its caller's policy or any ancestor's delegation depth.
- **External content is framed as data.** Results of network sandbox tools reach the model inside an `<external_content>` block. `web_fetch` refuses redirects to another host.
- **Docker egress allowlists are enforced.** The container's only route out is the filtering proxy, which binds the gateway only, allows ports 80/443 unless an entry names one, and checks CONNECT targets and absolute URLs.
- **The SSRF block list is complete.** It covers CGNAT, 198.18/15, 0/8, multicast, reserved ranges, NAT64, 6to4, Teredo and IPv4-mapped addresses.
- **Workspace file access goes through `os.Root`,** closing a symlink-swap race on writes.
- **Anthropic OAuth uses an independent random `state`** and rejects a mismatch.

### Compaction and long runs

- **A compaction summary is a structured handoff.** Nine sections after a discarded `<analysis>` scratchpad, up to 8,192 tokens, clamped on line boundaries. The summarizer reads bounded role-labelled text without thinking signatures, usage records or image bytes; a later summary extends the earlier one and keeps user corrections word for word.
- **Old tool results are cleared before a summary is paid for.** `ToolResultClearing` (on by default) replaces old results with stable placeholders; the boundary persists in `NativeState`, so the prompt cache survives.
- **Context is reattached after a summary.** `CompactionReattacher` lets extensions restore bounded context; skills read earlier and the todo plan are reattached by default, capped at `min(8192, budget/8)` tokens.
- **A host can compact on demand.** `NativeRun.Compact` takes a focus; `CompactionSettings.Instructions` sets a standing one. `CompactionSettings.SharePromptCache` (off by default, scripted-provider coverage only) asks for the summary as one more turn of the live conversation.
- **A failing summarizer stops after three consecutive failures;** a provider overflow still forces one. `Hooks.BeforeCompact` can defer a summary with `Skip`. `RunResult.Compactions` reports each compaction.
- **Mid-run checkpoints.** `NativeRun.Checkpoint` receives a resumable state after every settled turn, and a failed or cancelled run returns its last resumable `NativeState`.
- **Per-request compaction cost no longer grows with the transcript** (incremental prefix digest; `BenchmarkPrepareLongTranscript`). Oversized-result previews repeat byte for byte.
- **Inline images are charged a flat estimate** (about 1,372 tokens), not a quarter of their base64 length.
- **Job state survives a resume** through the native checkpoint; a job interrupted by the end of a run comes back as cancelled.

### Tokens and prompt cache

- **Plugins no longer rewrite the cached prefix.** The todo plan and the goal condition are persisted, append-only messages; a resume or goal revision appends a system message only when the text changed. Late system text on providers without mid-conversation system messages becomes an in-place reminder.
- **Anthropic requests place up to four rolling cache breakpoints** (tools, system, previous request end, newest stable message).
- **The wrap-up turn keeps its tools** and sends `tool_choice` none. Antigravity honors `toolChoice`.
- **Side calls do not write cache.** Compaction, advisor and memory calls send `cacheRetention: "none"`.
- **Usage carries reasoning tokens and 1-hour cache writes** (`protocol.Usage.ReasoningTokens`, `CacheWrite1hTokens`, `Usage.Add`); pricing bills 1-hour writes at their premium.
- **A run returns to its primary model.** `ai.FallbackLadder` (`NativeRun.Fallback`) probes the primary after time or successes on a fallback rung.
- **One token estimator** (`ai`, 3.5 characters per token) serves transports and compaction.

### Telemetry

- **Recorders are bounded** (`telemetry.Limits`) and count drops; payloads are opt-in, lazy and bounded.
- **Spans are queryable.** Attempt events and request spans carry OpenTelemetry `gen_ai.*` attributes, cost, context window, time to first token and fallback rung. Without payload capture a plain admission failure is `provider_admission`, not `unknown`.
- **`export.NewExporter` streams settled spans in batches without blocking,** with head sampling, heartbeats, flush on close and a standard-library OTLP/HTTP JSON sink (`export.OTLP`).
- **`llm.FileSink` writes mode 0600** and reports write failures.

### Tools and plugins

- **Shell calls are bounded.** A per-call timeout (default 120 s, maximum 600 s), head-and-tail output with the full log in `.shell_logs/`, live streaming, a background mode, and a process-tree kill on timeout or cancel.
- **`write_file` is hash-guarded and atomic,** preserving the file's mode.
- **`browser_use` execs an argv** with an allowlisted command set instead of a shell string.
- **Extension failures are always reported.** A panic at any dispatch point reaches `OnError` as `extension:<name>.<point>`; a failing `BeginRun` fails the run. `Registry.Unload` is order-independent.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.2.0`, apply the **Migration** notes, then run their compatibility checks.

## 0.1.9 — 2026-11-09

Baseline: v0.1.8. Public API gains `Registry.SetSchemaTransform`; no configuration migration and no change for hosts that do not call it.

- **A host can declare tools to the model as stubs.** `Registry.SetSchemaTransform(fn)` installs a rewrite applied to each permitted tool schema just before it is declared to the model. Execution, validation and permission keep using the tool's own schema; only the declaration changes. A host uses it to declare a tool as a stub and deliver the full definition on demand, which keeps the declared tool array stable for prompt caching. `Agent.Describe` reports `schema_transform` when set.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.1.9`, then run their compatibility checks.

## 0.1.8 — 2026-10-28

Baseline: v0.1.7. No public contract changes and no configuration migration.

- **Antigravity keeps a conversation on one session, so its prompt cache can hit.** The native payload drew a fresh `trajectory_id` for every request and ignored the `sessionId` option, so each turn looked like a new trajectory to the backend. A `sessionId` (unless `cacheRetention` is `none`) now sends `request.sessionId` and a trajectory UUID derived from it; `requestId` stays unique per request. Without a session the payload is unchanged.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.1.8`, then run their compatibility checks.

## 0.1.7 — 2026-10-28

Baseline: v0.1.6. Public contracts only gain `telemetry.WithRecordingParent`; no configuration migration is required.

- An async sub-agent (`spawn_subagent` with `async: true`) records its spans under the parent's `agentray.agent.run` span, which joins the job, instead of under the spawn call's `agentray.tool.execute` span, which ends when the call returns its receipt. A child that opened its run after that span ended was a late no-op, so its whole run (spans, attempts, advisor reviews) sometimes vanished from the run's telemetry. The safe-telemetry lineage (origin execution, parent span) is unchanged.
- `telemetry.WithRecordingParent(ctx, parent)` rebinds only the private recording parent and keeps the context's safe scope.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.1.7`, then run their compatibility checks.

## 0.1.6 — 2026-10-28

Baseline: v0.1.5. Public contracts are unchanged and no configuration migration is required.

- Native runs call the `BeforeAgentStart` and `BeforeProviderRequest` hooks, which were declared but never invoked. `BeforeAgentStart` sees the assembled system prompt, the seed messages and the task once per run and may reshape the prompt and seeds (and the prompt again after a goal revision). `BeforeProviderRequest` sees every provider attempt as a read-only projection — model, visible messages with the system prompt first, tool schemas and cache key — so a host can measure exactly what each request carries; a hook that returns a changed request is reported through `OnError` and not applied.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.1.6`, then run their compatibility checks.

## 0.1.5 — 2026-10-25

Baseline: v0.1.4. Public contracts only gain the optional `PromptCacheKey` run field; no configuration migration is required.

- Native runs put recalled memory in a user-role transcript message after the system prompt, holding only facts the visible transcript lacks. The system prompt stays byte-identical across runs, so a resumed conversation reuses the provider's prompt cache. `StartPiRunWithLifecycle` keeps memory inline.
- An explicit `PromptCacheKey` gives OpenCode Go one stable `x-opencode-session` per conversation. Before, the session was derived from the first message, so it changed whenever the system prompt changed, and a run without history got a new one every time.
- Memory consolidation stages only the finished run's own messages, not the restored transcript or the recall block.
- After compaction, the request pins `[Current request — verbatim]` only while the summary has absorbed the request. A request still verbatim in the retained tail is no longer copied between the summary and the tail, where a new task each run made the provider's cache miss the whole tail on every run.
- A saved summary that covers a whole earlier run (the next message is the new run's system message) is reapplied instead of being dropped for the full history.
- Devin: a caller session now names the prompt-cache scope, and each agent thread in it gets its own Cascade thread (the session plus its first message). Parallel subagents no longer share message IDs, and a conversation keeps one thread across runs instead of a random thread per request.
- Devin: send only the latest system prompt. Each native run appends its own system message, and Cascade joined them all, so every resumed run resent each earlier run's prompt (about 4k tokens per run) and pushed the cacheable prefix further from the cache.
- Devin: read the compressed end-of-stream trailer. Its errors (rate limit, quota) were dropped, so a refused request came back as an empty answer with no error and the fallback chain never moved on.
- Claude Sonnet 5.5 turns thinking off with `between_tools`; Claude Opus 5.5 and the Fable/Mythos models omit the field. These models answer `thinking: disabled` with a 400, which made every reasoning-off request fail.
- Fix a panic ("slice bounds out of range") when parallel subagents deduplicate recalled memory: the accent folder was a shared, stateful transformer.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.1.5`, then run their compatibility checks.

## 0.1.4 — 2026-10-17

Baseline: v0.1.3. Public contracts are unchanged and no configuration migration is required.

- Send the `system` role to OpenAI-compatible gateways; `developer` stays for api.openai.com only. Some gateways silently dropped `developer`, so the model ran without its instructions.
- Send portable tool names on Chat Completions (both the provider and the native transport) and map the model's calls back, so a gateway that rewrites a name starting with a digit no longer breaks tool calls.
- OpenCode Go requests carry `x-opencode-session` (the caller's session, or one derived from the transcript), which that endpoint requires.
- A conversation resumes on the configured primary when the model it last ran on was replaced, instead of failing before its first turn.
- Telemetry keeps the approved provider/model labels when a gateway answers a routed alias with the upstream model name.
- The first read of each enabled skill in a run costs no tool call; tool results say when two calls or fewer remain before `MaxToolCalls`.
- The goal gate hides a closing status line that carries a reason or ends a line of prose.
- A stop-guard nudge asks the model to give the complete answer again, so a closing status note no longer replaces the result.
- Memory consolidation no longer retains environment failures (unavailable tools, services, credentials) and retracts such entries it finds.
- Add a how-agents-work explainer page to the docs.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.1.4`, then run their compatibility checks.

## 0.1.3 — 2026-10-10

Baseline: v0.1.2. Existing single-scope `MemoryStore` implementations remain compatible; the new host interfaces are optional.

- Add typed memory retention with host-authorized targets, fact keys, expiry, confirmation and source provenance. Scoped hosts can support explicit replacement, full reads and erasure of retained memory revisions.
- Redact common credential-shaped strings before memory retention and consolidation model input. This does not replace the host's authorization or general personal-data policy.
- Recover queued consolidation evidence without replaying primary runs, use host-selected consolidation memories when available, and expose an error callback through the full preset.
- Add regression coverage for credential redaction and capacity-bounded recovery that drains queued evidence without polling an idle store.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.1.3`, then run their compatibility checks. Hosts opt into the new retention interfaces and remain responsible for scope, lifetime, provenance and erasure policy.

## 0.1.2 — 2026-10-10

Baseline: v0.1.1. This release preserves the existing public contracts and requires no configuration migration.

- Reduce allocations in lossless JSON parsing, transcript serialization and event-stream draining while preserving Unicode, raw numeric values and property ordering.
- Bound memory consolidation staging and model input, drain queued evidence fairly across scopes, and retry a timed-out deferred pass once while preserving durable evidence on failure.
- Add coalesced telemetry change notifications so hosts can collect records without idle polling.
- Clarify delegated-task context and host authority in subagent tool guidance.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.1.2`, then run their compatibility checks.
