# Changelog

## Unreleased

- Native runs call the `BeforeAgentStart` and `BeforeProviderRequest` hooks, which were declared but never invoked. `BeforeAgentStart` sees the assembled system prompt, the seed messages and the task once per run and may reshape the prompt and seeds (and the prompt again after a goal revision). `BeforeProviderRequest` sees every provider attempt as a read-only projection — model, visible messages with the system prompt first, tool schemas and cache key — so a host can measure exactly what each request carries; a hook that returns a changed request is reported through `OnError` and not applied.

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
