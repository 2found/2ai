# Changelog

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
