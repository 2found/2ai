# Native Go agent engine

This package ports Pi's agent loop, Agent, proxy and stream contracts from
commit `eeac84ca92498ac18b6832754d01aef1d3c5f654`. The server uses it by default.
It directly consumes the native `ai` package and `telemetry`; it does not launch
a subprocess or import application policy.

## Upstream recheck

On 2026-10-05, upstream `main` was
`200387122ca450d6387f033949423114a270b96c`. Compared with the pinned reference,
`packages/agent` and `packages/telemetry` changed only changelogs and package
metadata (version 1.0.2); their source and tests are unchanged. The AI dependency
delta was audited: thinking-level sampling defaults and
summary-only reasoning selection are ported for the existing compatible
adapters, with a new 14-case upstream oracle. The remaining provider delta is
Bedrock thinking-block binding; Bedrock is outside the configured provider
scope and has no Go adapter here. This check does not establish complete
parity of the entire application with upstream `main`.

## Boundaries

| Module | Responsibility |
| --- | --- |
| `agentcore/engine` | Agent state, turns, streaming events, tool scheduling and validation |
| `ai` | Lossless transcript, provider requests, events, OAuth account attempts |
| `telemetry` | Request spans, records, attributes and trace snapshots |
| `agentcore/plugins` | Composable tools and host policies |
| Consuming application | Model ladders, credentials, durable sessions, admission, children and summaries |

The engine uses concrete structs and callbacks. Retry, pricing, permissions,
compaction, persistence and application tools belong to the host. Tests enforce
these dependency boundaries.

## Public behavior

`Run` and `Continue` execute the low-level loop. `AgentLoop` and
`AgentLoopContinue` expose event streams. `Agent` adds persistent state,
steering, follow-up, queue modes, hooks and explicit continuation.
`RunToolCall` applies argument preparation, validation and before/after hooks
to programmatic and nested calls as well as model-generated calls.

Transcripts preserve block ordering, thinking and tool signatures, tool IDs,
system/tool declaration changes, usage, extension fields and explicit nulls.
`MessageList` and concrete value containers retain the live identity semantics
needed by the port. Provider messages pass through `ai.TranscriptContext`;
the legacy `ChatRequest` projection is not used by the native engine.

Provider message and block references remain live across loop events. Start and
update events shallow-copy the message envelope; the final message is the
provider's original result. Event sinks run under the provider payload lock.
Use `event.Await(work)` to release that lock while waiting for provider work;
access live fields only before or after that call. Outside the sink, use the
retained event's `Synchronize` method to inspect its payload. A result wait does
not consume FIFO events.
Producer settlement is observed separately from terminal-message publication;
reader cancellation does not implicitly cancel a provider.

The host commits a selected model before admitting its tool effects. Durable
message/effect records, leases and model-selection records support resume.
Retries stop after visible output; each child owns its ladder, tools, session
and credential binding. Summaries use their own tier and no executable tools.

## Provider scope

- Codex: native Responses SSE/WebSocket and account-pool lifecycle.
- Claude Code: native Anthropic events, CLI OAuth identity and account rotation.
- Antigravity: native Cloud Code request envelope, account/project pairing,
  thinking signatures, tool calls, usage and terminal validation.
- Gemini: native OpenAI-compatible API using the configured Google endpoint.

Other existing Go adapters remain where callers still use them; expanding the
Pi provider catalog is outside this completion scope. The server always uses the native engine; the runtime-selection environment
flag and legacy execution branch have been removed. Native history is revision-checked; legacy
history requires an explicit migration or a fresh conversation.

## Verification and provenance

```sh
go test -race ./...
go vet ./...
go run ./examples/scripted
```

The ordinary Go suite includes the former `pi_native` integration contracts.
Recorded JSON fixtures cover agent/loop behavior, transcript and reference
semantics, validation, provider events, telemetry and host callbacks. HTTP
and runner tests cover credentials, tools, children, summaries, errors,
cancellation, accounting and durable recovery.

Twelve pinned-source ownership cases cover retained provider messages and edits
to timestamp, text and usage from `message_start`, `message_update`, `message_end`
and `finishTurn`. They supplement the stream-alias fixtures with cross-event
sharing and result-identity checks. Gated native tests cover yielding to a
provider from an event sink, reads of the terminal response under its payload
lock, and provider work continuing after the terminal event. Direct reads or
edits from other goroutines still require caller synchronization.

Another 48 pinned-source cases cover content-array mutation and replacement
across assistant events, finish hooks and tool-result message events.
`ai.MessageContent.Blocks` and `ToolResult.Content` use the same concrete
`*ai.BlockList`, so edits to membership or length reach retained references.
Replacing the whole content list leaves earlier references on the old list.
Provider and snapshot tests also check growth, sparse holes and shared-list
identity; callers use list methods instead of slice indexing or `append`.

Forty-five pinned-source prompt cases cover Unicode in string content and
system-section names/values, before and after mutation, through prompt replay
and state JSON. Lone UTF-16 units retain their JSON escapes instead of becoming
replacement characters. Native host tests cover prompt and initial-system-prompt
decoding through the stream callback boundary and state export.

Forty-eight pinned-source failure cases cover thrown errors/strings, provider
error messages and listener edits. They compare events, state at every event,
and final state, including empty messages and lone UTF-16 units. Failure-message
construction and state export retain the same code units as the transcript.

Twelve pinned-source tool runs cover Unicode names, call IDs and labels through
declaration removal, execution updates, pending-call sets and final state.
Removal references retain their extension metadata and decoded names, so replay
deletes the named tool without changing its identity during JSON import/export.

Eight pinned-source metadata runs cover distinct UTF-16 property names across
messages, blocks, tool declarations/results and usage/cost extensions. They
compare provider inputs, events and final state after listener edits. Typed
envelopes use the shared raw-object codec so opaque keys survive without
decoding their values; duplicate keys retain the last value.

The TypeScript source, worker, fixture generators and Bun configuration have
been removed. `third_party/pi/UPSTREAM.json` preserves hashes of the 412
original files verified before removal. `LICENSE.pi` retains attribution.
Fixture agreement does not establish universal byte parity: arbitrary
JavaScript objects/prototypes, platform stacks and every malformed transport
input are not equivalent to Go values or diagnostics. The port preserves the
recorded contracts and supported native execution paths.
