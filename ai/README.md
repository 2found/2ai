# AI providers

The native agent engine imports this package directly. Requests use
`TranscriptContext`; streamed results retain thinking signatures, tool-call IDs,
usage and ordered content blocks without converting through `agentcore.ChatRequest`.

`MessageContent.Blocks` is a `*BlockList`: shallow copies retain the same list
when a provider or hook changes its entries or length. Use `Get`/`Len` to read
and `Set`/`Append`/`Delete`/`SetLength` to edit membership. `Values` returns a
detached slice of shared block pointers. `BlockReferences` creates a new list;
copy `MessageContent` to retain the existing list. Live access uses the owning
stream's synchronization; snapshots detach lists while preserving aliases.

Current integration scope:

| Provider | Native path |
| --- | --- |
| Codex | `StreamCodexResponsesPooled`, Responses SSE/WebSocket |
| Claude Code | `StreamClaudeCodePooled`, Anthropic SSE with CLI OAuth identity |
| Antigravity | `StreamAntigravityPooled`, Cloud Code SSE with per-account project |
| Devin | `StreamDevinPooled`, Connect-RPC protobuf/gzip to Cascade (`devin-agent`) |
| Gemini | `StreamOpenAICompletionsSimple`, Google OpenAI-compatible endpoint |

The host resolves endpoints and binds credentials to a provider row. OAuth pools
acquire/report per attempt and never store access tokens in transcripts. Auth
rotation stops after visible output. Host callback failures do not trigger
provider retries or account penalties. Antigravity retains the established
text/tool capabilities; parallel-call controls are explicitly unsupported.

`NativeProvider.Stream` dispatches these transports from the model's `api`.
Bind its optional `Tokens` field for account pools. `NewNativeClient` binds a
host `ClientSpec` and returns model candidates without putting credentials in
model metadata. `ScriptedStream` supplies native offline demos/test scripts. `FallbackProvider.Stream`
composes native providers under the same `StreamFn` contract consumed by the
engine. Each candidate has a bounded retry budget; fallback stops after visible
output, cancellation, or a host/protocol failure. A final message retains its
native pointer and payload synchronization through the composition.

Durable consumers use `FallbackProvider.Run` with request callbacks for fresh
candidate preparation, persisted selection, usage accounting for every settled
attempt, and terminal publication. Those callbacks contain host policy; AI owns
the retry/fallback loop. Wrap setup failures in `PreparationError` so a failing
credential/context callback cannot be mistaken for a provider outage.

Files are grouped by provider and concern within the Go package. Shared
transcript, event, HTTP and OAuth helpers have concrete data/callback contracts.
Existing legacy adapters and catalog helpers still serve public SDK, discovery
and compatibility callers, so they have not been deleted as unused.

Run `go test -race ./ai` for recorded Pi fixtures and local HTTP checks.
Provider text coverage includes 252 payload cases captured from unchanged
pinned Pi source, 21 local HTTP comparisons, and 12 Antigravity text cases
with a transport check. Claude HTTP expectations include the SDK's body
transformation. These checks preserve lone UTF-16 units in Codex instructions
and raw reasoning fields while filtering them from Pi's selected text fields.
TypeScript references and generators are removed; JSON fixtures, `LICENSE.pi`
and source hashes in `../third_party/pi/UPSTREAM.json` remain. The fixtures
establish tested behavior, not exhaustive equivalence to arbitrary JavaScript.

`FallbackSelection` is the credential-free journal record for a committed model
selection. `ParseFallbackSelection` validates transition generations and record
shape; the host also checks the selected provider/model against its configured
candidates before resuming.

Thinking-level sampling defaults follow upstream
`200387122ca450d6387f033949423114a270b96c`: model defaults are overridden by
`samplingParamsByThinkingLevel` at the clamped level, then by request values.
Both direct and simple OpenAI-compatible calls use this order; Responses uses
medium for a summary-only request. Codex and Anthropic keep these sampling
keys out of their request bodies. `testdata/pi-thinking-sampling.json` contains
14 outputs generated from upstream `resolveSamplingParams` and its level
helpers, with source paths and a source-slice hash. Existing transport oracles
remain pinned to their original revision.
