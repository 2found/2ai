# AGENTS.md

**2ai** is the shared Go AI foundation for 2found. All paths and commands are
relative to this repository. Read [README.md](README.md) or
[README.vi.md](README.vi.md), then [BRANDING.md](BRANDING.md).

## Layout

Four layer directories. `test/layering` enforces the rule on production imports
and fails on a Go package directory outside them and `examples/`.

- `core/` imports only `core/`.
- `feature/` imports `core/` and itself; one feature never imports another.
- `compose/` imports `core/` and `feature/`.
- `test/` holds fixtures and black-box suites; production code never imports it.

## Ownership

- `core/ai/` and `core/ai/protocol/`: provider transports, transcripts, OAuth, retry/fallback.
- `core/agentcore/engine/`: native model loop, streams and tool scheduling.
- `core/agentcore/host/`: reusable host checkpoint and policy utilities.
- `core/agentcore/`: composition and governed tool dispatch; `feature/plugins/` extends it.
- `core/telemetry/`: spans and export; `feature/telemetry/llm/`: LLM records.
- `feature/credential/`, `feature/sandbox/`: host-bound vault and execution tools.
- `core/jsonjs/`: shared lossless JSON values.
- `compose/preset/`: the default plugin set.
- `test/agentcore/`, `test/telemetry/`: fixtures and conformance cases for consumers; `test/integration/`: black-box suite.

Read the nearest package README and boundary tests before changing its contract.
AI must not import AgentCore in production. The engine must not import plugins,
application policy or launch a runtime subprocess. Plugins do not import each
other; `compose/preset` assembles them. Applications own credentials,
grants, durable stores, pricing and tenant identity. Do not add product routes,
SQL stores or a competing agent loop here. A workspace is not an OS sandbox.

## Verify

```sh
go test -race ./...
go vet ./...
go run ./examples/scripted
```

Tests use recorded fixtures, scripted providers and local HTTP servers. Live
provider/Docker tests are opt-in; never supply real credentials to an ordinary
test run. Preserve failure, cancellation, permission and race coverage.

## Dependencies and publication

Use versioned Go dependencies; no committed sibling `replace` or `go.work`.
Keep English and Vietnamese root READMEs in sync. Preserve MIT and upstream
notices, fixture provenance and [SOURCE.json](SOURCE.json). The latter records
the original snapshot before import migration, not current-file checksums.
Never commit keys, tokens, vaults, `.env`, generated binaries or runtime state.
Publishing/tagging requires an explicit request. Test/build never deploys.
