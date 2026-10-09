# 2ai

**AI foundation for systems that need models, tools and agents.**

[English](README.md) · [Tiếng Việt](README.vi.md) · [2found](https://2found.dev)

2ai is a Go library for building AI applications with shared provider contracts,
a native agent loop and telemetry. It supplies the foundation used by
[2agent / Soot](https://github.com/2found/2agent) and AgentRay. Your application
owns configuration, credentials, permissions and durable state.

## Start in a minute

Requires **Go 1.25 or later**. For private repository access, authenticate GitHub
and set `GOPRIVATE=github.com/2found/*` before downloading modules.

```sh
git clone https://github.com/2found/2ai.git
cd 2ai
go mod download
go run ./examples/scripted
```

Expected output: `Hello from 2ai.` The example runs the native agent with a
scripted provider; it needs no API key, external service or JavaScript runtime.
Read [the complete example](examples/scripted/main.go), then explore the public API:

```sh
go doc ./agentcore.Config
go doc ./ai.FallbackProvider
go doc ./telemetry.NewInMemory
```

To use 2ai in your own Go module:

```sh
go get github.com/2found/2ai@main
```

Go records a commit-derived version in `go.mod`. Commit `go.mod` and `go.sum`;
keep deployments pinned to that version. Import packages such as
`github.com/2found/2ai/ai` and `github.com/2found/2ai/agentcore`.

## Choose the layer you need

| Package | Use it for | Reference |
| --- | --- | --- |
| `ai`, `ai/protocol` | Transcripts, provider streaming, OAuth and retry/fallback | [AI](ai/README.md) |
| `agentcore` | Compose agents, limits, policies and tool grants | [AgentCore](agentcore/README.md) |
| `agentcore/engine` | Low-level native loop and event streams | [Engine](agentcore/engine/README.md) |
| `agentcore/host` | Checkpoints and reusable host utilities | `go doc ./agentcore/host` |
| `agentcore/plugins` | Goals, memory, todo, jobs, subagents and guards | [Plugins](agentcore/plugins/README.md) |
| `telemetry`, `telemetry/export`, `telemetry/llm` | Span recording and settled model traces | [Telemetry](telemetry/README.md) |
| `credential` | In-memory host vault and secret references | `go doc ./credential` |
| `sandbox` | Host-bound file, shell, HTTP and execution tools | `go doc ./sandbox` |
| `jsonjs` | Lossless JSON and shared value identity | `go doc ./jsonjs` |
| `testing/agentcore` | Native session fixtures for consumer tests | `go doc ./testing/agentcore` |

The foundation has no application server, product database or CLI. Use
[2agent](https://github.com/2found/2agent) for Soot's workspace, conversations,
schedules, packs and app.

## Contracts that matter

- Providers preserve ordered content, thinking signatures, tool IDs and usage.
- AI owns bounded retry/fallback. Visible output, cancellation and host failures
  stop fallback; the host binds selection and accounting callbacks.
- AgentCore owns one model loop. The engine stays below application policy;
  plugins extend composition without creating another runtime.
- The native subagent plugin owns delegation guidance alongside its tool schema;
  hosts supply authorized destinations and callbacks, and skills supply domain
  knowledge. No separate orchestrator agent or model loop is required.
- Hosts resolve credentials and grant tools. Installing a capability does not
  grant permission. A workspace directory does not provide OS isolation.
- Native checkpoints and live value identity have ownership rules. Read package
  documentation before copying state or handling events concurrently.

Configured native transports cover Codex, Claude Code, Antigravity, Devin and
OpenAI-compatible endpoints, including Gemini. Support varies by provider;
[AI documentation](ai/README.md) is the contract, not a promise of support for
every model in a provider catalog.

## Versioned releases

A new `VERSION` pushed to `main` releases only after CI passes. Each tag publishes
Go-module source, matching documentation and a checksummed download index.
[Release policy](docs/release.md) · [Releases and downloads](https://github.com/2found/2ai/releases).
Pin a published `v<version>` in consumers; `@main` above remains a development path.

## Develop and verify

```sh
go test -race ./...
go vet ./...
go run ./examples/scripted
```

Ordinary checks use recorded fixtures, scripted providers and local HTTP
servers. Install Node.js 22.15+ and Python 3 to exercise the optional sandbox
eval tests; the Go library itself has no JavaScript runtime dependency.
Live provider and Docker tests skip unless their prerequisites are
explicitly supplied. The suite covers failure, cancellation, permissions,
retry accounting and concurrency. No sibling checkout or local `replace` is
required. For coordinated development, use an uncommitted Go workspace.

## For coding agents

Read [AGENTS.md](AGENTS.md) first. It gives ownership boundaries, verification
commands and credential rules. Read the nearest package README and its boundary
tests next; use `go doc` to inspect exported types. Keep runtime behavior in its
owning layer and workload knowledge in the consuming application's config,
skills or tools. Keep this README and [the Vietnamese version](README.vi.md)
in sync when changing setup or supported behavior.

## Origin and license

Extracted from AgentRay's shared AI packages. [SOURCE.json](SOURCE.json) records
the source commit, original file hashes and path mapping. `credential`,
`sandbox`, JSON primitives and test helpers accompany the three main packages
so consumers build independently. AgentRay application and analytics code remain
in their own repository.

[MIT](LICENSE), with original notices preserved. Derived Pi, TypeBox,
partial-json and Unicode components retain their own notices beside the code;
[Pi provenance](third_party/pi/README.md) records upstream source hashes.

A [2found](https://2found.dev) project. **From idea to company.**
