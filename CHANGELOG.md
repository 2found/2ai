# Changelog

## 0.1.2 — 2026-10-10

Baseline: v0.1.1. This release preserves the existing public contracts and requires no configuration migration.

- Reduce allocations in lossless JSON parsing, transcript serialization and event-stream draining while preserving Unicode, raw numeric values and property ordering.
- Bound memory consolidation staging and model input, drain queued evidence fairly across scopes, and retry a timed-out deferred pass once while preserving durable evidence on failure.
- Add coalesced telemetry change notifications so hosts can collect records without idle polling.
- Clarify delegated-task context and host authority in subagent tool guidance.

Upgrade consuming Go modules with `go get github.com/2found/2ai@v0.1.2`, then run their compatibility checks.
