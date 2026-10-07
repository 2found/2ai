# Pi provenance

The native Go port was derived from [earendil-works/pi](https://github.com/earendil-works/pi/tree/eeac84ca92498ac18b6832754d01aef1d3c5f654), commit `eeac84ca92498ac18b6832754d01aef1d3c5f654`.

`UPSTREAM.json` records the original 412 file hashes; these were verified before removing the development reference. `LICENSE` retains the upstream MIT notice. This directory contains no executable dependency or build tooling.

The TypeScript source, worker, generated bundles, fixture generators and Bun configuration were removed after native integration. Recorded JSON fixtures and Go contract tests remain in `ai/testdata`, `agentcore/engine/testdata`, `telemetry/testdata` and `internal/runtime/testdata`. Source hashes establish provenance, not universal behavioral equivalence.
