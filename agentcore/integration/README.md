# agentcore integration tests

Black-box tests that compose the public `agentcore` API with real plugins live
here. Keeping them outside the kernel package root makes the ownership boundary
visible:

- root tests exercise loop-owned behavior and may use package-private helpers;
- this suite imports `agentcore` exactly as a consumer does;
- plugin unit tests stay beside their plugin;
- `live_provider_test.go` remains opt-in through the existing
  `AGENTRAY_TEST_*` environment variables.

Run the whole module with:

```sh
go test ./agentcore/...
```
