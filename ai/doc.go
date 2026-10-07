// Package ai owns provider transports, native transcripts, stream events,
// OAuth/account pools and model catalogs. The native engine consumes its
// AssistantMessageEventStream and lossless Message types directly.
//
// The native providers include Codex Responses, Antigravity, Gemini-compatible
// Completions and Claude Code/Anthropic Messages. Provider-neutral client
// contracts, provider session state and retry/error classification live in
// ai/protocol, below both this package and agentcore. Neither ai nor protocol
// imports the agent runtime.
//
// NativeProvider dispatches native transports; FallbackProvider composes them
// with bounded retries and ordered fallback under the same StreamFn contract.
// Durable hosts bind selection and usage callbacks through FallbackProvider.Run;
// retry/fallback never replays a request after visible content has escaped.
//
// NewClient retains the Chat/Stream client API used by SDK integrations and
// connectivity probes. New adds managed-provider identity and model discovery;
// Collection routes those clients by model. These client contracts are distinct
// from the native transcript and stream types used by engine.
//
// Conversation-scoped provider state can cache endpoint capabilities or
// account-specific transport state. It is an optimization, not durable agent
// history; an empty cache must remain correct, and account rotation selectively
// clears account-scoped records.
package ai
