// Package host supplies reusable policy and checkpoint utilities around the
// native engine. It owns request compaction and transcript identity/validation;
// engine owns scheduling, ai owns provider messages, and consumers own storage,
// credentials, model selection and the summarizer callback.
// InputMessages converts new host instructions, and ProjectMessage/ProjectContent
// produce detached display views for plugins and observers. Neither projection
// is a substitute for the native transcript.
//
// A Compactor transforms only the provider request view. Keep the original native
// messages as the authoritative transcript and persist Checkpoint alongside
// them. Restore verifies the summary envelope/revision; Transform verifies that
// its prefix still matches before using it. A legacy agentcore.Message display
// projection is not a native checkpoint and must not be replayed as one.
package host
