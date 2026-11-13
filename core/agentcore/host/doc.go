// Package host supplies reusable policy and checkpoint utilities around the
// native engine. It owns request compaction and transcript identity/validation;
// engine owns scheduling, ai owns provider messages, and consumers own storage,
// credentials, model selection and the summarizer callback.
// InputMessages converts new host instructions, and ProjectMessage/ProjectContent
// produce detached display views for plugins and observers. Neither projection
// is a substitute for the native transcript.
//
// A Compactor transforms only the provider request view (Prepare). Keep the
// original native messages as the authoritative transcript and persist
// Checkpoint and Cleared alongside them. Restore verifies the summary
// envelope/revision, and every request verifies that its prefix still matches
// before using it. The prefix digest and the facts read from each message are
// memoized, so a request re-derives only the messages appended since the last.
// A legacy agentcore.Message display projection is not a native checkpoint and
// must not be replayed as one.
//
// Requests are reduced in three steps, cheapest first:
//
//   - Tool-result clearing (CompactionPolicy.Clearing, the analogue of the
//     Messages API's clear_tool_uses) replaces old tool results with a
//     placeholder naming the call and how to recover it, keeping the most
//     recent Keep. The boundary is persisted (Cleared/RestoreCleared), so a
//     cleared result is the same bytes on every request; it moves only for a
//     saving of ClearAtLeast, and only past Trigger, once the prompt cache has
//     gone cold, or together with a new summary.
//   - Summary: a structured handoff (SummaryInstructions, ExtractSummary,
//     ClampSummary) replaces the settled prefix. CompactionPolicy.Summarize
//     receives the uncleared native prefix, which RenderTranscript turns into
//     bounded role-labelled text and FoldRendered splits on message
//     boundaries. CompactionPolicy.SummarizeView, when set, is tried first with
//     the cleared live view so the call can read the conversation's prompt
//     cache. Reattach appends context the work needs verbatim after the
//     summary, within MaxReattachTokens. Allow defers a summary, Observe
//     reports each one, and after MaxSummaryFailures consecutive failures the
//     Compactor stops asking until a forced (recovery) compaction succeeds.
//     Compact summarizes now, with its own instructions: the host's /compact.
//   - Preview: a tool result still too large for the budget is archived
//     (CompactionPolicy.Archive) and replaced by its head and the locator. The
//     preview size depends only on the budget, so it too repeats exactly.
package host
