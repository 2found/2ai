# advisor

**Extension.** Ejectable — the loop never names it. Without it, the answer the
agent produces is the answer the run returns.

A reviewer reads the finished work before the run is allowed to end, and a note
that matters sends the agent back to resolve it. Same seam as
[`finishguard`](../finishguard/README.md); what changes is that the verdict
comes from a reviewer instead of a rule. It is the capability finishguard's
README lists as its own limitation: *"the guard sees only passive evidence — it
cannot run a check of its own, call a tool, or consult a model."*

The plugin never learns the reviewer is a model. It calls a `Reviewer` func the
host supplies, which is what keeps it testable without a provider and keeps the
model, tier, and credential choices where those decisions already live
(`internal/runtime/advisor.go` supplies agentray's).

Ported from oh-my-pi's advisor subsystem.

## Model Experience

### The reviewer raises a concern or a blocker

#### What the model sees

The advisories, injected as one synthetic user message, and the run continues
instead of returning:

```
[System: an advisor reviewed the answer you were about to give and raised the
following. It is a second opinion from a reviewer that did NOT do the work and
may be wrong.

<advisory severity="concern" guidance="weigh, don&#39;t blindly obey">
the revenue figure sums a filtered and an unfiltered query
</advisory>

Check each point against what you actually did. Fix what is right; for anything
you judge wrong or already handled, say so in one line with the reason. This
note is not visible to the user and is not a question to answer: your next
message is what the user reads, so reply with the COMPLETE answer they should
see, never a comment about this note. If nothing needs to change, send the
answer again unchanged.]
```

Two parts of that wording are load-bearing.

`guidance="weigh, don't blindly obey"` is the agent's **only** cue for how to
treat an advisory — the primary system prompt never mentions them. A tool-less
reviewer is wrong often enough that a note which reads like an order gets
obeyed when it should have been argued with.

The closing instruction exists because the injection arrives as a *user*
message, so whatever the model says next is what the owner reads. Without it, a
model that judges itself already compliant replies with its verdict on the note
("the concern does not apply") and that verdict silently **replaces** the
answer. The owner asked a question and gets back a self-audit about a reply
they never saw. Same failure the evidence guard's nudge documents; the repair
path must always terminate in the full answer.

The user sees a curated progress note (`reviewing the answer with the
advisor`) — never the raw injection.

#### Token effect

**One reviewer call per round** (input: a bounded window of the transcript;
output: capped and terse), plus the injection and the extra agent turn it buys.
Bounded by `MaxRounds` (default 2). This is the most expensive plugin in the
composition per run, which is why the host is expected to make it opt-in.

#### KV cache effect

**Append-only for the agent.** The advisory lands after the rejected answer;
nothing earlier moves. The reviewer's own call shares no prefix with the run
and is not cached across rounds.

### The reviewer raises only nits

#### What the model sees

Nothing. The finish is accepted and the nits go to `OnNotes`.

At a finish there is no next step boundary for an aside to ride, so a nit has
nowhere to go except a whole extra turn — which costs more than the nit is
worth. omp delivers nits as batched asides because its advisor watches a
*running* session; that channel does not exist here.

#### Token effect

**Zero-direct** beyond the reviewer call itself.

### The reviewer is silent, errors, or the cap is spent

#### What the model sees

Nothing. The run returns the answer it produced.

Silence is the expected outcome of a run that went fine — the shipped prompt
says so in as many words, because a reviewer's default failure is not missing
bugs, it is finding something to say.

## Impact on the agent

- Consulted only on a **normal** finish. A budget wrap-up, a tool-budget stop,
  a `MaxTurns` stop, an abort, or a terminal tool never re-opens the run.
- **A failed reviewer accepts the finish.** A provider outage, a timeout, or an
  unparseable response leaves the run exactly as it would have been with no
  advisor configured. The advisor is a second opinion, never a dependency.
- The cap is enforced against `StopInfo.Attempt`, a count the **loop** keeps —
  so a reviewer that never runs out of objections is still bounded.
- The injection is persisted to the durable log like a steer, so a resumed run
  replays the conversation the model actually saw.
- The plugin needs **no kernel change**: `StopInfo` carries the answer and the
  tool trace but not the history, so the extension implements `RunObserver` and
  keeps the last `PhaseRequest` snapshot — exactly the context the answer came
  out of.
- A `PhaseRebase` (compaction, a context edit) replaces the history the earlier
  notes were about, so the dedupe history is dropped with it.
- **Ordering matters.** Interceptors are consulted in registration order and
  the first to say `Continue` wins. Register the advisor **after** the goal gate
  and the finish guard: an unmet goal makes any review of that answer moot, and
  a rule that costs no tokens should not be rediscovered by a pro-tier call.

## The emission guard

Between what the reviewer says and what the agent reads sits
[`emission.go`](./emission.go), and it is the part that decides whether the
feature is usable.

Reviewer models do not obey the prose rules in their own prompt. oh-my-pi issue
#3520 recorded one session with **309 advise calls covering 92 unique notes —
114× "Stop.", 52× "No issue; continue.", 41× "Done."** The fix is to make the
rules executable:

1. **Clamp** each note to `MaxNoteRunes` (2000). The reviewer read a transcript
   that may contain attacker-controlled text and its note becomes a user
   message in the agent's conversation, so an unbounded note is an unbounded
   injection.
2. **Normalize** — lowercase, NFKC, non-alphanumeric runs to one space, trim.
   `"Stop."`, `"*Stop*"`, `"  stop  "` all key to `stop`.
3. **Drop content-free phrases** — `stop`, `done`, `lgtm`, `no issue continue`,
   and similar. Silence is how "no concerns" is expressed. A genuine blocker
   that merely *opens* with such a word (`"Stop: the revenue query sums a
   filtered and an unfiltered CTE"`) does not match, because normalization does
   not truncate.
4. **Dedupe by normalized text, by escalation rank.** A repeat passes only when
   its severity strictly exceeds what was already delivered for that text, so a
   reviewer cannot buy a second turn by re-raising the same point — but a real
   `nit → concern → blocker` escalation gets through.
5. **Bound breadth** at `MaxNotesPerReview` (3). Suppressed noise never spends
   the budget: a junk note must not burn the slot for the real concern behind
   it in the same review.

The gate is deliberately **invisible to the reviewer**. Telling a model its
note was suppressed teaches it to rephrase the same useless note to get past
the filter, which defeats the dedupe and costs a round trip to do it.

Rendering escapes the note text, so a quoted `</advisory>` cannot close the
element and let what follows read as the host's own instructions.

## Known limitations and deferred work

- **The reviewer has no tools here.** It reviews what is in the transcript. In
  agentray that is usually enough (the SQL and the rows it returned are both
  there), but a reviewer that could check a claim against the source would
  catch a class this one cannot. omp grants `read`/`grep`/`glob`; the agentray
  analog is a nested run with its own scope gating and budget — a second
  governed agent, not a config change.
- **`Delivered` is per run, not per session.** A resumed run gets a fresh
  dedupe history, so a note it raised before its crash can be raised again.
- **First interceptor to say `Continue` wins.** If the goal gate or the finish
  guard re-opens the run that turn, the advisor is not consulted at all.
- **The reviewer sees a window, not the whole run.** The host bounds the
  transcript; the elision is marked so the reviewer knows not to conclude that
  an unseen step never happened, but it can still miss what fell in the gap.
- **A different tier may mean a different provider.** The reviewer reads the
  run's transcript, so turning it on can send that conversation to a vendor the
  run itself did not use. That is a workspace configuration question, not
  something the plugin can decide.

## Periodic and multiple reviewers

`Plugin.IntervalTurns` enables review at existing turn boundaries, with
`MaxPeriodicReviews` (default 8), `CooldownTurns` after errors (default 3), and
`Timeout` (default 30 seconds). Finish review retains its separate two-round cap.
There is no background scheduler: reviews are synchronous, cancellable callbacks
before the next primary request. Custom callbacks must respect their context.

`Plugin.Reviewers` accepts up to four uniquely named `ReviewerConfig` values,
each with its own reviewer, cadence, caps, cooldown and emission guard. Evidence
is bounded to 32 text messages / 32 tool traces, and delivered-note history to
32 notes. Cadence, budgets, cooldown and dedupe are checkpointed. A fresh user
request after finish review starts a new review allowance. Compaction resets
note dedupe but does not replenish review quotas.

`NativeWithOptions(provider, options)` applies these scheduling controls to the
native reviewer; all fallback attempts are charged to the owning agent. Reviewer
errors accept the primary result and defer periodic retries through cooldown.
