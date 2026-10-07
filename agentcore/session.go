package agentcore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"time"
)

// SessionEntryKind classifies one append-only entry in a durable session log
// (pi's semi-durable harness). The log is the source of truth: run state is
// rebuilt by reducing it, never by mutating a row in place.
type SessionEntryKind string

// Native Pi records keep opaque JSON in SessionEntry.Content. The host owns
// their persistence; legacy RecoverSession must not project this transcript.
const (
	EntryPiState SessionEntryKind = "pi_state"
	// EntryPiModelSelection commits a host-owned native ladder identity without credentials.
	EntryPiModelSelection SessionEntryKind = "pi_model_selection"
	EntryPiEvent          SessionEntryKind = "pi_event"
	EntryPiEffectStart    SessionEntryKind = "pi_effect_start"
	EntryPiEffectDone     SessionEntryKind = "pi_effect_done"
	// EntryPiAnswer is host workflow metadata. Content holds the exact native
	// user message to append; it never replaces an already emitted tool result.
	EntryPiAnswer SessionEntryKind = "pi_answer"
	EntryPiGoal   SessionEntryKind = "pi_goal"
	// EntryPiGoalRevision records a committed host contract change within a
	// physical tool effect. It never substitutes for a native tool result.
	EntryPiGoalRevision SessionEntryKind = "pi_goal_revision"
	// Invocation binds an isolated native run to its immutable request; Result
	// records a completed child's reattachable answer without rewriting messages.
	EntryPiInvocation  SessionEntryKind = "pi_invocation"
	EntryPiChildResult SessionEntryKind = "pi_child_result"
	// EntryPiDelegation records continuation of a parked delegation. It links
	// an answered parent question to a new child question or a final delivery,
	// without replacing the original provider tool result.
	EntryPiDelegation SessionEntryKind = "pi_delegation"
	// EntryPiDelegationBatch commits the resumed batch policy and its deliveries
	// before any completion context reaches the next native model request.
	EntryPiDelegationBatch SessionEntryKind = "pi_delegation_batch"
	// Request-view summary metadata; never replaces native transcript messages.
	EntryPiContextSummary SessionEntryKind = "pi_context_summary"
)

const (
	// EntryMessage records one conversation message reaching its final form
	// (message_end): a user prompt, an assistant turn, or a tool result.
	EntryMessage SessionEntryKind = "message"
	// EntryLeaf marks the run reaching a final answer — a leaf of the session
	// tree. Its presence means the run completed normally.
	EntryLeaf SessionEntryKind = "leaf"
	// EntryCompaction brackets a compaction. Final=false is the start record
	// (compaction in flight); Final=true is the completion record carrying the
	// summary. A start with no matching completion means compaction was
	// interrupted and must be re-run on resume.
	EntryCompaction SessionEntryKind = "compaction"
	// EntryBranchSummary records a fork/branch checkpoint summary.
	EntryBranchSummary SessionEntryKind = "branch_summary"
	// EntryModelChange records the active model switching (escalation or a
	// save-point bump), so a resumed run reconstructs the right model.
	EntryModelChange SessionEntryKind = "model_change"
	// EntryActiveToolsChange records the active tool set changing mid-run (a
	// PrepareNextTurn swap), so a resumed run rebuilds the tools the crashed
	// run ended on rather than the registry it started with.
	EntryActiveToolsChange SessionEntryKind = "active_tools_change"
	// EntryTurnInterrupted is written by recovery to mark a turn that never
	// completed (a crash between a tool result and the next assistant turn).
	EntryTurnInterrupted SessionEntryKind = "turn_interrupted"
	// EntryToolDisabled records the circuit breaker disabling a tool for the rest
	// of the run after repeated failures, so the disable is reconstructed on
	// resume (and the broken tool isn't retried from scratch).
	EntryToolDisabled SessionEntryKind = "tool_disabled"
	// EntryLeafMove is a control entry (not a tree node): it moves the session's
	// active leaf to Target, so the next appended entry chains from there. This
	// is how a consumer rewinds a session to an earlier entry, or switches to
	// another branch, without rewriting history (pi's branch()/leaf pointer).
	EntryLeafMove SessionEntryKind = "leaf_move"
	// EntryGoal records the run's goal-gate condition (Config.Goal), written at
	// the start of a goal-gated run so a resume re-arms the gate — without it a
	// crashed /goal run would resume ungated while its replayed transcript still
	// carries the gate's nudge messages. A leaf closes the goal along with the
	// run, so later runs chained onto the same log are not gated by it.
	EntryGoal SessionEntryKind = "goal"
	// EntryQuestion records a tool call that parked the run waiting on a human
	// (the ask tool): CallID links it to the dangling call, Question carries the
	// validated arguments. The run ends without a tool result, so the call stays
	// dangling in the log — a resume either closes it with a matching
	// EntryAnswer or re-issues it when the tool is retry-safe (re-park).
	EntryQuestion SessionEntryKind = "question"
	// EntryAnswer records the human's answer to a parked EntryQuestion, keyed by
	// the same CallID. Recovery closes the dangling call with Answer as its tool
	// result — the answer IS the tool result, not a new user turn.
	EntryAnswer SessionEntryKind = "answer"

	// --- Side records (pi's intent/progress granularity) ---
	//
	// The kinds below are written OUTSIDE the per-turn save-point buffer —
	// directly to the store, mid-turn — because the crash window they cover is
	// inside a turn. To keep the log a valid prefix of the conversation they
	// are NOT tree nodes: buildChain skips them, so they never become the leaf
	// and never break a pending entry's chain. The fold reads them by scanning
	// the raw log, not the active path. Each is one half of an
	// intent→effect→settlement triple (pi's op state machine): the intent or
	// in-progress effect rides the side record, the settlement is an ordinary
	// chain entry.
	//
	// EntryInbox is a queued steering or follow-up message (Lane selects the
	// drain point). Written by Agent.Steer / Agent.FollowUp the moment the
	// caller enqueues, so a crash between queue and drain cannot lose human
	// input. Settled by EntryInboxDone when the loop delivers it into the
	// transcript as an EntryMessage.
	EntryInbox SessionEntryKind = "inbox"
	// EntryInboxDone settles an EntryInbox (Target = the inbox entry's ID). It
	// IS a chain entry: the fold's inbox scan reads it from the raw log, but
	// writing it through the save-point buffer keeps the delivered message and
	// its settlement in one atomic turn flush.
	EntryInboxDone SessionEntryKind = "inbox_done"
	// EntryAssistantFrame is a throttled snapshot of the assistant text
	// streamed so far this turn (pi's durable partial frames). Settled by the
	// turn's assistant EntryMessage; a trailing run of frames with no such
	// message is the draft the crashed turn had produced.
	EntryAssistantFrame SessionEntryKind = "assistant_frame"
	// EntryToolProgress is a throttled snapshot of one streaming tool call's
	// partial output (pi's tool progress checkpoint). Settled by the call's
	// tool-result EntryMessage; a trailing one tells recovery how far the
	// interrupted call reported getting.
	EntryToolProgress SessionEntryKind = "tool_progress"
	// EntryToolOutcome records one completed physical tool execution before the
	// turn can place its results in assistant source order. It is a side record:
	// a fast parallel sibling becomes durable without waiting for an earlier slow
	// call, while the eventual tool-result EntryMessage remains the canonical
	// transcript settlement. Recovery uses the outcome only when that settlement
	// is absent, so a normal run never duplicates a result.
	EntryToolOutcome SessionEntryKind = "tool_outcome"
)

// ToolOutcomeRecord is the bounded, model-visible result of one completed tool
// call plus the run semantics that must survive a crash before source-order
// placement. It deliberately stores no raw/unbounded tool output: Message is
// the same truncated or spill-backed value the live model receives.
type ToolOutcomeRecord struct {
	Message     Message          `json:"message"`
	Trace       ToolTrace        `json:"trace"`
	Invocations []ToolInvocation `json:"invocations,omitempty"`
	Extra       []Message        `json:"extra,omitempty"`
	Terminate   bool             `json:"terminate,omitempty"`
	Executed    bool             `json:"executed,omitempty"`
}

// SessionEntry is one immutable record in the append-only session log. The log
// is a tree, not just a line: ID/ParentID give each entry a stable address and
// a tree parent, so a consumer can branch from any earlier entry in place (pi's
// id/parentId session format). Both are optional — an entry without them chains
// implicitly to the entry appended before it, so a flat log written by an older
// writer is a single-branch tree and reduces exactly as before.
type SessionEntry struct {
	Seq      int              `json:"seq"` // append order, assigned by the store
	Kind     SessionEntryKind `json:"kind"`
	ID       string           `json:"id,omitempty"`        // stable entry id (writer-assigned)
	ParentID string           `json:"parent_id,omitempty"` // tree parent; for outcome side records, the intent anchor
	Target   string           `json:"target,omitempty"`    // EntryLeafMove: the new active leaf
	Turn     int              `json:"turn,omitempty"`
	Message  *Message         `json:"message,omitempty"` // EntryMessage
	Model    string           `json:"model,omitempty"`   // EntryModelChange
	Tools    []string         `json:"tools,omitempty"`   // EntryActiveToolsChange
	Tool     string           `json:"tool,omitempty"`    // EntryToolDisabled
	// Lane selects which queue an EntryInbox feeds: "steer" drains at the top
	// of the next turn, "follow" drains when the run would stop.
	Lane string `json:"lane,omitempty"`
	// CallID links a side record to the tool call it concerns: an
	// EntryToolProgress reports on it, an EntryQuestion parks it, an
	// EntryAnswer resolves it (the provider's call id, stable across a resume).
	CallID string `json:"call_id,omitempty"`
	// Content carries a side record's payload: the accumulated text for an
	// EntryAssistantFrame, the accumulated partial output for an
	// EntryToolProgress. Snapshots, not deltas — the newest wins.
	Content string `json:"content,omitempty"`
	Goal    string `json:"goal,omitempty"`    // EntryGoal: the run's completion condition
	Summary string `json:"summary,omitempty"` // EntryCompaction (completion) / EntryBranchSummary
	Final   bool   `json:"final,omitempty"`   // EntryCompaction completion marker
	// Retained is the transcript a completed EntryCompaction left behind: the
	// summary (or elided fallback) plus the recent tail kept verbatim, MINUS the
	// run's own leading system prompt, which every run re-derives and prepends
	// itself. It makes the compaction a self-contained checkpoint (pi's
	// CompactionEntry.retainedTail): reduce restarts history here instead of
	// replaying the span the summary already represents.
	//
	// Without it a resumed run silently rebuilds the FULL pre-compaction
	// history — measured at 3.75x the messages and 2.8x over the context budget
	// on a 60-turn run — then pays to summarize it all again, and the fresh
	// summary is not the one the original run reasoned over. Nil on a legacy
	// entry (and on the start half of the bracket), which reduces exactly as
	// before.
	Retained []Message `json:"retained,omitempty"`
	// State is the rest of the reduced run state as of a completed
	// EntryCompaction — everything the fold accumulated that Retained does not
	// carry. Retained makes the checkpoint self-contained for the TRANSCRIPT;
	// this makes it self-contained for the run, which is what lets a resume read
	// a suffix of the log instead of all of it (see SessionWindowStore).
	//
	// Nil on a legacy entry, and on the start half of the bracket. A nil State is
	// why a windowed read is an optimization a store opts into rather than the
	// default: a log whose newest checkpoint predates this field has no suffix
	// that reduces to the same answer, so it is read whole.
	State *CheckpointState `json:"state,omitempty"`
	// Usage records what the summarization call itself cost (EntryCompaction
	// completion / EntryBranchSummary). Compaction and branch summaries are real
	// billable provider calls; without this they are invisible spend (pi #6671).
	Usage *Usage `json:"usage,omitempty"`
	// Question is the parked call's validated arguments (EntryQuestion), kept
	// verbatim so a consumer can re-render the prompt and options without
	// knowing the tool's schema.
	Question json.RawMessage `json:"question,omitempty"`
	// Answer is the human's reply to a parked question (EntryAnswer): the text
	// recovery feeds back as the call's tool result.
	Answer string `json:"answer,omitempty"`
	// Outcome is populated only for EntryToolOutcome. CallID is kept on the
	// envelope so stores and recovery can identify it without decoding a nested
	// message first.
	Outcome   *ToolOutcomeRecord `json:"outcome,omitempty"`
	CreatedAt time.Time          `json:"created_at"`
}

// CheckpointState is the non-transcript half of a checkpoint: the run state a
// fold would have accumulated by the time the compaction completed. The loop
// stamps it from state it already holds rather than re-deriving it, so it is
// exact by construction.
// It is AUTHORITATIVE, not additive: the fold adopts it wholesale, because a
// suffix has no earlier entries to merge with. A writer that fills it partially
// does not degrade gracefully — it erases the fields it left out — which is why
// only the loop writes one.
type CheckpointState struct {
	Model         string   `json:"model,omitempty"`
	ActiveTools   []string `json:"active_tools,omitempty"`
	DisabledTools []string `json:"disabled_tools,omitempty"`
	Goal          string   `json:"goal,omitempty"`
	// Completed records whether an EntryLeaf was already seen before this
	// checkpoint — true only on a chained log whose earlier run finished. Without
	// it a window would report a chained session as unfinished where the whole
	// log reports it finished, and the two reads would disagree about whether
	// there is an answer to reattach to.
	Completed bool `json:"completed,omitempty"`
}

// clone snapshots the state for a log entry. The loop keeps one running mirror
// and stamps it on every checkpoint, so without the copy each entry would share
// the slice the next disabled tool appends to — and a log's older checkpoints
// would appear to have known about tools disabled after them.
func (c CheckpointState) clone() *CheckpointState {
	out := c
	out.ActiveTools = slices.Clone(c.ActiveTools)
	out.DisabledTools = slices.Clone(c.DisabledTools)
	return &out
}

// SessionStore is the append-only durability seam (extends the working-memory
// MemoryStore conceptually; kept separate so a consumer can adopt durability
// incrementally). The store assigns Seq and never mutates a written entry.
// Append takes an immutable value snapshot: later changes through the caller's
// pointers or slices must not alter the stored record. Log likewise returns a
// snapshot whose mutation cannot rewrite the store.
type SessionStore interface {
	// Append records one entry; the store assigns its Seq. It MUST be safe for
	// concurrent use on the same sessionID: a run's loop appends serially, but
	// Steer/FollowUp enqueue and side-record writes race it from other
	// goroutines, and a sibling process resuming the same log appends too. A
	// store that assigns Seq by read-max-then-insert must resolve collisions
	// (retry on a unique violation, or serialize per session) — a lost or
	// duplicated Seq silently corrupts the fold.
	Append(ctx context.Context, sessionID string, entry SessionEntry) error
	// Log returns the full ordered entry log for a session.
	Log(ctx context.Context, sessionID string) ([]SessionEntry, error)
}

// ErrSessionLeaseLost reports that a durable-session owner no longer holds the
// fencing token that authorized its resume. Callers must stop provider/tool
// work and may retry by acquiring a fresh lease.
var ErrSessionLeaseLost = errors.New("agentcore: durable session lease lost")

// ErrNoPendingQuestion reports that an answer does not match an open ask call.
var ErrNoPendingQuestion = errors.New("agentcore: no pending question")

// ErrAnswerConflict reports a second, different answer for an already-answered
// call. Retrying the same answer is idempotent; changing it is not.
var ErrAnswerConflict = errors.New("agentcore: answer conflicts with recorded answer")

// SessionLeaseStore is an optional SessionStore capability for exclusive,
// renewable ownership of a resumed durable session. The returned context must
// be used for all resumed work and appends: distributed backends attach a
// fencing token to it and cancel it if renewal fails. Local stores may implement
// the same contract with an in-process lock.
type SessionLeaseStore interface {
	AcquireSessionLease(ctx context.Context, sessionID string) (leaseCtx context.Context, release func() error, err error)
}

type sessionLeaseMarkerKey struct{}

// AcquireSessionLease uses the strongest ownership contract a store exposes.
// Legacy stores remain source-compatible and receive a no-op lease; they retain
// their previous single-owner semantics.
func AcquireSessionLease(ctx context.Context, store SessionStore, sessionID string) (context.Context, func() error, error) {
	if held, _ := ctx.Value(sessionLeaseMarkerKey{}).(string); held == sessionID {
		return ctx, func() error { return nil }, nil
	}
	if leases, ok := store.(SessionLeaseStore); ok {
		leaseCtx, release, err := leases.AcquireSessionLease(ctx, sessionID)
		if err != nil {
			return ctx, nil, err
		}
		if leaseCtx == nil {
			leaseCtx = ctx
		}
		if release == nil {
			release = func() error { return nil }
		}
		return context.WithValue(leaseCtx, sessionLeaseMarkerKey{}, sessionID), release, nil
	}
	return ctx, func() error { return nil }, nil
}

// RecordSessionAnswer appends one answer to the currently pending question.
// It is idempotent for an exact retry and rejects a different answer for the
// same call. Hosts should call it while holding the session lease so the read
// and append form one logical ownership interval.
// Native Pi questions use a physical effect ID and an append-only user message;
// legacy questions retain their original provider-call/tool-result workflow.
// Delegated questions forward to the settled child question under its lease
// before recording the parent's acknowledgement. Exact retries finish a partial
// child/parent commit without changing or duplicating the child's answer.
func RecordSessionAnswer(ctx context.Context, store SessionStore, sessionID, callID, answer string) (bool, error) {
	log, err := store.Log(ctx, sessionID)
	if err != nil {
		return false, err
	}
	for _, entry := range log {
		if (entry.Kind != EntryAnswer && entry.Kind != EntryPiAnswer) || entry.CallID != callID {
			continue
		}
		if entry.Answer == answer {
			return false, nil
		}
		return false, fmt.Errorf("%w for call %q", ErrAnswerConflict, callID)
	}
	pendingID, _, found := PendingQuestion(log)
	if !found {
		return false, ErrNoPendingQuestion
	}
	if callID == "" {
		callID = pendingID
	}
	if callID != pendingID {
		return false, fmt.Errorf("%w: expected call %q, got %q", ErrNoPendingQuestion, pendingID, callID)
	}
	entry := SessionEntry{Kind: EntryAnswer, CallID: callID, Answer: answer}
	for _, candidate := range log {
		if id, question, native := PiQuestionFromEntry(candidate); native && id == callID {
			if _, route, routed := PiChildQuestionFromEntry(candidate); routed {
				if err := forwardPiChildAnswer(ctx, store, sessionID, route, answer); err != nil {
					return false, err
				}
			}
			entry.Kind = EntryPiAnswer
			message, err := json.Marshal(map[string]any{
				"role": "user", "content": "Human answer to question " + string(question) + ":\n" + answer,
				"timestamp": time.Now().UnixMilli(), "agentrayAnswerId": callID,
			})
			if err != nil {
				return false, err
			}
			entry.Content = string(message)
			break
		}
	}
	if err := store.Append(ctx, sessionID, entry); err != nil {
		return false, err
	}
	return true, nil
}

// Commit the child answer first. If the parent append fails, an exact retry
// finds the identical child answer and finishes the parent append; conflicting
// answers cannot partially overwrite either workflow. Leases always descend
// through the session namespace, so a forged cyclic route cannot deadlock.
func forwardPiChildAnswer(ctx context.Context, store SessionStore, parentID string, route *ChildQuestionError, answer string) (err error) {
	if !strings.HasPrefix(route.SessionID, parentID+"/") {
		return errors.New("delegated question is outside the parent session")
	}
	childCtx, release, err := AcquireSessionLease(ctx, store, route.SessionID)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, release()) }()
	entries, err := store.Log(childCtx, route.SessionID)
	if err != nil {
		return err
	}
	matched := false
	for _, entry := range entries {
		id, question, valid := PiQuestionFromEntry(entry)
		if !valid || id != route.QuestionID {
			continue
		}
		var actual, expected any
		a, b := json.NewDecoder(bytes.NewReader(question)), json.NewDecoder(bytes.NewReader(route.Question))
		a.UseNumber()
		b.UseNumber()
		if a.Decode(&actual) != nil || b.Decode(&expected) != nil || !reflect.DeepEqual(actual, expected) {
			return errors.New("delegated question differs from the child's settled receipt")
		}
		matched = true
		break
	}
	if !matched {
		return errors.New("delegated question has no settled child receipt")
	}
	_, err = RecordSessionAnswer(childCtx, store, route.SessionID, route.QuestionID, answer)
	return err
}

// SessionBatchStore is an optional SessionStore capability for committing one
// save point atomically. The loop buffers chain entries for a turn and hands
// the whole ordered slice to AppendBatch; either every entry must become
// visible, with consecutive sequence numbers in the supplied order, or none of
// them may become visible.
//
// Side records (stream frames, tool progress/outcomes, queued input) still use Append:
// their purpose is to become durable while a turn is in flight. AppendBatch is
// for the settled transcript boundary only. Backends that do not implement the
// capability remain supported; the loop falls back to ordered Append calls and
// retains the successfully written prefix on failure.
type SessionBatchStore interface {
	AppendBatch(ctx context.Context, sessionID string, entries []SessionEntry) error
}

// appendSessionBatch writes a buffered save point through the strongest
// contract the store offers. The count is the durable prefix: all-or-none for
// SessionBatchStore, and the successfully appended prefix for legacy stores.
func appendSessionBatch(ctx context.Context, store SessionStore, sessionID string, entries []SessionEntry) (int, error) {
	if len(entries) == 0 {
		return 0, nil
	}
	if batches, ok := store.(SessionBatchStore); ok {
		if err := batches.AppendBatch(ctx, sessionID, entries); err != nil {
			return 0, err
		}
		return len(entries), nil
	}
	for i, entry := range entries {
		if err := store.Append(ctx, sessionID, entry); err != nil {
			return i, err
		}
	}
	return len(entries), nil
}

// SessionWindowStore is an optional SessionStore capability: read the tail of a
// log instead of all of it.
//
// Resume is the one operation whose cost grows with the whole history. A long
// run's log is mostly messages the newest checkpoint already stands for — on a
// 4,200-turn run, 10,600 entries and 8 MiB, of which the resume needs the last
// few dozen — and reading it whole makes crash recovery slower the longer the
// run got, which is exactly backwards.
//
// The two methods are deliberately MECHANICAL. Neither decides whether a window
// is safe; both report facts, and LoadResumeLog (in the kernel) applies the
// rule. A store that answers these correctly cannot make a resume wrong, which
// is the property that matters when the alternative is every backend
// re-implementing the fold's preconditions.
type SessionWindowStore interface {
	// LogFrom returns the session's entries with Seq >= sinceSeq, in order.
	LogFrom(ctx context.Context, sessionID string, sinceSeq int) ([]SessionEntry, error)
	// CheckpointSeq reports the Seq of the newest completed EntryCompaction that
	// carries BOTH Retained and State (0 when there is none), and whether the
	// session contains any EntryLeafMove. Zero is unambiguous as "none" whatever
	// a store's Seq origin: a compaction always follows the messages it
	// summarizes, so it is never a log's first entry.
	//
	// Both facts are needed because both can defeat a window. Without State the
	// suffix loses the model, tools, and goal the fold had accumulated; with a
	// leaf move the newest checkpoint by Seq may sit on an ABANDONED branch, and
	// resuming from it would continue work the log says was rewound away.
	//
	// A third fact defeats it the same way: an EntryInbox queued BEFORE the
	// checkpoint and still unsettled (no EntryInboxDone naming it). The fold
	// reads pending inbox items from the raw log, so a window that starts past
	// one silently drops queued human input. A store that sees such an item
	// reports seq=0 — the full read is the only correct answer.
	CheckpointSeq(ctx context.Context, sessionID string) (seq int, branched bool, err error)
}

// LoadResumeLog reads the entries a resume needs: a suffix when the store can
// serve one and the log's shape allows it, the whole log otherwise.
//
// The suffix begins AT the checkpoint entry, not after it — the checkpoint is
// what carries the retained transcript and the run state, so dropping it would
// drop everything the window exists to preserve. Entries after it chain onto it
// implicitly, so the suffix is a well-formed log in its own right and reduces
// through the same fold, unchanged.
//
// Every failure degrades to the full read rather than to a partial one. A
// resume that reads too much is slow; a resume that reads too little silently
// forgets work the run already did.
func LoadResumeLog(ctx context.Context, store SessionStore, sessionID string) ([]SessionEntry, error) {
	ws, ok := store.(SessionWindowStore)
	if !ok {
		return store.Log(ctx, sessionID)
	}
	seq, branched, err := ws.CheckpointSeq(ctx, sessionID)
	if err != nil || branched || seq <= 0 {
		return store.Log(ctx, sessionID)
	}
	window, err := ws.LogFrom(ctx, sessionID, seq)
	if err != nil || len(window) == 0 || window[0].Seq != seq {
		// A window that does not start where it was told to is not the window
		// this function reasoned about; read the log rather than fold a suffix
		// whose first entry might not be the checkpoint.
		return store.Log(ctx, sessionID)
	}
	return window, nil
}

// InboxItem is one queued-but-undelivered steering or follow-up message: the
// EntryInbox intent (ID) plus the message it carries.
type InboxItem struct {
	ID      string
	Lane    string
	Message Message
}

// Inbox lanes: which drain point an EntryInbox feeds.
const (
	InboxSteer  = "steer"
	InboxFollow = "follow"
)

// ReducedState is the run state rebuilt by folding a session log: the message
// history to resume from, the active model and tools, and flags for whether the
// run completed and whether a compaction was left unfinished.
type ReducedState struct {
	Messages          []Message
	Model             string
	ActiveTools       []string
	DisabledTools     []string // tools the circuit breaker disabled (EntryToolDisabled)
	Completed         bool     // an EntryLeaf was seen
	PendingCompaction bool     // a compaction start with no completion
	Goal              string   // goal-gate condition of the unfinished tail run (EntryGoal)
	LastTurn          int
	// Inbox holds queued steering/follow-up messages never delivered before the
	// crash (EntryInbox with no matching EntryInboxDone). Side records are read
	// from the raw log — they are not tree nodes — so a pending inbox survives
	// even when it was appended mid-turn between two buffered entries.
	Inbox []InboxItem
	// Draft is the partial assistant text the in-flight turn had produced when
	// the run died (the newest trailing EntryAssistantFrame), "" when the last
	// turn settled or produced no text.
	Draft string
	// ToolProgress maps a tool-call ID to the newest partial output it reported
	// before the crash (trailing EntryToolProgress records), for calls whose
	// result never landed.
	ToolProgress map[string]string
	// ToolOutcomes holds completed physical executions whose canonical result
	// message may not have reached the transcript before a crash. Recovery
	// chooses them by the assistant's original call order.
	ToolOutcomes map[string]ToolOutcomeRecord
}

// ReduceSession folds an append-only log into the current run state. It is a
// pure function of the log — the heart of durable resume. The fold walks only
// the ACTIVE branch (leaf → root, reversed): a log that was rewound or branched
// reduces to the history the next turn should actually see, while abandoned
// branches stay in the log for inspection. A flat log's active branch is the
// whole log, so pre-tree logs reduce exactly as before.
func ReduceSession(log []SessionEntry) ReducedState {
	var rs ReducedState
	path := ActivePath(log)
	for _, e := range path {
		if e.Turn > rs.LastTurn {
			rs.LastTurn = e.Turn
		}
		switch e.Kind {
		case EntryMessage:
			if e.Message != nil {
				rs.Messages = append(rs.Messages, *e.Message)
			}
		case EntryBranchSummary:
			// A branch switch distilled the abandoned branch into this summary;
			// surface it to the resumed run as context, marker-prefixed like a
			// compaction checkpoint so the model reads it as background.
			if s := e.Summary; s != "" {
				rs.Messages = append(rs.Messages, Message{Role: RoleSystem, Content: branchSummaryMarker + "\n" + s})
			}
		case EntryModelChange:
			rs.Model = e.Model
		case EntryActiveToolsChange:
			rs.ActiveTools = e.Tools
		case EntryToolDisabled:
			// Append-once: a tool is disabled at most once per run, but guard against
			// a duplicated log entry so the reduced set stays a clean set.
			if e.Tool != "" && !slices.Contains(rs.DisabledTools, e.Tool) {
				rs.DisabledTools = append(rs.DisabledTools, e.Tool)
			}
		case EntryCompaction:
			// A start (Final=false) opens a pending compaction; the completion
			// (Final=true) closes it.
			if !e.Final {
				rs.PendingCompaction = true
				break
			}
			rs.PendingCompaction = false
			// A completed compaction is a CHECKPOINT, so history restarts at its
			// retained transcript: everything older is what the summary now
			// stands for, and replaying it would hand the resumed run a context
			// the live run had already shrunk away (pi: "context never reads past
			// a compaction"). Entries appended after this one chain on normally,
			// and a later compaction resets again — so the fold ends on the most
			// recent checkpoint plus the work done since.
			//
			// A nil Retained is a legacy entry or a compaction that predates this
			// field; leaving the accumulated history alone reproduces the old
			// full-replay behavior rather than truncating to nothing.
			if e.Retained != nil {
				rs.Messages = slices.Clone(e.Retained)
			}
			// The rest of the state the checkpoint captured. Applying it here is
			// what makes a suffix of the log fold to the same answer as the whole
			// log: the entries before the checkpoint contributed exactly this, so
			// replacing rather than accumulating is not a shortcut — it is the
			// same value arrived at directly. A full-log fold reaches this line
			// having already accumulated it, so the assignment is a no-op there,
			// which is why one fold serves both reads.
			if e.State != nil {
				rs.Model = e.State.Model
				rs.ActiveTools = slices.Clone(e.State.ActiveTools)
				rs.DisabledTools = slices.Clone(e.State.DisabledTools)
				rs.Goal = e.State.Goal
				rs.Completed = rs.Completed || e.State.Completed
			}
		case EntryGoal:
			rs.Goal = e.Goal
		case EntryLeaf:
			rs.Completed = true
			// The goal belongs to the run that just finished; a later run chained
			// onto this log (a chat continuation) must not inherit its gate.
			rs.Goal = ""
		}
	}
	// Side records are not tree nodes, so they are read from the raw log rather
	// than the active path — one forward pass collects every side kind.
	//
	// Pending inbox: every EntryInbox not settled by an EntryInboxDone. Both
	// halves are attributed to a branch by position — each anchors to the chain
	// node appended just before it — so a Rewind drops intents queued on the
	// abandoned branch AND the settlements recorded there: a steer delivered on
	// a rewound branch becomes pending again on the new one, which is the only
	// consistent answer (its delivered message is gone from the transcript).
	//
	// In-flight work: side records newer than the last SETTLING chain entry —
	// the last assistant or tool-result message — belong to the turn that never
	// finished. Mid-turn chain entries (a delivered steer, an inbox settlement)
	// can land after the last side record when a graceful interruption flushes
	// the turn's buffer, so the boundary is the last settling entry, not the
	// last chain entry. Newest wins: each record carries the accumulated
	// snapshot, not a delta.
	type queuedItem struct {
		item   InboxItem
		anchor int // Seq of the chain node this intent was appended after; -1 at log start
	}
	var queued []queuedItem
	var doneEntries []SessionEntry // EntryInboxDone chain nodes; filtered by branch below
	lastNodeSeq := -1
	lastSettled := -1
	var draft string
	var progress map[string]string
	type anchoredOutcome struct {
		callID   string
		outcome  ToolOutcomeRecord
		anchor   int
		anchorID string
	}
	var outcomes []anchoredOutcome
	for i, e := range log {
		if !isSideRecord(e.Kind) && e.Kind != EntryLeafMove {
			lastNodeSeq = e.Seq
		}
		switch e.Kind {
		case EntryInbox:
			if e.Message == nil {
				continue
			}
			key := e.ID
			if key == "" {
				key = fmt.Sprintf("#%d", e.Seq)
			}
			queued = append(queued, queuedItem{
				item:   InboxItem{ID: key, Lane: e.Lane, Message: *e.Message},
				anchor: lastNodeSeq,
			})
		case EntryInboxDone:
			doneEntries = append(doneEntries, e)
		case EntryAssistantFrame:
			if i > lastSettled {
				draft = e.Content
			}
		case EntryToolProgress:
			if i > lastSettled && e.CallID != "" {
				if progress == nil {
					progress = map[string]string{}
				}
				progress[e.CallID] = e.Content
			}
		case EntryToolOutcome:
			if e.CallID != "" && e.Outcome != nil {
				outcomes = append(outcomes, anchoredOutcome{callID: e.CallID, outcome: *e.Outcome, anchor: lastNodeSeq, anchorID: e.ParentID})
			}
		case EntryMessage:
			if e.Message != nil && (e.Message.Role == RoleAssistant || e.Message.Role == RoleTool) {
				// A settling entry closes the in-flight window: everything
				// collected so far belongs to turns that finished.
				lastSettled = i
				draft = ""
				progress = nil
			}
		}
	}
	if len(queued) > 0 || len(doneEntries) > 0 || len(outcomes) > 0 {
		activeSeq := make(map[int]bool, len(path))
		activeID := make(map[string]bool, len(path))
		for _, e := range path {
			activeSeq[e.Seq] = true
			if e.ID != "" {
				activeID[e.ID] = true
			}
		}
		var settled map[string]bool
		for _, d := range doneEntries {
			if activeSeq[d.Seq] {
				if settled == nil {
					settled = map[string]bool{}
				}
				settled[d.Target] = true
			}
		}
		for _, q := range queued {
			live := q.anchor == -1 || activeSeq[q.anchor]
			if live && !settled[q.item.ID] {
				rs.Inbox = append(rs.Inbox, q.item)
			}
		}
		for _, outcome := range outcomes {
			live := outcome.anchor == -1 || activeSeq[outcome.anchor]
			if outcome.anchorID != "" {
				live = activeID[outcome.anchorID]
			}
			if !live {
				continue
			}
			if rs.ToolOutcomes == nil {
				rs.ToolOutcomes = make(map[string]ToolOutcomeRecord)
			}
			// Completion records are immutable snapshots, but a later record for
			// the same call may include source-order accounting such as a breaker
			// note. Raw-log order makes the newest record authoritative.
			rs.ToolOutcomes[outcome.callID] = outcome.outcome
		}
	}
	rs.Draft = draft
	rs.ToolProgress = progress
	// A canonical result settles and supersedes its completion side record. This
	// keeps a whole-log fold equivalent to a checkpoint window that legitimately
	// starts after old side records, and prevents normal resumes from retaining
	// already-materialized outcomes as live state.
	for _, entry := range path {
		if entry.Kind == EntryMessage && entry.Message != nil && entry.Message.Role == RoleTool && entry.Message.ToolCallID != "" {
			delete(rs.ToolOutcomes, entry.Message.ToolCallID)
		}
	}
	if len(rs.ToolOutcomes) == 0 {
		rs.ToolOutcomes = nil
	}
	return rs
}

// RecoveryPolicy governs how an interrupted run is resumed.
type RecoveryPolicy string

const (
	// RecoveryMarkInterrupted is the conservative default: rebuild state, mark an
	// unfinished turn interrupted, re-run an unfinished compaction, and re-issue
	// only the tool calls whose tools declare themselves retry-safe.
	RecoveryMarkInterrupted RecoveryPolicy = "mark_interrupted"
)

// RetrySafeTool is an optional Tool capability: a tool that is safe to re-run
// after a crash (idempotent / read-only) declares RetrySafe() true. Recovery
// auto-retries only these; a tool that does not implement it is treated as
// non-idempotent and is never auto-retried (its dangling call is left for the
// model to decide).
type RetrySafeTool interface {
	RetrySafe() bool
}

// CallRetrySafeTool refines RetrySafeTool per invocation: a tool whose
// retry-safety depends on the specific call's arguments (spawn_subagent is
// crash-safe only when it self-forks, not when it routes to a delegate agent)
// implements this and receives the original dangling call. When both interfaces
// are implemented, this one wins.
type CallRetrySafeTool interface {
	RetrySafeCall(call ToolCall) bool
}

// isRetrySafe reports whether a registered tool opts into post-crash retry for
// this specific dangling call.
func isRetrySafe(tools *ToolSet, call ToolCall) bool {
	if tools == nil {
		return false
	}
	t, ok := tools.Get(call.Name)
	if !ok {
		return false
	}
	if cs, ok := t.(CallRetrySafeTool); ok {
		return cs.RetrySafeCall(call)
	}
	rs, ok := t.(RetrySafeTool)
	return ok && rs.RetrySafe()
}

// ResumePlan is the recovery output: the history to resume from, the active
// model/tools, the conservative decisions about an interrupted turn, and the
// side-record state (queued inbox, partial draft, tool progress) the crashed
// run had in flight.
type ResumePlan struct {
	Messages        []Message
	Model           string
	ActiveTools     []string
	DisabledTools   []string   // tools the circuit breaker disabled; re-applied on resume
	Completed       bool       // run already reached a leaf; nothing to resume
	Goal            string     // goal-gate condition of the interrupted run, re-armed on resume
	Interrupted     bool       // the last turn did not complete
	RerunCompaction bool       // an unfinished compaction must be re-run first
	RetryCalls      []ToolCall // dangling calls whose tools are retry-safe
	DroppedCalls    []ToolCall // dangling calls left for the model (not retry-safe)
	// Answers is the out-of-band tool results recorded while the run was parked:
	// call id -> the human's answer, from EntryAnswer entries. A dangling call
	// with an entry here is closed with the answer as its result — it is neither
	// retried nor interrupted.
	Answers map[string]string
	// Inbox holds steering/follow-up messages queued but never delivered before
	// the crash (pi's durable inbox). The resumed run drains them at their
	// lane's point and settles each with EntryInboxDone.
	Inbox []InboxItem
	// Draft is the partial assistant text the in-flight turn had produced;
	// surfaced to the resumed run as context, not replayed as a message.
	Draft string
	// ToolProgress maps a dangling call's ID to the last partial output it
	// reported, so its interrupted note can say how far it got.
	ToolProgress map[string]string
	// ToolOutcomes are completed calls recovered from side records. They are
	// neither retried nor marked interrupted; the resume path materializes their
	// messages beside unresolved siblings in original source order.
	ToolOutcomes map[string]ToolOutcomeRecord
}

// RecoverSession turns a durable log into a conservative resume plan. It reduces
// the log, then — under the (default) mark_interrupted policy — detects a turn
// that crashed mid-flight: an assistant message whose tool calls have no matching
// tool result. Retry-safe calls are queued for re-run; the rest are dropped so
// non-idempotent side effects are never silently repeated.
func RecoverSession(log []SessionEntry, tools *ToolSet, policy RecoveryPolicy) ResumePlan {
	rs := ReduceSession(log)
	plan := ResumePlan{
		Messages:        rs.Messages,
		Model:           rs.Model,
		ActiveTools:     rs.ActiveTools,
		DisabledTools:   rs.DisabledTools,
		Completed:       rs.Completed,
		Goal:            rs.Goal,
		RerunCompaction: rs.PendingCompaction,
		Inbox:           rs.Inbox,
		Draft:           rs.Draft,
		ToolProgress:    rs.ToolProgress,
		ToolOutcomes:    rs.ToolOutcomes,
	}
	if rs.Completed {
		return plan // a leaf exists: the run finished, nothing to recover
	}

	// Answers recorded while the run was parked (EntryAnswer) close their calls
	// before the retry/drop classification below: the answer IS the tool result,
	// so an answered call is neither re-run nor interrupted.
	answers := map[string]string{}
	for _, e := range ActivePath(log) {
		if e.Kind == EntryAnswer && e.CallID != "" {
			answers[e.CallID] = e.Answer
		}
	}
	if len(answers) > 0 {
		plan.Answers = answers
	}

	// Find dangling tool calls: those issued by an assistant message that never
	// received a tool result.
	satisfied := map[string]bool{}
	for _, m := range rs.Messages {
		if m.Role == RoleTool && m.ToolCallID != "" {
			satisfied[m.ToolCallID] = true
		}
	}
	for _, m := range rs.Messages {
		if m.Role != RoleAssistant {
			continue
		}
		for _, c := range m.ToolCalls {
			if satisfied[c.ID] {
				continue
			}
			plan.Interrupted = true
			if _, answered := answers[c.ID]; answered {
				continue // closed by its EntryAnswer at stitch time
			}
			if _, completed := rs.ToolOutcomes[c.ID]; completed {
				continue // physical effect completed; resume materializes its journaled result
			}
			if isRetrySafe(tools, c) {
				plan.RetryCalls = append(plan.RetryCalls, c)
			} else {
				plan.DroppedCalls = append(plan.DroppedCalls, c)
			}
		}
	}
	// A run that stopped without a leaf but with no dangling call is still
	// considered interrupted (it never reached a final answer).
	if !plan.Interrupted && len(rs.Messages) > 0 {
		plan.Interrupted = true
	}
	return plan
}

// interruptedCallNote is the synthesized tool result that closes a dangling
// call in a recovered transcript, shared by CloseDanglingCalls and the drive-
// level resume path so the model always sees the same wording.
const interruptedCallNote = "[interrupted: this tool call did not complete before the run was suspended and was not re-run automatically]"

// CloseDanglingCalls returns a transcript in which every assistant tool call
// that never received a result is satisfied by a synthesized interrupted-note
// tool message. Providers reject a history with an unanswered tool call, so
// this is what makes a recovered transcript replayable. Already-satisfied calls
// and non-assistant messages pass through untouched, in order.
func CloseDanglingCalls(messages []Message) []Message {
	satisfied := map[string]bool{}
	for _, m := range messages {
		if m.Role == RoleTool && m.ToolCallID != "" {
			satisfied[m.ToolCallID] = true
		}
	}
	out := make([]Message, 0, len(messages))
	for _, m := range messages {
		out = append(out, m)
		if m.Role != RoleAssistant {
			continue
		}

		for _, c := range m.ToolCalls {
			if satisfied[c.ID] {
				continue
			}
			out = append(out, Message{
				Role:       RoleTool,
				ToolCallID: c.ID,
				Name:       c.Name,
				Content:    interruptedCallNote,
			})
			satisfied[c.ID] = true
		}
	}
	return out
}

// PendingQuestion returns the newest parked call still awaiting an answer: an
// EntryQuestion with no matching EntryAnswer, or a settled native parked effect
// with no EntryPiAnswer. Native IDs identify physical effects, not provider calls.
// For a delegated ask, the ID belongs to this parent and the question comes
// from the child's routed receipt; the original spawn arguments stay in audit.
// The second return is the
// question's validated arguments; the third reports whether one was found.
// Consumers (the answer route, the reattach read) use it to render or resolve
// the open question without knowing which tool asked it.
func PendingQuestion(log []SessionEntry) (callID string, question json.RawMessage, found bool) {
	answered := map[string]bool{}
	for _, e := range log {
		if e.Kind == EntryAnswer || e.Kind == EntryPiAnswer {
			answered[e.CallID] = true
		}
	}
	for i := len(log) - 1; i >= 0; i-- {
		e := log[i]
		if id, question, native := PiQuestionFromEntry(e); native && !answered[id] {
			return id, question, true
		}
		if e.Kind == EntryQuestion && !answered[e.CallID] {
			return e.CallID, e.Question, true
		}
	}
	return "", nil, false
}

// The run's goal as durable STATE.
//
// The loop owns the goal only as a fact about the run: it writes the condition
// to the durable log once, and recovers it when a crashed run resumes. What to
// DO about an unmet goal — the completion protocol, the sentinel, the nudge, the
// stall breaker — is policy, and policy lives in agentcore/plugins/goal as a
// StopInterceptor. That split is deepseek-harness's: dsh-goal is an
// event-sourced service, and continuation is a separate consumer package.
//
// The state stays here for one reason: only the loop may write the durable log.
// A plugin that persisted its own gate condition would be a second writer to the
// record resume depends on, and "model-visible means logged" would stop being a
// property the core can guarantee.

// goalFromLog recovers the goal-gate condition from a durable log, so a crashed
// goal-gated run comes back gated even when the resuming caller cannot re-supply
// the condition.
//
// The fold matches RecoverSession's: the last EntryGoal wins, and EntryLeaf
// clears it — the goal belongs to the run that finished, so a later run chained
// onto the same log (a chat continuation) must not inherit its gate.
func goalFromLog(entries []SessionEntry) string {
	goal := ""
	for _, e := range entries {
		switch e.Kind {
		case EntryGoal:
			goal = e.Goal
		case EntryLeaf:
			goal = ""
		}
	}
	return goal
}
