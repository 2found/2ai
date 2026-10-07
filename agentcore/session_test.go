package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/2found/2ai/ai"
)

// memSessionStore is an in-memory append-only SessionStore for tests.
type memSessionStore struct {
	mu  sync.Mutex
	log map[string][]SessionEntry
}

func newMemSessionStore() *memSessionStore {
	return &memSessionStore{log: map[string][]SessionEntry{}}
}

func (m *memSessionStore) Append(_ context.Context, id string, e SessionEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	e.Seq = len(m.log[id])
	m.log[id] = append(m.log[id], e)
	return nil
}

func (m *memSessionStore) Log(_ context.Context, id string) ([]SessionEntry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]SessionEntry, len(m.log[id]))
	copy(out, m.log[id])
	return out, nil
}

// retrySafeProbe is a retry-safe (idempotent) tool.
type retrySafeProbe struct{}

func (retrySafeProbe) Name() string { return "safe_read" }
func (retrySafeProbe) Schema() ToolSchema {
	return ToolSchema{Name: "safe_read", Description: "idempotent read", Parameters: map[string]any{"type": "object"}}
}
func (retrySafeProbe) Run(context.Context, string) (string, error) { return "ok", nil }
func (retrySafeProbe) RetrySafe() bool                             { return true }

// TestRecoverInterruptedAfterToolResult simulates a crash: the log ends at a
// tool result with no following assistant turn and no leaf. Recovery resumes to
// the same history and marks the turn interrupted.
func TestRecoverInterruptedAfterToolResult(t *testing.T) {
	log := []SessionEntry{
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "do it"}},
		{Kind: EntryMessage, Turn: 1, Message: &Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "safe_read"}}}},
		{Kind: EntryMessage, Turn: 1, Message: &Message{Role: RoleTool, ToolCallID: "c1", Name: "safe_read", Content: "ok"}},
		// crash here: no next assistant turn, no leaf.
	}
	tools := NewToolSet(retrySafeProbe{})
	plan := RecoverSession(log, tools, RecoveryMarkInterrupted)
	if plan.Completed {
		t.Fatalf("run did not complete; Completed must be false")
	}
	if !plan.Interrupted {
		t.Fatalf("interrupted run must be flagged: %+v", plan)
	}
	if len(plan.Messages) != 3 {
		t.Fatalf("should resume to the same 3-message leaf, got %d", len(plan.Messages))
	}
	// The tool call was satisfied (result present), so nothing dangling to retry.
	if len(plan.RetryCalls) != 0 || len(plan.DroppedCalls) != 0 {
		t.Fatalf("satisfied call should not be re-run: retry=%v dropped=%v", plan.RetryCalls, plan.DroppedCalls)
	}
}

// TestRecoverDanglingCallRetrySafety verifies a crash between an assistant tool
// call and its result re-runs only retry-safe tools; non-idempotent tools are
// dropped (left for the model), never silently re-executed.
func TestRecoverDanglingCallRetrySafety(t *testing.T) {
	log := []SessionEntry{
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "go"}},
		{Kind: EntryMessage, Turn: 1, Message: &Message{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "c1", Name: "safe_read"}, // retry-safe
			{ID: "c2", Name: "noop"},      // not retry-safe (noopTool has no RetrySafe)
		}}},
		// crash: neither tool produced a result.
	}
	tools := NewToolSet(retrySafeProbe{}, noopTool{})
	plan := RecoverSession(log, tools, RecoveryMarkInterrupted)
	if !plan.Interrupted {
		t.Fatalf("dangling calls must mark the run interrupted")
	}
	if len(plan.RetryCalls) != 1 || plan.RetryCalls[0].Name != "safe_read" {
		t.Fatalf("retry-safe call should be queued: %+v", plan.RetryCalls)
	}
	if len(plan.DroppedCalls) != 1 || plan.DroppedCalls[0].Name != "noop" {
		t.Fatalf("non-idempotent call must be dropped, not retried: %+v", plan.DroppedCalls)
	}
}

// TestRecoverUnfinishedCompactionReRuns verifies a compaction start with no
// completion entry sets RerunCompaction.
func TestRecoverUnfinishedCompactionReRuns(t *testing.T) {
	log := []SessionEntry{
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "x"}},
		{Kind: EntryCompaction, Turn: 2}, // started, never completed
	}
	plan := RecoverSession(log, nil, RecoveryMarkInterrupted)
	if !plan.RerunCompaction {
		t.Fatalf("unfinished compaction must set RerunCompaction: %+v", plan)
	}

	// With a completion entry it is considered done.
	log = append(log, SessionEntry{Kind: EntryCompaction, Turn: 2, Final: true})
	if RecoverSession(log, nil, RecoveryMarkInterrupted).RerunCompaction {
		t.Fatalf("completed compaction must not re-run")
	}
}

// TestReduceSessionReconstructsDisabledTools verifies the circuit breaker's
// disable records reduce into a deduplicated DisabledTools set on both the
// reduced state and the resume plan.
func TestReduceSessionReconstructsDisabledTools(t *testing.T) {
	log := []SessionEntry{
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "go"}},
		{Kind: EntryToolDisabled, Turn: 2, Tool: "flaky"},
		{Kind: EntryToolDisabled, Turn: 2, Tool: "flaky"}, // duplicate: must fold to one
		{Kind: EntryToolDisabled, Turn: 3, Tool: "other"},
	}
	rs := ReduceSession(log)
	if len(rs.DisabledTools) != 2 || rs.DisabledTools[0] != "flaky" || rs.DisabledTools[1] != "other" {
		t.Fatalf("reduced disabled set wrong: %v", rs.DisabledTools)
	}
	plan := RecoverSession(log, nil, RecoveryMarkInterrupted)
	if len(plan.DisabledTools) != 2 {
		t.Fatalf("resume plan should carry the disabled set: %v", plan.DisabledTools)
	}
}

// TestCompactionUsageAccounted verifies the compaction summarization call's
// spend is folded into the run's usage and stamped on the durable completion
// entry — compaction is a real provider call, not free.
func TestCompactionUsageAccounted(t *testing.T) {
	// Each turn produces a bulky tool result; the tiny context budget forces a
	// summarization call partway through, which consumes its own usage. The
	// final response is the answer.
	call := 0
	var totalIn, totalCost float64
	stream := func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		call++
		if strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
			totalIn += 100
			totalCost += 0.01
			return nativeEmit(&ai.Message{Role: "assistant", StopReason: "stop", Usage: &ai.Usage{Input: 100, Output: 20, Cost: ai.UsageCost{Total: 0.01}},
				Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "earlier work summarized"})})(ctx, model, view, options)
		}
		totalIn += 10
		totalCost += 0.001
		if call >= 4 {
			return nativeEmit(&ai.Message{Role: "assistant", StopReason: "stop", Usage: &ai.Usage{Input: 10, Output: 5, Cost: ai.UsageCost{Total: 0.001}},
				Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})})(ctx, model, view, options)
		}
		return nativeEmit(&ai.Message{Role: "assistant", StopReason: "toolUse", Usage: &ai.Usage{Input: 10, Output: 5, Cost: ai.UsageCost{Total: 0.001}}, Content: ai.BlockContent(
			ai.ContentBlock{Type: "toolCall", ID: fmt.Sprintf("c%d", call), Name: "big", Arguments: json.RawMessage(`{}`)},
		)})(ctx, model, view, options)
	}
	limits := DefaultLimits()
	limits.MaxContextTokens = 5000
	limits.MaxToolResultLen = 64 * 1024
	agent, err := New(Config{
		NativeProvider: nativeCandidate("test", stream),
		Model:          "test",
		Tools:          NewToolSet(bigResultTool{}),
		Policy:         NewAllowList("big"),
		Limits:         &limits,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "start "+bigText(8000))
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if call < 4 {
		t.Fatalf("compaction never ran a summarization call (provider calls = %d)", call)
	}
	// Run usage must include the compaction call's spend.
	if diff := res.Usage.CostUSD - totalCost; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("run cost %.4f, want %.4f (compaction spend folded in)", res.Usage.CostUSD, totalCost)
	}
	if res.Usage.InputTokens != int(totalIn) {
		t.Fatalf("run input tokens %d, want %d", res.Usage.InputTokens, int(totalIn))
	}
}

// bigResultTool returns an oversized result so the context estimate crosses the
// compaction threshold within one turn.
type bigResultTool struct{}

func (bigResultTool) Name() string { return "big" }
func (bigResultTool) Schema() ToolSchema {
	return ToolSchema{Name: "big", Description: "returns a big result", Parameters: map[string]any{"type": "object"}}
}
func (bigResultTool) Run(context.Context, string) (string, error) { return bigText(8000), nil }

// TestGoalFromLogClearsAtLeaf pins the rule that keeps a chat continuation from
// inheriting a finished run's gate: the goal belongs to the run that recorded
// it, and EntryLeaf ends that run.
//
// Without this, a second prompt on the same session would resume gated by a
// goal the user already satisfied, and the model would be told to keep working
// on it forever.
func TestGoalFromLogClearsAtLeaf(t *testing.T) {
	cases := []struct {
		name    string
		entries []SessionEntry
		want    string
	}{
		{"no entries", nil, ""},
		{"goal only", []SessionEntry{{Kind: EntryGoal, Goal: "ship it"}}, "ship it"},
		{
			"last goal wins",
			[]SessionEntry{{Kind: EntryGoal, Goal: "first"}, {Kind: EntryGoal, Goal: "second"}},
			"second",
		},
		{
			"leaf clears the finished run's goal",
			[]SessionEntry{{Kind: EntryGoal, Goal: "ship it"}, {Kind: EntryLeaf}},
			"",
		},
		{
			"a run chained after a leaf carries its own goal",
			[]SessionEntry{
				{Kind: EntryGoal, Goal: "old"},
				{Kind: EntryLeaf},
				{Kind: EntryGoal, Goal: "new"},
			},
			"new",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := goalFromLog(tc.entries); got != tc.want {
				t.Fatalf("goalFromLog = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestGoalFromLogMatchesRecoverSession keeps the two folds from drifting: the
// loop resolves the goal from goalFromLog before extensions begin, while
// RecoverSession derives it again for the recovery plan. If they ever disagree,
// a resumed run would be gated by one condition and reason about another.
func TestGoalFromLogMatchesRecoverSession(t *testing.T) {
	entries := []SessionEntry{
		{Kind: EntryGoal, Goal: "old"},
		{Kind: EntryLeaf},
		{Kind: EntryGoal, Goal: "current"},
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "go"}},
	}
	plan := RecoverSession(entries, NewToolSet(), RecoveryMarkInterrupted)
	if got := goalFromLog(entries); got != plan.Goal {
		t.Fatalf("goalFromLog = %q but RecoverSession = %q", got, plan.Goal)
	}
}

// TestSideRecordsConcurrentWrites ensures direct appends from tool goroutines
// and the stream frame writer do not race with the save-point flush.
func TestSideRecordsConcurrentWrites(t *testing.T) {
	ctx := context.Background()
	store := newMemSessionStore()
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_ = store.Append(ctx, "concurrent", SessionEntry{
				Kind:    EntryToolProgress,
				CallID:  "c",
				Content: "progress",
			})
		}(i)
	}
	wg.Wait()
	log, _ := store.Log(ctx, "concurrent")
	if len(log) != 20 {
		t.Fatalf("expected 20 entries, got %d", len(log))
	}
}

// TestFoldSeesFramesBehindAFlushedTurn covers the ordering a graceful
// interruption produces: mid-turn side records land first, then the deferred
// flush commits the turn's buffered chain entries AFTER them. A tail scan that
// stops at the first chain entry loses the draft; the fold must bound the
// in-flight window by the last SETTLING entry (assistant/tool message), not
// the last chain entry.
func TestFoldSeesFramesBehindAFlushedTurn(t *testing.T) {
	ctx := context.Background()
	store := newMemSessionStore()
	user := Message{Role: RoleUser, Content: "go"}
	steer := Message{Role: RoleUser, Content: "mid-turn steer"}
	for _, e := range []SessionEntry{
		{Kind: EntryMessage, Turn: 1, Message: &user},
		// Turn 2 streams a draft and tool progress, delivers a steer (buffered
		// chain entries), then is interrupted — the deferred flush commits the
		// buffered entries AFTER the side records.
		{Kind: EntryAssistantFrame, Turn: 2, Content: "half-written answer"},
		{Kind: EntryToolProgress, Turn: 2, CallID: "c1", Content: "tool got this far"},
		{Kind: EntryMessage, Turn: 2, Message: &steer},
		{Kind: EntryInboxDone, Turn: 2, Target: "inbox-9"},
	} {
		if err := store.Append(ctx, "s", e); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	log, _ := store.Log(ctx, "s")
	rs := ReduceSession(log)
	if rs.Draft != "half-written answer" {
		t.Fatalf("draft behind flushed chain entries must be recovered, got %q", rs.Draft)
	}
	if rs.ToolProgress["c1"] != "tool got this far" {
		t.Fatalf("tool progress behind flushed chain entries must be recovered, got %v", rs.ToolProgress)
	}
}

// TestInboxDoesNotLeakAcrossRewind covers both directions of branch leakage:
// an intent settled on an abandoned branch is pending again after the rewind
// (its delivered message is gone), and an intent queued on the abandoned
// branch is not delivered into the new one.
func TestInboxDoesNotLeakAcrossRewind(t *testing.T) {
	ctx := context.Background()
	store := newMemSessionStore()
	q := Message{Role: RoleUser, Content: "question"}
	a1 := Message{Role: RoleAssistant, Content: "first answer"}
	steer := Message{Role: RoleUser, Content: "delivered steer"}
	a2 := Message{Role: RoleAssistant, Content: "steered answer"}
	stale := Message{Role: RoleUser, Content: "queued on the old branch"}
	entries := []SessionEntry{
		{Kind: EntryMessage, Turn: 1, Message: &q},                               // seq 0
		{Kind: EntryMessage, Turn: 1, Message: &a1},                              // seq 1
		{Kind: EntryInbox, ID: "i-delivered", Lane: InboxSteer, Message: &steer}, // seq 2, anchored to seq 1
		{Kind: EntryMessage, Turn: 2, Message: &steer},                           // seq 3 (delivered)
		{Kind: EntryInboxDone, Turn: 2, Target: "i-delivered"},                   // seq 4
		{Kind: EntryMessage, Turn: 2, Message: &a2},                              // seq 5
		{Kind: EntryInbox, ID: "i-stale", Lane: InboxSteer, Message: &stale},     // seq 6, anchored to seq 5
	}
	for _, e := range entries {
		if err := store.Append(ctx, "s", e); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	// Rewind to just after the first answer: the delivered steer, its
	// settlement, the second answer, and the stale intent all sit on the
	// abandoned branch.
	if _, err := Rewind(ctx, store, "s", "#1", BranchOptions{}); err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	log, _ := store.Log(ctx, "s")
	rs := ReduceSession(log)

	var ids []string
	for _, it := range rs.Inbox {
		ids = append(ids, it.ID)
	}
	if len(ids) != 1 || ids[0] != "i-delivered" {
		t.Fatalf("after rewind the rewound-away delivery must be pending again and the stale intent dropped, got %v", ids)
	}
}
