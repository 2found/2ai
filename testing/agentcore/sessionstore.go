// Package agentcoretest contains reusable conformance checks for agentcore
// boundary implementations. Keeping the checks outside the kernel lets every
// host backend prove the same semantics without adding test machinery to the
// runtime package itself.
package agentcoretest

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/engine"
	nativehost "github.com/2found/2ai/agentcore/host"
	"github.com/2found/2ai/ai"
)

// SessionStoreHarness supplies one backend and a factory for fresh, valid
// session IDs. Durable backends may use NewSessionID to create required parent
// rows; in-memory backends can simply return unique strings.
type SessionStoreHarness struct {
	Store        agentcore.SessionStore
	NewSessionID func(t *testing.T) string
}

// RunSessionStoreConformance exercises the observable SessionStore contract,
// including optional batch and window capabilities. Backends should invoke it
// unchanged so local and server persistence cannot drift semantically.
func RunSessionStoreConformance(t *testing.T, harness SessionStoreHarness) {
	t.Helper()
	if harness.Store == nil {
		t.Fatal("SessionStoreHarness.Store is nil")
	}
	if harness.NewSessionID == nil {
		t.Fatal("SessionStoreHarness.NewSessionID is nil")
	}

	t.Run("snapshot_and_typed_round_trip", func(t *testing.T) {
		id := harness.NewSessionID(t)
		ctx := context.Background()
		created := time.Date(2026, 9, 18, 12, 30, 0, 123_000_000, time.UTC)
		message := agentcore.Message{
			Role: agentcore.RoleAssistant, Content: "calling", Directive: true,
			ToolCalls: []agentcore.ToolCall{{ID: "call-1", Name: "read", Arguments: `{"path":"README.md"}`}},
			ReasoningBlocks: []agentcore.ReasoningBlock{{
				Type: agentcore.ReasoningBlockThinking, Text: "opaque", Signature: "sig", ReplayScope: "anthropic:scope",
			}},
			Usage: &agentcore.Usage{InputTokens: 11, OutputTokens: 7, CacheReadTokens: 3, CostUSD: 0.125},
		}
		first := agentcore.SessionEntry{
			Kind: agentcore.EntryMessage, ID: "assistant-1", ParentID: "root", Turn: 2,
			Message: &message, Tools: []string{"read", "write"}, Question: []byte(`{"prompt":"continue?"}`),
			CreatedAt: created,
		}
		outcome := agentcore.SessionEntry{
			Kind: agentcore.EntryToolOutcome, ID: "outcome-1", ParentID: "assistant-1", Turn: 2, CallID: "call-1",
			Outcome: &agentcore.ToolOutcomeRecord{
				Message: agentcore.Message{Role: agentcore.RoleTool, ToolCallID: "call-1", Name: "read", Content: "contents"},
				Trace: agentcore.ToolTrace{
					CallID: "call-1", Tool: "read", Args: `{"path":"README.md"}`, Allowed: true,
					ResultMeta: "17 bytes", LatencyMS: 9, IdempotencyKey: "idem-1", SpillLocator: "spill://one",
				},
				Extra: []agentcore.Message{{Role: agentcore.RoleSystem, Content: "extra"}}, Terminate: true, Executed: true,
			},
			CreatedAt: created.Add(time.Second),
		}
		if err := harness.Store.Append(ctx, id, first); err != nil {
			t.Fatalf("Append(first): %v", err)
		}
		if err := harness.Store.Append(ctx, id, outcome); err != nil {
			t.Fatalf("Append(outcome): %v", err)
		}

		// A store owns an immutable snapshot after Append, not aliases into the
		// caller's structs.
		first.Message.Content = "mutated at source"
		first.Message.ToolCalls[0].Name = "mutated"
		first.Message.ReasoningBlocks[0].Signature = "mutated"
		first.Message.Usage.InputTokens = 999
		first.Tools[0] = "mutated"
		first.Question[0] = 'X'
		outcome.Outcome.Message.Content = "mutated outcome"
		outcome.Outcome.Extra[0].Content = "mutated extra"

		got := mustLog(t, harness.Store, id)
		assertTypedEntries(t, got, created)

		// Reads are snapshots too: mutating one result must not rewrite the log.
		got[0].Message.Content = "mutated through read"
		got[0].Message.ToolCalls[0].Name = "mutated through read"
		got[0].Message.ReasoningBlocks[0].Signature = "mutated through read"
		got[1].Outcome.Extra[0].Content = "mutated through read"
		assertTypedEntries(t, mustLog(t, harness.Store, id), created)
	})

	t.Run("session_isolation", func(t *testing.T) {
		ctx := context.Background()
		left, right := harness.NewSessionID(t), harness.NewSessionID(t)
		appendMessage(t, ctx, harness.Store, left, "left")
		appendMessage(t, ctx, harness.Store, right, "right")
		if got := mustLog(t, harness.Store, left); len(got) != 1 || got[0].Message.Content != "left" {
			t.Fatalf("left session leaked or lost data: %+v", got)
		}
		if got := mustLog(t, harness.Store, right); len(got) != 1 || got[0].Message.Content != "right" {
			t.Fatalf("right session leaked or lost data: %+v", got)
		}
	})

	t.Run("optional_lease_serializes_one_session", func(t *testing.T) {
		leases, ok := harness.Store.(agentcore.SessionLeaseStore)
		if !ok {
			t.Skip("backend does not implement SessionLeaseStore")
		}
		id := harness.NewSessionID(t)
		other := harness.NewSessionID(t)
		_, release, err := leases.AcquireSessionLease(context.Background(), id)
		if err != nil {
			t.Fatalf("first acquire: %v", err)
		}
		defer func() { _ = release() }()

		// A different durable session must not queue behind this one.
		_, releaseOther, err := leases.AcquireSessionLease(context.Background(), other)
		if err != nil {
			t.Fatalf("independent acquire: %v", err)
		}
		if err := releaseOther(); err != nil {
			t.Fatalf("independent release: %v", err)
		}

		waitCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		if _, _, err := leases.AcquireSessionLease(waitCtx, id); err == nil {
			t.Fatal("contending owner acquired a live session")
		}
		if err := release(); err != nil {
			t.Fatalf("first release: %v", err)
		}
		_, releaseAgain, err := leases.AcquireSessionLease(context.Background(), id)
		if err != nil {
			t.Fatalf("acquire after release: %v", err)
		}
		if err := releaseAgain(); err != nil {
			t.Fatalf("second release: %v", err)
		}
	})

	t.Run("concurrent_append_has_total_order", func(t *testing.T) {
		const writers = 32
		id := harness.NewSessionID(t)
		ctx := context.Background()
		errs := make(chan error, writers)
		var wg sync.WaitGroup
		for i := 0; i < writers; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				content := fmt.Sprintf("writer-%02d", i)
				msg := agentcore.Message{Role: agentcore.RoleUser, Content: content}
				errs <- harness.Store.Append(ctx, id, agentcore.SessionEntry{Kind: agentcore.EntryMessage, Message: &msg})
			}(i)
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Fatalf("concurrent Append: %v", err)
			}
		}
		log := mustLog(t, harness.Store, id)
		if len(log) != writers {
			t.Fatalf("len(Log) = %d, want %d", len(log), writers)
		}
		seen := make(map[string]bool, writers)
		assertStrictlyIncreasingSeq(t, log)
		for _, entry := range log {
			if entry.Message == nil {
				t.Fatalf("entry %d has no message", entry.Seq)
			}
			seen[entry.Message.Content] = true
		}
		for i := 0; i < writers; i++ {
			if !seen[fmt.Sprintf("writer-%02d", i)] {
				t.Errorf("writer-%02d was lost", i)
			}
		}
	})

	t.Run("batch_is_atomic_and_contiguous", func(t *testing.T) {
		batches, ok := harness.Store.(agentcore.SessionBatchStore)
		if !ok {
			t.Skip("backend does not implement SessionBatchStore")
		}
		const batchCount, batchSize, singles = 16, 3, 16
		id := harness.NewSessionID(t)
		ctx := context.Background()
		if err := batches.AppendBatch(ctx, id, nil); err != nil {
			t.Fatalf("empty AppendBatch: %v", err)
		}
		errs := make(chan error, batchCount+singles)
		var wg sync.WaitGroup
		for batch := 0; batch < batchCount; batch++ {
			wg.Add(1)
			go func(batch int) {
				defer wg.Done()
				entries := make([]agentcore.SessionEntry, batchSize)
				for item := range entries {
					content := fmt.Sprintf("batch-%02d-%d", batch, item)
					msg := agentcore.Message{Role: agentcore.RoleUser, Content: content}
					entries[item] = agentcore.SessionEntry{Kind: agentcore.EntryMessage, Message: &msg}
				}
				errs <- batches.AppendBatch(ctx, id, entries)
			}(batch)
		}
		for single := 0; single < singles; single++ {
			wg.Add(1)
			go func(single int) {
				defer wg.Done()
				content := fmt.Sprintf("single-%02d", single)
				msg := agentcore.Message{Role: agentcore.RoleUser, Content: content}
				errs <- harness.Store.Append(ctx, id, agentcore.SessionEntry{Kind: agentcore.EntryMessage, Message: &msg})
			}(single)
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Fatalf("concurrent batch/append: %v", err)
			}
		}

		log := mustLog(t, harness.Store, id)
		if want := batchCount*batchSize + singles; len(log) != want {
			t.Fatalf("len(Log) = %d, want %d", len(log), want)
		}
		assertStrictlyIncreasingSeq(t, log)
		positions := make(map[string]int, len(log))
		for position, entry := range log {
			if entry.Message == nil {
				t.Fatalf("entry at position %d has no message", position)
			}
			positions[entry.Message.Content] = position
		}
		for batch := 0; batch < batchCount; batch++ {
			start := positions[fmt.Sprintf("batch-%02d-0", batch)]
			for item := 1; item < batchSize; item++ {
				if got := positions[fmt.Sprintf("batch-%02d-%d", batch, item)]; got != start+item {
					t.Errorf("batch %d was interleaved: item %d at %d, want %d", batch, item, got, start+item)
				}
			}
		}
	})

	t.Run("branch_and_side_record_semantics", func(t *testing.T) {
		id := harness.NewSessionID(t)
		ctx := context.Background()
		user := agentcore.Message{Role: agentcore.RoleUser, Content: "root"}
		assistant := agentcore.Message{Role: agentcore.RoleAssistant, Content: "", ToolCalls: []agentcore.ToolCall{{ID: "call-branch", Name: "write", Arguments: `{}`}}}
		entries := []agentcore.SessionEntry{
			{Kind: agentcore.EntryMessage, ID: "root", Turn: 1, Message: &user},
			{Kind: agentcore.EntryMessage, ID: "calls", ParentID: "root", Turn: 1, Message: &assistant},
			{Kind: agentcore.EntryToolOutcome, ID: "outcome", ParentID: "calls", Turn: 1, CallID: "call-branch", Outcome: &agentcore.ToolOutcomeRecord{
				Message: agentcore.Message{Role: agentcore.RoleTool, ToolCallID: "call-branch", Name: "write", Content: "done"},
				Trace:   agentcore.ToolTrace{CallID: "call-branch", Tool: "write", Args: `{}`, Allowed: true}, Executed: true,
			}},
			{Kind: agentcore.EntryLeafMove, Target: "root"},
		}
		appendBatchOrEntries(t, ctx, harness.Store, id, entries)
		log := mustLog(t, harness.Store, id)
		state := agentcore.ReduceSession(log)
		if len(state.Messages) != 1 || state.Messages[0].Content != "root" {
			t.Fatalf("active branch messages = %+v, want only root", state.Messages)
		}
		if state.ToolOutcomes != nil {
			t.Fatalf("abandoned branch outcome remained active: %+v", state.ToolOutcomes)
		}
	})

	t.Run("completed_tool_outcome_is_recoverable", func(t *testing.T) {
		id := harness.NewSessionID(t)
		ctx := context.Background()
		user := agentcore.Message{Role: agentcore.RoleUser, Content: "do it"}
		assistant := agentcore.Message{Role: agentcore.RoleAssistant, ToolCalls: []agentcore.ToolCall{{ID: "call-recover", Name: "write", Arguments: `{}`}}}
		entries := []agentcore.SessionEntry{
			{Kind: agentcore.EntryMessage, ID: "recover-root", Message: &user},
			{Kind: agentcore.EntryMessage, ID: "recover-call", ParentID: "recover-root", Message: &assistant},
			{Kind: agentcore.EntryToolOutcome, ParentID: "recover-call", CallID: "call-recover", Outcome: &agentcore.ToolOutcomeRecord{
				Message: agentcore.Message{Role: agentcore.RoleTool, ToolCallID: "call-recover", Name: "write", Content: "done"},
				Trace:   agentcore.ToolTrace{CallID: "call-recover", Tool: "write", Allowed: true}, Executed: true,
			}},
		}
		appendBatchOrEntries(t, ctx, harness.Store, id, entries)
		plan := agentcore.RecoverSession(mustLog(t, harness.Store, id), nil, agentcore.RecoveryMarkInterrupted)
		if plan.ToolOutcomes["call-recover"].Message.Content != "done" {
			t.Fatalf("completed outcome not recovered: %+v", plan.ToolOutcomes)
		}
		if len(plan.RetryCalls) != 0 || len(plan.DroppedCalls) != 0 {
			t.Fatalf("completed physical call was retried/dropped: retry=%+v dropped=%+v", plan.RetryCalls, plan.DroppedCalls)
		}
	})

	t.Run("native_reverse_completion_recovers_once_in_source_order", func(t *testing.T) {
		id := harness.NewSessionID(t)
		ctx := context.Background()
		initial := `{"model":{"id":"test"},"messages":[{"role":"user","content":"run both","timestamp":1},{"role":"assistant","content":[{"type":"toolCall","id":"slow-call","name":"slow","arguments":{}},{"type":"toolCall","id":"fast-call","name":"fast","arguments":{}}],"stopReason":"toolUse","timestamp":2}]}`
		entries := []agentcore.SessionEntry{
			{Kind: agentcore.EntryPiState, Content: initial},
			{Kind: agentcore.EntryPiEffectStart, CallID: "slow-call", Content: `{"effectId":"slow-effect"}`},
			{Kind: agentcore.EntryPiEffectStart, CallID: "fast-call", Content: `{"effectId":"fast-effect"}`},
			// Physical completion order is independent of native message order.
			{Kind: agentcore.EntryPiEffectDone, CallID: "fast-call", Content: `{"effectId":"fast-effect"}`},
			{Kind: agentcore.EntryPiEffectDone, CallID: "slow-call", Content: `{"effectId":"slow-effect"}`},
		}
		results := []string{
			`{"role":"toolResult","toolCallId":"slow-call","toolName":"slow","content":[{"type":"text","text":"slow-result"}],"timestamp":3,"details":{"opaque":"9007199254740993","signature":"keep-slow"},"isError":false}`,
			`{"role":"toolResult","toolCallId":"fast-call","toolName":"fast","content":[{"type":"text","text":"fast-result"}],"timestamp":4,"details":{"signature":"keep-fast"},"isError":false}`,
		}
		for i, result := range results {
			entries = append(entries, agentcore.SessionEntry{Kind: agentcore.EntryPiEvent, Content: `{"type":"message_start","message":` + result + `}`})
			// Crash after the final immutable message_start, before message_end.
			if i == 0 {
				entries = append(entries, agentcore.SessionEntry{Kind: agentcore.EntryPiEvent, Content: `{"type":"message_end","message":` + result + `}`})
			}
		}
		for _, entry := range entries {
			if err := harness.Store.Append(ctx, id, entry); err != nil {
				t.Fatal(err)
			}
		}
		state, err := agentcore.RecoverNativeTranscript(mustLog(t, harness.Store, id))
		if err != nil {
			t.Fatal(err)
		}
		var restored struct {
			Model    json.RawMessage
			Messages []*ai.Message
		}
		if err := json.Unmarshal(state, &restored); err != nil {
			t.Fatal(err)
		}
		requests := 0
		stream := func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, opts map[string]any) (*ai.AssistantMessageEventStream, error) {
			requests++
			messages := view.Messages()
			if len(messages) != 4 {
				t.Errorf("native replay has %d messages, want 4", len(messages))
				return nil, fmt.Errorf("unexpected native transcript length")
			}
			for i, expected := range results {
				raw, err := json.Marshal(messages[i+2])
				if err != nil || !nativehost.SameJSON(raw, json.RawMessage(expected)) {
					t.Errorf("native result %d changed: %s, %v", i, raw, err)
					return nil, fmt.Errorf("native transcript changed")
				}
			}
			return ai.ScriptedStream(ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "continued"}), StopReason: "stop"})(ctx, model, view, opts)
		}
		agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: restored.Model, Messages: restored.Messages}, AgentConfig: engine.AgentConfig{StreamFn: stream}})
		if err != nil {
			t.Fatal(err)
		}
		if err := agent.Continue(ctx); err != nil {
			t.Fatal(err)
		}
		if requests != 1 {
			t.Fatalf("continuation requests = %d, want 1", requests)
		}
		checkpoint, err := json.Marshal(agent.State())
		if err != nil {
			t.Fatal(err)
		}
		if err := harness.Store.Append(ctx, id, agentcore.SessionEntry{Kind: agentcore.EntryPiState, Content: string(checkpoint)}); err != nil {
			t.Fatal(err)
		}
		// Recovery is read-only and repeatable. A new checkpoint absorbs earlier
		// events without appending or replaying either settled physical effect.
		for range 2 {
			raw, err := agentcore.RecoverNativeTranscript(mustLog(t, harness.Store, id))
			if err != nil {
				t.Fatal(err)
			}
			var after struct{ Messages []json.RawMessage }
			if err := json.Unmarshal(raw, &after); err != nil || len(after.Messages) != 5 {
				t.Fatalf("checkpoint duplicated or lost a message: %s, %v", raw, err)
			}
			for i, expected := range results {
				if !nativehost.SameJSON(after.Messages[i+2], json.RawMessage(expected)) {
					t.Fatalf("checkpoint result %d changed: %s", i, after.Messages[i+2])
				}
			}
		}
		if got := len(mustLog(t, harness.Store, id)); got != len(entries)+1 {
			t.Fatalf("recovery mutated journal: %d entries, want %d", got, len(entries)+1)
		}
	})

	t.Run("checkpoint_window_matches_full_fold", func(t *testing.T) {
		windows, ok := harness.Store.(agentcore.SessionWindowStore)
		if !ok {
			t.Skip("backend does not implement SessionWindowStore")
		}
		id := harness.NewSessionID(t)
		ctx := context.Background()
		old := agentcore.Message{Role: agentcore.RoleUser, Content: "old"}
		tail := agentcore.Message{Role: agentcore.RoleAssistant, Content: "new"}
		entries := []agentcore.SessionEntry{
			{Kind: agentcore.EntryMessage, ID: "old", Turn: 1, Message: &old},
			{Kind: agentcore.EntryModelChange, ID: "old-model", ParentID: "old", Turn: 1, Model: "superseded"},
			{Kind: agentcore.EntryCompaction, ID: "checkpoint", ParentID: "old-model", Turn: 4, Final: true, Summary: "summary",
				Retained: []agentcore.Message{{Role: agentcore.RoleSystem, Content: "summary"}},
				State:    &agentcore.CheckpointState{Model: "model-v2", ActiveTools: []string{"read"}, DisabledTools: []string{"write"}, Goal: "finish"}},
			{Kind: agentcore.EntryMessage, ID: "tail", ParentID: "checkpoint", Turn: 5, Message: &tail},
		}
		appendBatchOrEntries(t, ctx, harness.Store, id, entries)
		seq, branched, err := windows.CheckpointSeq(ctx, id)
		if err != nil {
			t.Fatalf("CheckpointSeq: %v", err)
		}
		if seq <= 0 || branched {
			t.Fatalf("CheckpointSeq = (%d, %v), want positive/unbranched", seq, branched)
		}
		window, err := windows.LogFrom(ctx, id, seq)
		if err != nil {
			t.Fatalf("LogFrom: %v", err)
		}
		if len(window) == 0 || window[0].Seq != seq || window[0].Kind != agentcore.EntryCompaction {
			t.Fatalf("window starts incorrectly: %+v", window)
		}
		fullState := agentcore.ReduceSession(mustLog(t, harness.Store, id))
		windowState := agentcore.ReduceSession(window)
		if !reflect.DeepEqual(fullState, windowState) {
			t.Fatalf("window fold differs from full fold:\nfull=%+v\nwindow=%+v", fullState, windowState)
		}
	})

	t.Run("unsafe_windows_are_rejected", func(t *testing.T) {
		windows, ok := harness.Store.(agentcore.SessionWindowStore)
		if !ok {
			t.Skip("backend does not implement SessionWindowStore")
		}
		ctx := context.Background()

		t.Run("branch", func(t *testing.T) {
			id := harness.NewSessionID(t)
			root := agentcore.Message{Role: agentcore.RoleUser, Content: "root"}
			appendBatchOrEntries(t, ctx, harness.Store, id, []agentcore.SessionEntry{
				{Kind: agentcore.EntryMessage, ID: "root", Message: &root},
				{Kind: agentcore.EntryCompaction, ID: "cp", ParentID: "root", Final: true, Retained: []agentcore.Message{}, State: &agentcore.CheckpointState{}},
				{Kind: agentcore.EntryLeafMove, Target: "root"},
			})
			seq, branched, err := windows.CheckpointSeq(ctx, id)
			if err != nil {
				t.Fatalf("CheckpointSeq: %v", err)
			}
			if !branched {
				t.Fatalf("CheckpointSeq = (%d, false), want branched", seq)
			}
		})

		t.Run("unsettled_inbox", func(t *testing.T) {
			id := harness.NewSessionID(t)
			root := agentcore.Message{Role: agentcore.RoleUser, Content: "root"}
			queued := agentcore.Message{Role: agentcore.RoleUser, Content: "do not lose me"}
			appendBatchOrEntries(t, ctx, harness.Store, id, []agentcore.SessionEntry{
				{Kind: agentcore.EntryMessage, ID: "root", Message: &root},
				{Kind: agentcore.EntryInbox, ID: "inbox-1", Lane: agentcore.InboxSteer, Message: &queued},
				{Kind: agentcore.EntryCompaction, ID: "cp", ParentID: "root", Final: true, Retained: []agentcore.Message{}, State: &agentcore.CheckpointState{}},
			})
			seq, _, err := windows.CheckpointSeq(ctx, id)
			if err != nil {
				t.Fatalf("CheckpointSeq: %v", err)
			}
			if seq != 0 {
				t.Fatalf("CheckpointSeq = %d with pending pre-checkpoint inbox, want 0", seq)
			}
		})
	})
}

func appendMessage(t *testing.T, ctx context.Context, store agentcore.SessionStore, id, content string) {
	t.Helper()
	message := agentcore.Message{Role: agentcore.RoleUser, Content: content}
	if err := store.Append(ctx, id, agentcore.SessionEntry{Kind: agentcore.EntryMessage, Message: &message}); err != nil {
		t.Fatalf("Append(%q): %v", content, err)
	}
}

func appendBatchOrEntries(t *testing.T, ctx context.Context, store agentcore.SessionStore, id string, entries []agentcore.SessionEntry) {
	t.Helper()
	if batches, ok := store.(agentcore.SessionBatchStore); ok {
		if err := batches.AppendBatch(ctx, id, entries); err != nil {
			t.Fatalf("AppendBatch: %v", err)
		}
		return
	}
	for _, entry := range entries {
		if err := store.Append(ctx, id, entry); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
}

func mustLog(t *testing.T, store agentcore.SessionStore, id string) []agentcore.SessionEntry {
	t.Helper()
	log, err := store.Log(context.Background(), id)
	if err != nil {
		t.Fatalf("Log: %v", err)
	}
	return log
}

func assertStrictlyIncreasingSeq(t *testing.T, log []agentcore.SessionEntry) {
	t.Helper()
	for i := 1; i < len(log); i++ {
		if log[i].Seq <= log[i-1].Seq {
			t.Fatalf("sequence is not strictly increasing at %d: %d then %d", i, log[i-1].Seq, log[i].Seq)
		}
	}
}

func assertTypedEntries(t *testing.T, got []agentcore.SessionEntry, created time.Time) {
	t.Helper()
	if len(got) != 2 {
		t.Fatalf("len(Log) = %d, want 2", len(got))
	}
	assertStrictlyIncreasingSeq(t, got)
	first := got[0]
	if first.Kind != agentcore.EntryMessage || first.ID != "assistant-1" || first.ParentID != "root" || first.Turn != 2 || !first.CreatedAt.Equal(created) {
		t.Fatalf("message envelope did not round-trip: %+v", first)
	}
	if first.Message == nil || first.Message.Content != "calling" || len(first.Message.ToolCalls) != 1 || first.Message.ToolCalls[0].Name != "read" ||
		len(first.Message.ReasoningBlocks) != 1 || first.Message.ReasoningBlocks[0].Signature != "sig" ||
		first.Message.Usage == nil || first.Message.Usage.InputTokens != 11 {
		t.Fatalf("typed message did not round-trip: %+v", first.Message)
	}
	// JSONB backends may reformat raw JSON. The contract is the immutable
	// JSON value, including every field, rather than insignificant whitespace.
	var question any
	if !reflect.DeepEqual(first.Tools, []string{"read", "write"}) || json.Unmarshal(first.Question, &question) != nil || !reflect.DeepEqual(question, map[string]any{"prompt": "continue?"}) {
		t.Fatalf("slice/raw fields did not round-trip: tools=%v question=%s", first.Tools, first.Question)
	}
	second := got[1]
	if second.Kind != agentcore.EntryToolOutcome || second.CallID != "call-1" || second.Outcome == nil {
		t.Fatalf("outcome envelope did not round-trip: %+v", second)
	}
	if second.Outcome.Message.Content != "contents" || second.Outcome.Trace.SpillLocator != "spill://one" || len(second.Outcome.Extra) != 1 || second.Outcome.Extra[0].Content != "extra" || !second.Outcome.Terminate || !second.Outcome.Executed {
		t.Fatalf("tool outcome did not round-trip: %+v", second.Outcome)
	}
}
