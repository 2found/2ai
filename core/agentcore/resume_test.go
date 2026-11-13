package agentcore

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// The windowed resume rests on one claim: reading a SUFFIX of a log gives the
// same answer as reading all of it. Every test here attacks that claim, because
// a window that folds to a different state is not a slower resume — it is a run
// that silently forgets work it already did.

// logBuilder writes a session log the way the loop does, so the fixtures below
// exercise the real entry shapes rather than a convenient approximation.
type logBuilder struct {
	entries []SessionEntry
	seq     int
	state   CheckpointState
}

func (b *logBuilder) add(e SessionEntry) {
	b.seq++
	e.Seq = b.seq
	b.entries = append(b.entries, e)
}

func (b *logBuilder) message(turn int, role Role, content string) {
	b.add(SessionEntry{Kind: EntryMessage, Turn: turn, Message: &Message{Role: role, Content: content}})
}

func (b *logBuilder) goal(g string) {
	b.add(SessionEntry{Kind: EntryGoal, Goal: g})
	b.state.Goal = g
}

func (b *logBuilder) modelChange(turn int, model string) {
	b.add(SessionEntry{Kind: EntryModelChange, Turn: turn, Model: model})
	b.state.Model = model
}

func (b *logBuilder) disable(turn int, tool string) {
	b.add(SessionEntry{Kind: EntryToolDisabled, Turn: turn, Tool: tool})
	b.state.DisabledTools = append(b.state.DisabledTools, tool)
}

// compact writes the durable bracket the loop writes: a start entry, then a
// completion carrying the retained transcript and the mirrored run state.
func (b *logBuilder) compact(turn int, retained []Message) {
	b.add(SessionEntry{Kind: EntryCompaction, Turn: turn})
	b.add(SessionEntry{
		Kind: EntryCompaction, Turn: turn, Final: true,
		Retained: retained,
		State:    b.state.clone(),
	})
}

// checkpointSeq is the seq the store would hand LoadResumeLog.
func (b *logBuilder) checkpointSeq() int {
	seq := 0
	for _, e := range b.entries {
		if e.Kind == EntryCompaction && e.Final && e.Retained != nil && e.State != nil {
			seq = e.Seq
		}
	}
	return seq
}

// window is the suffix beginning at the newest checkpoint.
func (b *logBuilder) window() []SessionEntry {
	seq := b.checkpointSeq()
	var out []SessionEntry
	for _, e := range b.entries {
		if e.Seq >= seq {
			out = append(out, e)
		}
	}
	return out
}

// longRun builds a log shaped like a run that compacted several times: a goal,
// escalation, a disabled tool, and message traffic between checkpoints.
func longRun() *logBuilder {
	b := &logBuilder{}
	b.message(1, RoleUser, "audit every shard")
	b.goal("all shards reconciled")
	for turn := 1; turn <= 30; turn++ {
		b.message(turn, RoleAssistant, fmt.Sprintf("working on shard %d", turn))
		b.message(turn, RoleTool, fmt.Sprintf("shard %d ok", turn))
		switch turn {
		case 8:
			b.modelChange(turn, "claude-opus-5")
		case 12:
			b.disable(turn, "flaky_probe")
		case 21:
			b.disable(turn, "slow_probe")
		}
		if turn%10 == 0 {
			b.compact(turn, []Message{
				{Role: RoleSystem, Content: fmt.Sprintf("[checkpoint] shards 1-%d done", turn)},
				{Role: RoleAssistant, Content: fmt.Sprintf("continuing from shard %d", turn)},
			})
		}
	}
	b.message(31, RoleAssistant, "final stretch")
	return b
}

// TestWindowReducesLikeTheWholeLog is the property the whole feature rests on.
func TestWindowReducesLikeTheWholeLog(t *testing.T) {
	b := longRun()
	full := ReduceSession(b.entries)
	windowed := ReduceSession(b.window())

	if !reflect.DeepEqual(full, windowed) {
		t.Fatalf("window folded to a different state\n full: %+v\n window: %+v", full, windowed)
	}
	// A guard on the fixture itself: if the window were the whole log the
	// equality above would be trivially true and would prove nothing.
	if len(b.window()) >= len(b.entries) {
		t.Fatalf("fixture does not window: %d of %d entries", len(b.window()), len(b.entries))
	}
}

// TestWindowRecoversLikeTheWholeLog carries the property up one level, to the
// resume plan the loop actually acts on.
func TestWindowRecoversLikeTheWholeLog(t *testing.T) {
	b := longRun()
	// Leave a dangling tool call in the tail so recovery has real work to do —
	// a window that agrees only on empty plans agrees about nothing.
	b.add(SessionEntry{Kind: EntryMessage, Turn: 32, Message: &Message{
		Role:      RoleAssistant,
		ToolCalls: []ToolCall{{ID: "call-1", Name: "read_ledger", Arguments: "{}"}},
	}})

	full := RecoverSession(b.entries, nil, RecoveryMarkInterrupted)
	windowed := RecoverSession(b.window(), nil, RecoveryMarkInterrupted)
	if !reflect.DeepEqual(full, windowed) {
		t.Fatalf("window recovered a different plan\n full: %+v\n window: %+v", full, windowed)
	}
	if !full.Interrupted || len(full.DroppedCalls) != 1 {
		t.Fatalf("fixture lost its dangling call: %+v", full)
	}
}

// TestWindowKeepsStateWrittenBeforeTheCheckpoint is the specific way a naive
// suffix read goes wrong: the transcript restarts at the checkpoint, but the
// goal, the escalated model, and the disabled tools were all recorded BEFORE it
// and would simply vanish.
func TestWindowKeepsStateWrittenBeforeTheCheckpoint(t *testing.T) {
	b := longRun()
	rs := ReduceSession(b.window())

	if rs.Goal != "all shards reconciled" {
		t.Fatalf("window lost the run's goal: %q", rs.Goal)
	}
	if rs.Model != "claude-opus-5" {
		t.Fatalf("window lost the escalated model: %q", rs.Model)
	}
	if !reflect.DeepEqual(rs.DisabledTools, []string{"flaky_probe", "slow_probe"}) {
		t.Fatalf("window lost disabled tools: %v", rs.DisabledTools)
	}
	// Every one of those was written before the newest checkpoint — that is what
	// makes this test about the checkpoint's state and not about the tail.
	seq := b.checkpointSeq()
	for _, e := range b.entries {
		switch e.Kind {
		case EntryGoal, EntryModelChange, EntryToolDisabled:
			if e.Seq >= seq {
				t.Fatalf("fixture writes %s at seq %d, at or after the checkpoint at %d", e.Kind, e.Seq, seq)
			}
		}
	}
}

// TestCheckpointStateIsOwnedByItsEntry pins the clone.
//
// The loop keeps ONE running mirror of the state and stamps it on every
// checkpoint. A written entry is immutable — that is the log's whole premise —
// so it must own its copy, and the mirror must be free to keep changing without
// rewriting history that has already been recorded.
//
// The mutation is in-place on purpose. Appending to a shared slice usually
// reallocates and hides the aliasing, which is exactly what makes it a bug that
// surfaces later and at random rather than one a growing fixture would catch.
func TestCheckpointStateIsOwnedByItsEntry(t *testing.T) {
	mirror := CheckpointState{
		Model:         "claude-opus-5",
		ActiveTools:   []string{"run_sql", "create_chart"},
		DisabledTools: []string{"flaky_probe"},
		Goal:          "all shards reconciled",
	}
	stamped := mirror.clone()

	mirror.DisabledTools[0] = "something_else"
	mirror.ActiveTools[0] = "something_else"
	mirror.Model = "claude-haiku-4-5"

	if got := stamped.DisabledTools[0]; got != "flaky_probe" {
		t.Fatalf("a written checkpoint changed when the mirror did: disabled = %q", got)
	}
	if got := stamped.ActiveTools[0]; got != "run_sql" {
		t.Fatalf("a written checkpoint changed when the mirror did: active = %q", got)
	}
	if stamped.Model != "claude-opus-5" {
		t.Fatalf("a written checkpoint changed when the mirror did: model = %q", stamped.Model)
	}
}

// TestLoadResumeLogFallsBack covers every way a window can be unavailable or
// unsafe. Each case must end in the FULL log — the failure mode of reading too
// much is slowness, and of reading too little is lost work.
func TestLoadResumeLogFallsBack(t *testing.T) {
	base := longRun()

	cases := []struct {
		name  string
		store SessionStore
		want  int // entries expected back
	}{
		{
			name:  "a store without the capability",
			store: &plainStore{entries: base.entries},
			want:  len(base.entries),
		},
		{
			name:  "a store whose checkpoint lookup fails",
			store: &windowStore{entries: base.entries, seq: base.checkpointSeq(), checkErr: errors.New("db down")},
			want:  len(base.entries),
		},
		{
			name:  "a log with no self-contained checkpoint",
			store: &windowStore{entries: base.entries, seq: 0},
			want:  len(base.entries),
		},
		{
			name:  "a log that branched",
			store: &windowStore{entries: base.entries, seq: base.checkpointSeq(), branched: true},
			want:  len(base.entries),
		},
		{
			name:  "a window read that fails",
			store: &windowStore{entries: base.entries, seq: base.checkpointSeq(), fromErr: errors.New("db down")},
			want:  len(base.entries),
		},
		{
			name:  "a window that does not start at the checkpoint",
			store: &windowStore{entries: base.entries, seq: base.checkpointSeq(), skew: 1},
			want:  len(base.entries),
		},
		{
			name:  "a healthy window",
			store: &windowStore{entries: base.entries, seq: base.checkpointSeq()},
			want:  len(base.window()),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := LoadResumeLog(context.Background(), tc.store, "s1")
			if err != nil {
				t.Fatalf("LoadResumeLog: %v", err)
			}
			if len(got) != tc.want {
				t.Fatalf("read %d entries, want %d", len(got), tc.want)
			}
			// Whatever was read must fold to the same state. That is the only
			// thing a fallback is for.
			if !reflect.DeepEqual(ReduceSession(got), ReduceSession(base.entries)) {
				t.Fatal("the log that was read folds to a different state than the whole log")
			}
		})
	}
}

// TestMemorySessionStoreServesAWindow checks the in-process store against the
// same contract, since it is what every test and local run resumes through.
func TestMemorySessionStoreServesAWindow(t *testing.T) {
	ctx := context.Background()
	m := NewMemorySessionStore()
	b := longRun()
	for _, e := range b.entries {
		if err := m.Append(ctx, "s1", e); err != nil {
			t.Fatalf("append: %v", err)
		}
	}

	seq, branched, err := m.CheckpointSeq(ctx, "s1")
	if err != nil || branched {
		t.Fatalf("CheckpointSeq = %d, %v, %v", seq, branched, err)
	}
	got, err := LoadResumeLog(ctx, m, "s1")
	if err != nil {
		t.Fatalf("LoadResumeLog: %v", err)
	}
	full, _ := m.Log(ctx, "s1")
	if len(got) >= len(full) {
		t.Fatalf("read %d of %d entries — no window was served", len(got), len(full))
	}
	if !reflect.DeepEqual(ReduceSession(got), ReduceSession(full)) {
		t.Fatal("the window folds to a different state than the whole log")
	}

	// A leaf move means the newest checkpoint may sit on an abandoned branch, so
	// the store must stop offering a window at all.
	if err := m.Append(ctx, "s1", SessionEntry{Kind: EntryLeafMove, Target: "#3"}); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, branched, _ := m.CheckpointSeq(ctx, "s1"); !branched {
		t.Fatal("a log with a leaf move must report itself branched")
	}
	got, _ = LoadResumeLog(ctx, m, "s1")
	if len(got) != len(full)+1 {
		t.Fatalf("a branched log must be read whole, got %d of %d entries", len(got), len(full)+1)
	}
}

// TestUnsettledInboxDefeatsTheWindow covers the third window-safety rule: an
// EntryInbox queued before the newest checkpoint and never settled is invisible
// to a suffix read — the fold scans the raw log for pending inbox items — so
// the store must report no checkpoint and force the full read. Settling it
// restores the window.
func TestUnsettledInboxDefeatsTheWindow(t *testing.T) {
	ctx := context.Background()
	m := NewMemorySessionStore()
	b := longRun()
	for _, e := range b.entries {
		if err := m.Append(ctx, "s1", e); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	full, _ := m.Log(ctx, "s1")

	// A steer queued mid-run and never delivered sits before every checkpoint.
	msg := Message{Role: RoleUser, Content: "also check shard 7"}
	if err := m.Append(ctx, "s1", SessionEntry{Kind: EntryInbox, ID: "inbox-1", Lane: InboxSteer, Message: &msg}); err != nil {
		t.Fatalf("append inbox: %v", err)
	}
	if seq, _, _ := m.CheckpointSeq(ctx, "s1"); seq != 0 {
		t.Fatalf("an unsettled inbox must report no checkpoint, got seq %d", seq)
	}
	got, err := LoadResumeLog(ctx, m, "s1")
	if err != nil {
		t.Fatalf("LoadResumeLog: %v", err)
	}
	if len(got) != len(full)+1 {
		t.Fatalf("an unsettled inbox must force the full read, got %d of %d entries", len(got), len(full)+1)
	}
	if rs := ReduceSession(got); len(rs.Inbox) != 1 || rs.Inbox[0].ID != "inbox-1" {
		t.Fatalf("the pending steer must survive the fold, got %+v", rs.Inbox)
	}

	// Settling it makes the window safe again.
	if err := m.Append(ctx, "s1", SessionEntry{Kind: EntryInboxDone, Target: "inbox-1"}); err != nil {
		t.Fatalf("append done: %v", err)
	}
	got, err = LoadResumeLog(ctx, m, "s1")
	if err != nil {
		t.Fatalf("LoadResumeLog: %v", err)
	}
	if len(got) >= len(full)+2 {
		t.Fatalf("a settled inbox must not defeat the window, got %d of %d entries", len(got), len(full)+2)
	}
}

// --- stores ---

// plainStore implements only SessionStore, standing in for a backend that has
// not adopted the windowing capability.
type plainStore struct{ entries []SessionEntry }

func (p *plainStore) Append(context.Context, string, SessionEntry) error { return nil }
func (p *plainStore) Log(context.Context, string) ([]SessionEntry, error) {
	return p.entries, nil
}

// windowStore implements the capability and can fail or misbehave on demand.
type windowStore struct {
	entries  []SessionEntry
	seq      int
	branched bool
	checkErr error
	fromErr  error
	skew     int // start the window this many entries late
}

func (w *windowStore) Append(context.Context, string, SessionEntry) error { return nil }
func (w *windowStore) Log(context.Context, string) ([]SessionEntry, error) {
	return w.entries, nil
}
func (w *windowStore) CheckpointSeq(context.Context, string) (int, bool, error) {
	return w.seq, w.branched, w.checkErr
}
func (w *windowStore) LogFrom(_ context.Context, _ string, sinceSeq int) ([]SessionEntry, error) {
	if w.fromErr != nil {
		return nil, w.fromErr
	}
	var out []SessionEntry
	for _, e := range w.entries {
		if e.Seq >= sinceSeq+w.skew {
			out = append(out, e)
		}
	}
	return out, nil
}
