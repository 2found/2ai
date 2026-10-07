package agentcore

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// msgEntry builds an EntryMessage node with an explicit id/parent.
func msgEntry(id, parent string, role Role, content string) SessionEntry {
	return SessionEntry{Kind: EntryMessage, ID: id, ParentID: parent, Message: &Message{Role: role, Content: content}}
}

// TestFlatLogActivePathIsWholeLog pins backward compatibility: a legacy id-less
// log is a single-branch tree, so ActivePath and ReduceSession see every entry
// in order — exactly the pre-tree behavior.
func TestFlatLogActivePathIsWholeLog(t *testing.T) {
	log := []SessionEntry{
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "hi"}},
		{Kind: EntryMessage, Message: &Message{Role: RoleAssistant, Content: "hello"}},
		{Kind: EntryLeaf},
	}
	path := ActivePath(log)
	if len(path) != 3 {
		t.Fatalf("flat log path length = %d, want 3", len(path))
	}
	rs := ReduceSession(log)
	if len(rs.Messages) != 2 || rs.Messages[0].Content != "hi" || rs.Messages[1].Content != "hello" {
		t.Fatalf("flat log reduce changed: %+v", rs.Messages)
	}
	if !rs.Completed {
		t.Fatal("flat log with leaf must reduce Completed")
	}
}

func TestLinearActivePathRecognizesOnlyAnUnbranchedLog(t *testing.T) {
	linear := []SessionEntry{
		msgEntry("one", "", RoleUser, "one"),
		{Kind: EntryToolProgress, CallID: "call", Content: "half"},
		msgEntry("two", "one", RoleAssistant, "two"),
		{Kind: EntryLeafMove}, // an empty legacy control record changes nothing
		msgEntry("three", "two", RoleUser, "three"),
	}
	path, ok := linearActivePath(linear)
	if !ok {
		t.Fatal("linear production-shaped log was sent through tree fallback")
	}
	if len(path) != 3 || path[0].ID != "one" || path[1].ID != "two" || path[2].ID != "three" {
		t.Fatalf("linear path = %+v", path)
	}

	for name, log := range map[string][]SessionEntry{
		"explicit branch": {
			msgEntry("one", "", RoleUser, "one"),
			msgEntry("two", "one", RoleAssistant, "two"),
			msgEntry("fork", "one", RoleAssistant, "fork"),
		},
		"leaf move": {
			msgEntry("one", "", RoleUser, "one"),
			{Kind: EntryLeafMove, Target: "one"},
		},
		"duplicate id": {
			msgEntry("same", "", RoleUser, "one"),
			msgEntry("same", "same", RoleAssistant, "two"),
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, ok := linearActivePath(log); ok {
				t.Fatal("non-linear log bypassed the tree algorithm")
			}
		})
	}
}

// TestBranchByParentIDForksThePath verifies pi's append-is-branch model: an
// entry whose ParentID names an earlier entry forks the tree, and the active
// path follows the new branch while the abandoned one stays in the log.
func TestBranchByParentIDForksThePath(t *testing.T) {
	log := []SessionEntry{
		msgEntry("a", "", RoleUser, "task"),
		msgEntry("b", "", RoleAssistant, "first try"),
		msgEntry("c", "", RoleAssistant, "dead end"),
		// Fork: continue from "a" instead of "c".
		msgEntry("d", "a", RoleAssistant, "second try"),
	}
	if leaf := ActiveLeaf(log); leaf != "d" {
		t.Fatalf("leaf = %q, want d", leaf)
	}
	rs := ReduceSession(log)
	want := []string{"task", "second try"}
	if len(rs.Messages) != len(want) {
		t.Fatalf("reduced %d messages, want %d: %+v", len(rs.Messages), len(want), rs.Messages)
	}
	for i, w := range want {
		if rs.Messages[i].Content != w {
			t.Fatalf("message[%d] = %q, want %q", i, rs.Messages[i].Content, w)
		}
	}
}

// TestLeafMoveRewindsImplicitChain verifies an EntryLeafMove rewinds the leaf so
// the next id-less append chains from the move target, not the previous tail.
func TestLeafMoveRewindsImplicitChain(t *testing.T) {
	log := []SessionEntry{
		msgEntry("a", "", RoleUser, "task"),
		msgEntry("b", "", RoleAssistant, "wrong direction"),
		{Kind: EntryLeafMove, Target: "a"},
		// Legacy-style writer appends without ids after the rewind.
		{Kind: EntryMessage, Message: &Message{Role: RoleAssistant, Content: "fresh start"}},
	}
	rs := ReduceSession(log)
	want := []string{"task", "fresh start"}
	if len(rs.Messages) != len(want) {
		t.Fatalf("reduced %d messages, want %d: %+v", len(rs.Messages), len(want), rs.Messages)
	}
	for i, w := range want {
		if rs.Messages[i].Content != w {
			t.Fatalf("message[%d] = %q, want %q", i, rs.Messages[i].Content, w)
		}
	}
}

// TestSessionTreeAddressesLegacyEntries verifies id-less entries get stable
// synthetic ids so they remain valid Rewind targets.
func TestSessionTreeAddressesLegacyEntries(t *testing.T) {
	log := []SessionEntry{
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "hi"}},
		{Kind: EntryMessage, Message: &Message{Role: RoleAssistant, Content: "hello"}},
	}
	nodes := SessionTree(log)
	if len(nodes) != 2 {
		t.Fatalf("tree has %d nodes, want 2", len(nodes))
	}
	if nodes[0].ID != "#0" || nodes[1].ID != "#1" || nodes[1].ParentID != "#0" {
		t.Fatalf("synthetic ids wrong: %+v", nodes)
	}
}

// TestRewindWritesBranchSummaryAndMovesLeaf drives the full rewind flow against
// a store: the abandoned span (and only it) reaches the summarizer, the summary
// node chains from the target, the leaf moves, and the reduced history is the
// new branch with the summary folded in as marked system context.
func TestRewindWritesBranchSummaryAndMovesLeaf(t *testing.T) {
	ctx := context.Background()
	store := newMemSessionStore()
	const sid = "s1"
	for _, e := range []SessionEntry{
		msgEntry("a", "", RoleUser, "analyze signups"),
		msgEntry("b", "", RoleAssistant, "querying the wrong table"),
		msgEntry("c", "", RoleAssistant, "still wrong"),
	} {
		if err := store.Append(ctx, sid, e); err != nil {
			t.Fatal(err)
		}
	}

	var summarized []Message
	newLeaf, err := Rewind(ctx, store, sid, "a", BranchOptions{
		Summarize: func(_ context.Context, abandoned []Message) (string, Usage, error) {
			summarized = abandoned
			return "tried events_raw; it lacks signup rows", Usage{InputTokens: 40, OutputTokens: 10, CostUSD: 0.002}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if newLeaf == "" || newLeaf == "a" {
		t.Fatalf("new leaf should be the summary node, got %q", newLeaf)
	}
	// Only the abandoned span (b, c) is summarized — the shared prefix (a) is not.
	if len(summarized) != 2 || summarized[0].Content != "querying the wrong table" || summarized[1].Content != "still wrong" {
		t.Fatalf("summarizer saw wrong span: %+v", summarized)
	}

	// Continue down the new branch and reduce.
	if err := store.Append(ctx, sid, msgEntry("d", "", RoleAssistant, "using signup_events now")); err != nil {
		t.Fatal(err)
	}
	log, err := store.Log(ctx, sid)
	if err != nil {
		t.Fatal(err)
	}
	rs := ReduceSession(log)
	if len(rs.Messages) != 3 {
		t.Fatalf("reduced %d messages, want 3 (task, branch summary, new work): %+v", len(rs.Messages), rs.Messages)
	}
	if rs.Messages[0].Content != "analyze signups" {
		t.Fatalf("message[0] = %q", rs.Messages[0].Content)
	}
	if rs.Messages[1].Role != RoleSystem || !strings.HasPrefix(rs.Messages[1].Content, branchSummaryMarker) || !strings.Contains(rs.Messages[1].Content, "events_raw") {
		t.Fatalf("branch summary not folded as marked system context: %+v", rs.Messages[1])
	}
	if rs.Messages[2].Content != "using signup_events now" {
		t.Fatalf("message[2] = %q", rs.Messages[2].Content)
	}
	// The abandoned branch is still in the log for inspection.
	if len(log) != 6 { // a, b, c, branch_summary, leaf_move, d
		t.Fatalf("log length = %d, want 6", len(log))
	}
	// The summarization call's spend is stamped on the branch-summary entry so
	// tree navigation is never invisible spend.
	var su *Usage
	for _, e := range log {
		if e.Kind == EntryBranchSummary {
			su = e.Usage
		}
	}
	if su == nil || su.InputTokens != 40 || su.OutputTokens != 10 {
		t.Fatalf("branch summary entry missing usage stamp: %+v", su)
	}
}

// TestRewindDegradesToBareLeafMove verifies a failing summarizer never fails
// the rewind: the leaf still moves, with no summary node.
func TestRewindDegradesToBareLeafMove(t *testing.T) {
	ctx := context.Background()
	store := newMemSessionStore()
	const sid = "s1"
	for _, e := range []SessionEntry{
		msgEntry("a", "", RoleUser, "task"),
		msgEntry("b", "", RoleAssistant, "abandoned"),
	} {
		if err := store.Append(ctx, sid, e); err != nil {
			t.Fatal(err)
		}
	}
	newLeaf, err := Rewind(ctx, store, sid, "a", BranchOptions{
		Summarize: func(context.Context, []Message) (string, Usage, error) {
			return "", Usage{}, errors.New("summarizer down")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if newLeaf != "a" {
		t.Fatalf("new leaf = %q, want a", newLeaf)
	}
	log, _ := store.Log(ctx, sid)
	rs := ReduceSession(log)
	if len(rs.Messages) != 1 || rs.Messages[0].Content != "task" {
		t.Fatalf("reduce after degraded rewind: %+v", rs.Messages)
	}
}

// TestRewindRejectsUnknownTarget pins the error path.
func TestRewindRejectsUnknownTarget(t *testing.T) {
	ctx := context.Background()
	store := newMemSessionStore()
	if err := store.Append(ctx, "s1", msgEntry("a", "", RoleUser, "task")); err != nil {
		t.Fatal(err)
	}
	if _, err := Rewind(ctx, store, "s1", "nope", BranchOptions{}); err == nil {
		t.Fatal("rewind to an unknown target must error")
	}
}

// TestRecoverSessionFollowsActiveBranch verifies durable recovery resumes down
// the rewound branch, not the abandoned one.
func TestRecoverSessionFollowsActiveBranch(t *testing.T) {
	log := []SessionEntry{
		msgEntry("a", "", RoleUser, "task"),
		msgEntry("b", "", RoleAssistant, "abandoned work"),
		{Kind: EntryLeafMove, Target: "a"},
		msgEntry("c", "", RoleAssistant, "new branch work"),
		// no leaf: the run crashed on the new branch
	}
	plan := RecoverSession(log, NewToolSet(), RecoveryMarkInterrupted)
	if plan.Completed {
		t.Fatal("crashed branch must not be Completed")
	}
	if !plan.Interrupted {
		t.Fatal("crashed branch must be Interrupted")
	}
	for _, m := range plan.Messages {
		if m.Content == "abandoned work" {
			t.Fatal("recovery leaked the abandoned branch into the resume history")
		}
	}
}

