package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"
)

type countingLeaseStore struct {
	*MemorySessionStore
	acquires int
}

type routedAnswerStore struct {
	*MemorySessionStore
	failParent bool
}

func (s *routedAnswerStore) Append(ctx context.Context, id string, entry SessionEntry) error {
	if s.failParent && id == "parent" && entry.Kind == EntryPiAnswer {
		return errors.New("parent answer write failed")
	}
	return s.MemorySessionStore.Append(ctx, id, entry)
}

func appendRoutedQuestion(t *testing.T, store SessionStore, session, effect string, route *ChildQuestionError) {
	t.Helper()
	audit := PiToolOutcome{Parked: true, Executed: true, QuestionID: effect, ChildQuestion: route,
		Trace: ToolTrace{CallID: "provider-call", Tool: "ask", Allowed: true, Args: `{"question":"Which?","options":["one","two"]}`}}
	if route != nil {
		audit.Trace.Tool, audit.Trace.Args = "delegate", `{"task":"work"}`
	}
	raw, err := json.Marshal(map[string]any{"effectId": effect, "result": map[string]any{"details": audit}})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Append(context.Background(), session, SessionEntry{Kind: EntryPiEffectDone, CallID: "provider-call", Content: string(raw)}); err != nil {
		t.Fatal(err)
	}
}

func TestRecordSessionAnswerRoutesChildAcrossParentWriteFailure(t *testing.T) {
	store := &routedAnswerStore{MemorySessionStore: NewMemorySessionStore(), failParent: true}
	route := &ChildQuestionError{SessionID: "parent/child", QuestionID: "child-question", Question: json.RawMessage(`{"options":["one","two"],"question":"Which?"}`)}
	appendRoutedQuestion(t, store, route.SessionID, route.QuestionID, nil)
	appendRoutedQuestion(t, store, "parent", "parent-question", route)
	ctx, release, err := AcquireSessionLease(context.Background(), store, "parent")
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if _, err := RecordSessionAnswer(ctx, store, "parent", "parent-question", "one"); err == nil {
		t.Fatal("parent write failure was lost")
	}
	parent, _ := store.Log(ctx, "parent")
	child, _ := store.Log(ctx, route.SessionID)
	if _, _, pending := PendingQuestion(parent); !pending {
		t.Fatal("failed parent append closed its workflow")
	}
	if _, _, pending := PendingQuestion(child); pending {
		t.Fatal("child answer was not committed first")
	}
	store.failParent = false
	if _, err := RecordSessionAnswer(ctx, store, "parent", "parent-question", "two"); !errors.Is(err, ErrAnswerConflict) {
		t.Fatalf("partial commit allowed a different answer: %v", err)
	}
	appended, err := RecordSessionAnswer(ctx, store, "parent", "parent-question", "one")
	if err != nil || !appended {
		t.Fatalf("exact retry did not finish parent append: %v %v", appended, err)
	}
	appended, err = RecordSessionAnswer(ctx, store, "parent", "parent-question", "one")
	if err != nil || appended {
		t.Fatalf("settled retry was not idempotent: %v %v", appended, err)
	}
	for id, questionID := range map[string]string{"parent": "parent-question", route.SessionID: route.QuestionID} {
		entries, err := store.Log(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		answers := 0
		for _, entry := range entries {
			if entry.Kind == EntryPiAnswer {
				answers++
				if entry.CallID != questionID || entry.Answer != "one" {
					t.Fatalf("answer crossed workflow identities: %+v", entry)
				}
			}
		}
		if answers != 1 {
			t.Fatalf("%s received %d answers", id, answers)
		}
	}
}

func TestRecordSessionAnswerRejectsInvalidChildRoutes(t *testing.T) {
	for _, tc := range []struct{ name, session, questionID, question string }{
		{"outside parent", "other/child", "child-question", `{"question":"Which?","options":["one","two"]}`},
		{"self", "parent", "child-question", `{"question":"Which?","options":["one","two"]}`},
		{"missing receipt", "parent/child", "missing", `{"question":"Which?","options":["one","two"]}`},
		{"changed question", "parent/child", "child-question", `{"question":"Another?"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := NewMemorySessionStore()
			appendRoutedQuestion(t, store, tc.session, "child-question", nil)
			appendRoutedQuestion(t, store, "parent", "parent-question", &ChildQuestionError{SessionID: tc.session, QuestionID: tc.questionID, Question: json.RawMessage(tc.question)})
			ctx, release, err := AcquireSessionLease(context.Background(), store, "parent")
			if err != nil {
				t.Fatal(err)
			}
			defer release()
			if _, err := RecordSessionAnswer(ctx, store, "parent", "parent-question", "one"); err == nil {
				t.Fatal("invalid child route accepted")
			}
			for _, session := range []string{"parent", tc.session} {
				entries, err := store.Log(ctx, session)
				if err != nil {
					t.Fatal(err)
				}
				for _, entry := range entries {
					if entry.Kind == EntryPiAnswer {
						t.Fatal("invalid route committed an answer")
					}
				}
			}
		})
	}
}

func TestRecordSessionAnswerRoutesNestedChildren(t *testing.T) {
	store := NewMemorySessionStore()
	question := json.RawMessage(`{"question":"Which?","options":["one","two"]}`)
	appendRoutedQuestion(t, store, "parent/child/grandchild", "leaf-question", nil)
	appendRoutedQuestion(t, store, "parent/child", "child-question", &ChildQuestionError{SessionID: "parent/child/grandchild", QuestionID: "leaf-question", Question: question})
	appendRoutedQuestion(t, store, "parent", "parent-question", &ChildQuestionError{SessionID: "parent/child", QuestionID: "child-question", Question: question})
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	ctx, release, err := AcquireSessionLease(ctx, store, "parent")
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if _, err := RecordSessionAnswer(ctx, store, "parent", "parent-question", "one"); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"parent", "parent/child", "parent/child/grandchild"} {
		entries, err := store.Log(ctx, id)
		if err != nil || len(entries) != 2 || entries[1].Kind != EntryPiAnswer || entries[1].Answer != "one" {
			t.Fatalf("nested answer lost or repeated in %s: %+v %v", id, entries, err)
		}
	}
}

type leaseLossStore struct {
	*MemorySessionStore
	cancel context.CancelCauseFunc
}

func (s *leaseLossStore) AcquireSessionLease(ctx context.Context, _ string) (context.Context, func() error, error) {
	leaseCtx, cancel := context.WithCancelCause(ctx)
	s.cancel = cancel
	return leaseCtx, func() error { return nil }, nil
}

func (s *leaseLossStore) AppendBatch(ctx context.Context, id string, entries []SessionEntry) error {
	for _, entry := range entries {
		if entry.Kind == EntryLeaf {
			s.cancel(ErrSessionLeaseLost)
			return ErrSessionLeaseLost
		}
	}
	return s.MemorySessionStore.AppendBatch(ctx, id, entries)
}

func (s *countingLeaseStore) AcquireSessionLease(ctx context.Context, id string) (context.Context, func() error, error) {
	s.acquires++
	return s.MemorySessionStore.AcquireSessionLease(ctx, id)
}

func TestMemorySessionLeaseSerializesOneSession(t *testing.T) {
	store := NewMemorySessionStore()
	_, releaseFirst, err := AcquireSessionLease(context.Background(), store, "shared")
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}

	acquired := make(chan func() error, 1)
	go func() {
		_, release, err := AcquireSessionLease(context.Background(), store, "shared")
		if err == nil {
			acquired <- release
		}
	}()
	select {
	case <-acquired:
		t.Fatal("second owner acquired the same live session")
	case <-time.After(25 * time.Millisecond):
	}

	// A different session is independent and must not queue behind shared.
	_, releaseOther, err := AcquireSessionLease(context.Background(), store, "other")
	if err != nil {
		t.Fatalf("independent acquire: %v", err)
	}
	if err := releaseOther(); err != nil {
		t.Fatalf("release independent: %v", err)
	}

	if err := releaseFirst(); err != nil {
		t.Fatalf("release first: %v", err)
	}
	var releaseSecond func() error
	select {
	case releaseSecond = <-acquired:
	case <-time.After(time.Second):
		t.Fatal("second owner did not acquire after release")
	}
	if err := releaseSecond(); err != nil {
		t.Fatalf("release second: %v", err)
	}
	if err := releaseSecond(); err != nil { // idempotent
		t.Fatalf("second release: %v", err)
	}
}

func TestMemorySessionLeaseCanceledWaiter(t *testing.T) {
	store := NewMemorySessionStore()
	_, release, err := AcquireSessionLease(context.Background(), store, "shared")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = release() }()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, _, err = AcquireSessionLease(ctx, store, "shared")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("waiter error = %v, want deadline exceeded", err)
	}
}

func TestAcquireSessionLeaseNestedContextDoesNotDeadlock(t *testing.T) {
	store := NewMemorySessionStore()
	ctx, release, err := AcquireSessionLease(context.Background(), store, "shared")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = release() }()

	_, nestedRelease, err := AcquireSessionLease(ctx, store, "shared")
	if err != nil {
		t.Fatalf("nested acquire: %v", err)
	}
	if err := nestedRelease(); err != nil {
		t.Fatalf("nested release: %v", err)
	}
}

func TestRecordSessionAnswerIsIdempotentAndRejectsConflict(t *testing.T) {
	ctx := context.Background()
	store := NewMemorySessionStore()
	if err := store.Append(ctx, "s", SessionEntry{Kind: EntryQuestion, CallID: "call-1"}); err != nil {
		t.Fatal(err)
	}
	appended, err := RecordSessionAnswer(ctx, store, "s", "call-1", "yes")
	if err != nil || !appended {
		t.Fatalf("first answer: appended=%v err=%v", appended, err)
	}
	appended, err = RecordSessionAnswer(ctx, store, "s", "call-1", "yes")
	if err != nil || appended {
		t.Fatalf("exact retry: appended=%v err=%v", appended, err)
	}
	if _, err := RecordSessionAnswer(ctx, store, "s", "call-1", "no"); !errors.Is(err, ErrAnswerConflict) {
		t.Fatalf("conflicting retry error = %v, want ErrAnswerConflict", err)
	}
	log, err := store.Log(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	answers := 0
	for _, entry := range log {
		if entry.Kind == EntryAnswer {
			answers++
		}
	}
	if answers != 1 {
		t.Fatalf("answer entries = %d, want 1", answers)
	}
}

