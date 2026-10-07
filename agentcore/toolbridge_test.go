package agentcore

import (
	"context"
	"testing"
)

func TestMemorySessionOwnsNestedInvocationSlice(t *testing.T) {
	store := NewMemorySessionStore()
	entry := SessionEntry{
		Kind: EntryToolOutcome,
		Outcome: &ToolOutcomeRecord{Invocations: []ToolInvocation{{
			Trace: ToolTrace{Tool: "nested", Allowed: true}, Executed: true,
		}}},
	}
	if err := store.Append(context.Background(), "session", entry); err != nil {
		t.Fatalf("Append: %v", err)
	}
	entry.Outcome.Invocations[0].Trace.Tool = "mutated-writer"

	first, err := store.Log(context.Background(), "session")
	if err != nil {
		t.Fatalf("Log: %v", err)
	}
	if got := first[0].Outcome.Invocations[0].Trace.Tool; got != "nested" {
		t.Fatalf("stored invocation changed through writer alias: %q", got)
	}
	first[0].Outcome.Invocations[0].Trace.Tool = "mutated-reader"

	second, err := store.Log(context.Background(), "session")
	if err != nil {
		t.Fatalf("Log again: %v", err)
	}
	if got := second[0].Outcome.Invocations[0].Trace.Tool; got != "nested" {
		t.Fatalf("stored invocation changed through reader alias: %q", got)
	}
}

func TestNestedToolBudgetReservationIsAtomic(t *testing.T) {
	budget := newToolExecutionBudget(1)
	if !budget.reserve() {
		t.Fatal("first reservation failed")
	}
	if budget.reserve() {
		t.Fatal("second reservation exceeded the hard cap")
	}
	budget.release()
	if !budget.reserve() {
		t.Fatal("released reservation was not reusable")
	}
}
