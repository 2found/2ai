package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/2found/2ai/agentcore"
)

// fakeStore is a MemoryStore that records Remember calls and answers Recall
// with whatever it was given.
type fakeStore struct {
	agentcore.MemoryStore // embedded nil: only the methods under test are real
	remembered            []agentcore.MemoryEntry
	recall                []agentcore.MemoryEntry
}

func (f *fakeStore) Remember(_ context.Context, e agentcore.MemoryEntry) error {
	f.remembered = append(f.remembered, e)
	return nil
}

func (f *fakeStore) Recall(_ context.Context, scopeID, _ string, _ int) ([]agentcore.MemoryEntry, error) {
	var out []agentcore.MemoryEntry
	for _, e := range f.recall {
		if e.ScopeID == scopeID {
			out = append(out, e)
		}
	}
	return out, nil
}

// fakeCurator adds MemoryCurator to fakeStore: Supersede/Update refuse an id
// outside the pinned scope, the way the real store's scope predicate does.
type fakeCurator struct {
	*fakeStore
	superseded []string
	updates    []agentcore.MemoryEntry
}

var errWrongScope = errors.New("agent memory not found in scope")

func (f *fakeCurator) Supersede(_ context.Context, scopeID, id, _ string) error {
	for _, e := range f.recall {
		if e.ID == id && e.ScopeID == scopeID {
			f.superseded = append(f.superseded, id)
			return nil
		}
	}
	return errWrongScope
}

func (f *fakeCurator) Update(_ context.Context, scopeID, id string, e agentcore.MemoryEntry) error {
	for _, m := range f.recall {
		if m.ID == id && m.ScopeID == scopeID {
			f.updates = append(f.updates, e)
			return nil
		}
	}
	return errWrongScope
}

func runInfo() agentcore.RunInfo { return agentcore.RunInfo{ScopeID: "agent-1"} }

// A run with no store gets no curation tools at all — the plugin declines
// rather than advertising tools that can only fail.
func TestNoStoreNoTools(t *testing.T) {
	ext, err := Plugin{}.BeginRun(context.Background(), runInfo())
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	if ext != nil {
		t.Fatalf("expected nil extension without a store, got %T", ext)
	}
}

// A store that cannot revise entries still gets recall and learn, but no edit.
func TestNonCuratorStoreGetsRecallAndLearn(t *testing.T) {
	ext, err := Plugin{Store: &fakeStore{}}.BeginRun(context.Background(), runInfo())
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	tc, ok := ext.(agentcore.ToolContributor)
	if !ok {
		t.Fatal("extension does not contribute tools")
	}
	names := map[string]bool{}
	for _, tool := range tc.Tools() {
		names[tool.Name()] = true
	}
	if !names[ToolMemoryRecall] {
		t.Error("memory_recall missing for a store that can Recall")
	}
	if !names[ToolLearn] {
		t.Error("learn missing for a store that can Remember")
	}
	if names[ToolMemoryEdit] {
		t.Error("memory_edit offered for a store that cannot revise entries")
	}
}

// learn files the lesson under the RUN's scope as a learning — the model
// supplies neither.
func TestLearnPinsScopeAndKind(t *testing.T) {
	store := &fakeCurator{fakeStore: &fakeStore{}}
	ext, err := Plugin{Store: store}.BeginRun(context.Background(), runInfo())
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	var learn agentcore.Tool
	for _, tool := range ext.(agentcore.ToolContributor).Tools() {
		if tool.Name() == ToolLearn {
			learn = tool
		}
	}
	if learn == nil {
		t.Fatal("learn tool not contributed")
	}
	if _, err := learn.Run(context.Background(), `{"lesson":"always check the fence","tags":["governance"]}`); err != nil {
		t.Fatalf("learn: %v", err)
	}
	if len(store.remembered) != 1 {
		t.Fatalf("remembered %d entries, want 1", len(store.remembered))
	}
	got := store.remembered[0]
	if got.ScopeID != "agent-1" || got.Kind != agentcore.MemoryLearning || got.Content != "always check the fence" {
		t.Errorf("unexpected entry: %+v", got)
	}
	if _, err := learn.Run(context.Background(), `{"lesson":""}`); err == nil {
		t.Error("empty lesson accepted")
	}
}

// memory_edit update/forget reach the curator under the run's scope; an id in
// another scope is refused, and an unknown action is rejected before the store.
func TestMemoryEditRoundTripAndScopeFence(t *testing.T) {
	store := &fakeCurator{fakeStore: &fakeStore{recall: []agentcore.MemoryEntry{
		{ID: "m1", ScopeID: "agent-1", Content: "old"},
		{ID: "m2", ScopeID: "agent-2", Content: "not mine"},
	}}}
	ext, err := Plugin{Store: store}.BeginRun(context.Background(), runInfo())
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	var edit agentcore.Tool
	for _, tool := range ext.(agentcore.ToolContributor).Tools() {
		if tool.Name() == ToolMemoryEdit {
			edit = tool
		}
	}
	if edit == nil {
		t.Fatal("memory_edit tool not contributed")
	}
	ctx := context.Background()

	if _, err := edit.Run(ctx, `{"id":"m1","action":"update","content":"new wording"}`); err != nil {
		t.Fatalf("update: %v", err)
	}
	if len(store.updates) != 1 || store.updates[0].Content != "new wording" || store.updates[0].ScopeID != "agent-1" {
		t.Fatalf("unexpected update: %+v", store.updates)
	}
	if _, err := edit.Run(ctx, `{"id":"m1","action":"forget"}`); err != nil {
		t.Fatalf("forget: %v", err)
	}
	if len(store.superseded) != 1 || store.superseded[0] != "m1" {
		t.Fatalf("unexpected supersede: %v", store.superseded)
	}

	// Cross-scope: m2 belongs to agent-2, the run is agent-1.
	if _, err := edit.Run(ctx, `{"id":"m2","action":"forget"}`); !errors.Is(err, errWrongScope) {
		t.Fatalf("cross-scope forget: got %v, want refusal", err)
	}
	if _, err := edit.Run(ctx, `{"id":"m2","action":"update","content":"x"}`); !errors.Is(err, errWrongScope) {
		t.Fatalf("cross-scope update: got %v, want refusal", err)
	}

	// Bad input never reaches the store.
	if _, err := edit.Run(ctx, `{"id":"m1","action":"delete"}`); err == nil {
		t.Error("unknown action accepted")
	}
	if _, err := edit.Run(ctx, `{"id":"m1","action":"update"}`); err == nil {
		t.Error("update without content accepted")
	}
	if len(store.updates) != 1 || len(store.superseded) != 1 {
		t.Error("refused calls mutated the store")
	}
}
