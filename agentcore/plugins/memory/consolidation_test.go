package memory

import (
	"context"
	"errors"
	"testing"

	"encoding/json"
	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

type consolidationStore struct {
	pending []Rollout
	commits int
	entries []agentcore.MemoryEntry
}

func (s *consolidationStore) Recall(context.Context, string, string, int) ([]agentcore.MemoryEntry, error) {
	return s.entries, nil
}
func (s *consolidationStore) Remember(context.Context, agentcore.MemoryEntry) error { return nil }
func (s *consolidationStore) StageRollout(_ context.Context, _ string, r Rollout) error {
	for _, p := range s.pending {
		if p.ID == r.ID {
			return nil
		}
	}
	s.pending = append(s.pending, r)
	return nil
}
func (s *consolidationStore) PendingRollouts(context.Context, string, int) ([]Rollout, error) {
	return s.pending, nil
}
func (s *consolidationStore) CommitConsolidation(_ context.Context, _ string, _ Consolidation, changes []Change) error {
	s.commits++
	s.pending = nil
	for _, c := range changes {
		if c.Entry != nil {
			s.entries = append(s.entries, *c.Entry)
		}
	}
	return nil
}
func TestConsolidationFailureLeavesRolloutPendingAndScopesChanges(t *testing.T) {
	store := &consolidationStore{}
	calls := 0
	reports := 0
	plugin := Plugin{Store: store, OnConsolidationError: func(context.Context, error) { reports++ }, Consolidator: func(_ context.Context, in Consolidation) ([]Change, error) {
		calls++
		if len(in.Rollouts) == 0 {
			t.Fatal("missing evidence")
		}
		if calls == 1 {
			return nil, errors.New("offline")
		}
		return []Change{{Entry: &agentcore.MemoryEntry{Content: "Use verified evidence.", ScopeID: "other"}}}, nil
	}}
	ext, _ := plugin.BeginRun(context.Background(), agentcore.RunInfo{ScopeID: "own"})
	c := ext.(*curation)
	result := agentcore.RunResult{Turns: 1, StopReason: "stop", Final: "done", Messages: []agentcore.Message{{Role: agentcore.RoleAssistant, Content: "evidence"}}}
	c.FinalizeRun(context.Background(), result, nil)
	c.FinalizeRun(context.Background(), result, nil)
	if len(store.pending) != 1 || store.commits != 0 || reports != 2 {
		t.Fatal("failure consumed evidence or widened scope")
	}
	c.consolidator = func(context.Context, Consolidation) ([]Change, error) {
		return []Change{{Entry: &agentcore.MemoryEntry{Content: "Use verified evidence."}}}, nil
	}
	c.FinalizeRun(context.Background(), result, nil)
	if store.commits != 1 || len(store.pending) != 0 || len(store.entries) != 1 {
		t.Fatal("retry did not consolidate")
	}
}
func TestNativeConsolidationAccountsUsageAndSeesCompletedTranscript(t *testing.T) {
	store := &consolidationStore{}
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: `{"changes":[{"ids":[],"entry":{"content":"Validate before publishing."}}]}`}), Usage: &ai.Usage{Input: 7, Output: 3}})}}}
	a, err := agentcore.New(agentcore.Config{NativeProvider: &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "verified"}), Usage: &ai.Usage{Input: 2, Output: 1}})}}}, Model: "test", Definition: agentcore.AgentDefinition{ScopeID: "own"}, Extensions: []agentcore.ExtensionFactory{Plugin{Store: store, NativeProvider: provider}}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "work"}}})
	if err != nil || result.Usage.InputTokens != 9 || result.Usage.OutputTokens != 4 || store.commits != 1 {
		t.Fatalf("consolidation: usage=%+v commits=%d err=%v", result.Usage, store.commits, err)
	}
}
