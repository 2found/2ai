package integration

// End-to-end proof of the memory-curation contract through the real loop:
// the model calls learn / memory_edit, the calls pass the permission gate,
// the store applies them under the run's scope, and the NEXT run's recalled
// block shows the result — including the entry ids the edit tool needs.

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/memory"
	"github.com/2found/2ai/ai"
)

// memStore is an in-memory MemoryStore + MemoryCurator with the shipped
// store's semantics: supersede keeps the row and filters it from recall.
type memStore struct {
	entries []agentcore.MemoryEntry
	live    map[string]bool // id -> not superseded
	next    int
}

func newMemStore() *memStore { return &memStore{live: map[string]bool{}} }

func (s *memStore) Remember(_ context.Context, e agentcore.MemoryEntry) error {
	s.next++
	e.ID = fmt.Sprintf("mem-%d", s.next)
	s.entries = append(s.entries, e)
	s.live[e.ID] = true
	return nil
}

func (s *memStore) Recall(_ context.Context, scopeID, _ string, limit int) ([]agentcore.MemoryEntry, error) {
	var out []agentcore.MemoryEntry
	for i := len(s.entries) - 1; i >= 0 && len(out) < limit; i-- {
		e := s.entries[i]
		if e.ScopeID == scopeID && s.live[e.ID] {
			out = append(out, e)
		}
	}
	return out, nil
}

func (s *memStore) Supersede(_ context.Context, scopeID, id, _ string) error {
	for _, e := range s.entries {
		if e.ID == id && e.ScopeID == scopeID && s.live[id] {
			s.live[id] = false
			return nil
		}
	}
	return fmt.Errorf("agent memory not found in scope")
}

func (s *memStore) Update(_ context.Context, scopeID, id string, e agentcore.MemoryEntry) error {
	for _, old := range s.entries {
		if old.ID == id && old.ScopeID == scopeID && s.live[id] {
			s.next++
			e.ID = fmt.Sprintf("mem-%d", s.next)
			e.ScopeID = scopeID
			e.Kind = old.Kind // the successor inherits the kind
			s.entries = append(s.entries, e)
			s.live[e.ID] = true
			s.live[id] = false
			return nil
		}
	}
	return fmt.Errorf("agent memory not found in scope")
}

func (s *memStore) CreateSession(context.Context, string) (agentcore.Session, error) {
	return agentcore.Session{}, nil
}
func (s *memStore) SaveSession(context.Context, agentcore.Session) error { return nil }
func (s *memStore) Fork(context.Context, string) (agentcore.Session, error) {
	return agentcore.Session{}, nil
}

var (
	_ agentcore.MemoryStore   = (*memStore)(nil)
	_ agentcore.MemoryCurator = (*memStore)(nil)
)

func buildMemoryAgent(t *testing.T, store agentcore.MemoryStore, pol agentcore.Policy, provider *ai.FallbackProvider) *agentcore.Agent {
	t.Helper()
	a, err := agentcore.Build(
		agentcore.ModelPlugin{NativeProvider: provider, Model: "test"},
		agentcore.DefinitionPlugin{Definition: agentcore.AgentDefinition{ScopeID: "agent-1"}},
		agentcore.PolicyPlugin{Policy: pol},
		memory.Plugin{Store: store},
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	return a
}

// memoryText returns the first request's system prompt and recalled-memory
// context message: where a run shows the model its recalled memory.
func memoryText(rec *nativeRecorder) string {
	var sb strings.Builder
	for _, m := range rec.all()[0].Messages {
		if m.Role == agentcore.RoleSystem || strings.HasPrefix(m.Content, "# Recalled memory\n") {
			sb.WriteString(m.Content)
		}
	}
	return sb.String()
}

// The primary journey: learn files a lesson, the next run recalls it WITH its
// id, memory_edit updates then forgets it, and a third run's recall shows only
// the superseding wording — then nothing.
func TestMemoryCurationEndToEnd(t *testing.T) {
	ctx := context.Background()
	store := newMemStore()
	allow := agentcore.NewAllowList(memory.ToolLearn, memory.ToolMemoryEdit)

	// Run 1: the model learns a lesson.
	p1 := scriptedNativeProvider(
		agentcore.AssistantToolCall("c1", "learn", `{"lesson":"deploys moved to Thursday","tags":["deploy"]}`),
		agentcore.AssistantText("noted"),
	)
	if _, err := buildMemoryAgent(t, store, allow, p1).Prompt(ctx, "remember this"); err != nil {
		t.Fatalf("run 1: %v", err)
	}
	if len(store.entries) != 1 || store.entries[0].Kind != agentcore.MemoryLearning || store.entries[0].ScopeID != "agent-1" {
		t.Fatalf("learn wrote %+v", store.entries)
	}
	learnedID := store.entries[0].ID

	// Run 2: recall shows the lesson with its id; the model updates it, then
	// forgets the successor, then tries a second forget on the retracted id.
	recorder2 := &nativeRecorder{}
	p2 := recordedNativeProvider(recorder2,
		agentcore.AssistantToolCall("c1", "memory_edit", fmt.Sprintf(`{"id":%q,"action":"update","content":"deploys moved to Wednesday"}`, learnedID)),
		agentcore.AssistantToolCall("c2", "memory_edit", `{"id":"mem-2","action":"forget"}`),
		agentcore.AssistantToolCall("c3", "memory_edit", `{"id":"mem-2","action":"forget"}`),
		agentcore.AssistantText("done"),
	)
	res, err := buildMemoryAgent(t, store, allow, p2).Prompt(ctx, "fix that memory")
	if err != nil {
		t.Fatalf("run 2: %v", err)
	}

	// The recalled block carried the id — the model could only have passed it
	// back if the prompt rendered it.
	if sys := memoryText(recorder2); !strings.Contains(sys, "id "+learnedID) {
		t.Fatalf("recalled block did not render the entry id:\n%s", sys)
	}

	// update produced a live successor; forget retracted it; the second forget
	// was refused by the store (already retracted).
	var updateOK, forgetOK, reforgetRefused bool
	for _, tr := range res.Tools {
		switch tr.CallID {
		case "c1":
			updateOK = tr.Allowed && tr.Error == ""
		case "c2":
			forgetOK = tr.Allowed && tr.Error == ""
		case "c3":
			reforgetRefused = tr.Error != "" || strings.Contains(tr.Error, "not found")
		}
	}
	if !updateOK || !forgetOK {
		t.Fatalf("update/forget did not execute: %+v", res.Tools)
	}
	if !reforgetRefused {
		t.Error("re-forgetting a retracted memory was not refused")
	}
	if len(store.entries) != 2 {
		t.Fatalf("update should keep the old row as history, got %d rows", len(store.entries))
	}

	// Run 3: recall is empty — every row is retracted.
	recorder3 := &nativeRecorder{}
	p3 := recordedNativeProvider(recorder3, agentcore.AssistantText("ok"))
	if _, err := buildMemoryAgent(t, store, allow, p3).Prompt(ctx, "what do you know"); err != nil {
		t.Fatalf("run 3: %v", err)
	}
	if strings.Contains(memoryText(recorder3), "deploys moved") {
		t.Error("a retracted memory still appears in recall")
	}
}

// A run whose policy does not permit the tools never sees them advertised,
// and a direct call is blocked by the gate.
func TestMemoryCurationGatedByPolicy(t *testing.T) {
	ctx := context.Background()
	store := newMemStore()
	recorder := &nativeRecorder{}
	p := recordedNativeProvider(recorder,
		agentcore.AssistantToolCall("c1", "learn", `{"lesson":"x"}`),
		agentcore.AssistantText("done"),
	)
	res, err := buildMemoryAgent(t, store, agentcore.NewAllowList("run_sql"), p).Prompt(ctx, "learn something")
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	for _, name := range recorder.all()[0].Tools {
		if name == memory.ToolLearn || name == memory.ToolMemoryEdit {
			t.Errorf("%q advertised to a run whose policy denies it", name)
		}
	}
	if len(res.Tools) != 1 || res.Tools[0].Allowed {
		t.Fatalf("a denied learn call was not blocked: %+v", res.Tools)
	}
	if len(store.entries) != 0 {
		t.Error("a blocked learn call still wrote memory")
	}
}

// A run with no memory store gets neither tool — the plugin declines.
func TestMemoryCurationAbsentWithoutStore(t *testing.T) {
	ctx := context.Background()
	recorder := &nativeRecorder{}
	p := recordedNativeProvider(recorder, agentcore.AssistantText("ok"))
	_, err := buildMemoryAgent(t, nil, agentcore.NewAllowList(memory.ToolLearn, memory.ToolMemoryEdit), p).Prompt(ctx, "hi")
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	for _, name := range recorder.all()[0].Tools {
		if name == memory.ToolLearn || name == memory.ToolMemoryEdit {
			t.Errorf("%q advertised on a memoryless run", name)
		}
	}
}

// Recall is a per-task query, so its result differs between the runs of one
// conversation. It must not reach the system prompt — the prefix every request
// shares — or each new run rewrites the provider's prompt cache for the whole
// transcript. The block rides a transcript message instead, and a bullet the
// transcript already carries is not repeated.
func TestRecalledMemoryKeepsTheRequestPrefixStable(t *testing.T) {
	ctx := context.Background()
	store := newMemStore()
	_ = store.Remember(ctx, agentcore.MemoryEntry{ScopeID: "agent-1", Kind: agentcore.MemoryFact, Content: "the user prefers blue"})
	allow := agentcore.NewAllowList()

	rec1 := &nativeRecorder{}
	first, err := buildMemoryAgent(t, store, allow, recordedNativeProvider(rec1, agentcore.AssistantText("one"))).RunNative(ctx, agentcore.NativeRun{Task: "first", Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "first"}}})
	if err != nil {
		t.Fatalf("run 1: %v", err)
	}
	_ = store.Remember(ctx, agentcore.MemoryEntry{ScopeID: "agent-1", Kind: agentcore.MemoryFact, Content: "the user writes Go"})
	rec2 := &nativeRecorder{}
	if _, err := buildMemoryAgent(t, store, allow, recordedNativeProvider(rec2, agentcore.AssistantText("two"))).RunNative(ctx, agentcore.NativeRun{State: first.NativeState, Task: "second", Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "second"}}}); err != nil {
		t.Fatalf("run 2: %v", err)
	}

	// Providers collapse every system message into the head; the scripted
	// fixture keeps each run's copy in place, so compare without them.
	// The head becomes the latest system message, as the collapse keeps it.
	conversation := func(messages []agentcore.Message) (out []agentcore.Message) {
		out = []agentcore.Message{{Role: agentcore.RoleSystem}}
		for _, m := range messages {
			if m.Role == agentcore.RoleSystem {
				out[0] = m
			} else {
				out = append(out, m)
			}
		}
		return out
	}
	req1, req2 := conversation(rec1.all()[0].Messages), conversation(rec2.all()[0].Messages)
	if req1[0].Role != agentcore.RoleSystem || req2[0].Role != agentcore.RoleSystem || req1[0].Content != req2[0].Content {
		t.Fatalf("system prompt changed between runs:\n%q\n%q", req1[0].Content, req2[0].Content)
	}
	if strings.Contains(req1[0].Content, "Recalled memory") {
		t.Fatalf("recalled memory is in the system prompt:\n%s", req1[0].Content)
	}
	// Run 1: system, memory, input.
	if len(req1) != 3 || !strings.HasPrefix(req1[1].Content, "# Recalled memory\n") || !strings.Contains(req1[1].Content, "id mem-1) the user prefers blue") || req1[2].Content != "first" {
		t.Fatalf("run 1 request = %+v", req1)
	}
	// Run 2 extends run 1's request (plus its answer) instead of rewriting it.
	for i := range req1 {
		if req2[i].Role != req1[i].Role || req2[i].Content != req1[i].Content {
			t.Fatalf("run 2 request is not an extension of run 1 at message %d: %+v", i, req2[i])
		}
	}
	tail := req2[len(req1):]
	if len(tail) != 3 || tail[0].Content != "one" || tail[2].Content != "second" {
		t.Fatalf("run 2 tail = %+v", tail)
	}
	if !strings.Contains(tail[1].Content, "the user writes Go") || strings.Contains(tail[1].Content, "prefers blue") {
		t.Fatalf("run 2 memory should add only the new entry:\n%s", tail[1].Content)
	}
}
