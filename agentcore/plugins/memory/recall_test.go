package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
)

type recallProbe struct {
	agentcore.MemoryStore
	scope, query string
	limit, calls int
	err          error
}

func (p *recallProbe) Recall(_ context.Context, scope, query string, limit int) ([]agentcore.MemoryEntry, error) {
	p.scope, p.query, p.limit = scope, query, limit
	p.calls++
	if p.err != nil {
		return nil, p.err
	}
	entries := []agentcore.MemoryEntry{{ID: "secret", ScopeID: "another-agent", Content: "must not leak"}}
	for i := 0; i < 30; i++ {
		entries = append(entries, agentcore.MemoryEntry{ID: fmt.Sprint(i), ScopeID: scope, Kind: agentcore.MemoryLearning, Content: strings.Repeat("x", 4000)})
	}
	return entries, nil
}

func TestRecallIsScopedBoundedAndReadOnly(t *testing.T) {
	probe := &recallProbe{}
	ext, err := (Plugin{Store: probe}).BeginRun(context.Background(), runInfo())
	if err != nil {
		t.Fatal(err)
	}
	var tool agentcore.Tool
	for _, candidate := range ext.(agentcore.ToolContributor).Tools() {
		if candidate.Name() == ToolMemoryRecall {
			tool = candidate
		}
	}
	if tool == nil {
		t.Fatal("memory recall not contributed by extension")
	}
	raw, err := tool.Run(context.Background(), `{"query":"  staging rollout  ","limit":2}`)
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Entries []struct {
			ID, Content string
			Truncated   bool
		}
	}
	if err = json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatal(err)
	}
	if probe.scope != "agent-1" || probe.query != "staging rollout" || probe.limit != 2 || len(result.Entries) != 2 || strings.Contains(raw, "must not leak") {
		t.Fatalf("scope/bounds lost: %+v %s", probe, raw)
	}
	if result.Entries[0].ID != "0" || len(result.Entries[0].Content) > 2000 || !result.Entries[0].Truncated {
		t.Fatalf("recall result not bounded/identified: %s", raw)
	}
	for _, args := range []string{`{"query":""}`, `{"query":"x","limit":0}`, `{"query":"x","limit":21}`, `{"query":"x","scope":"another-agent"}`} {
		before := probe.calls
		if _, err := tool.Run(context.Background(), args); err == nil || probe.calls != before {
			t.Fatalf("invalid input reached memory store: %s %v", args, err)
		}
	}
	probe.err = errors.New("store unavailable")
	if _, err := tool.Run(context.Background(), `{"query":"x"}`); !errors.Is(err, probe.err) || probe.limit != 8 {
		t.Fatalf("default limit or store error lost: %+v %v", probe, err)
	}
}
