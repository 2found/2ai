package preset_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/preset"
	"github.com/2found/2ai/agentcore/plugins/spill"
	"github.com/2found/2ai/ai"
)

func TestNativeRunObservationToolsRemainAvailableUnderWorkloadPolicy(t *testing.T) {
	// A host grants no workload tools. Installing run-owned observation
	// capabilities must still advertise them without granting unrelated work.
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"fixture"}`), Stream: ai.ScriptedStream()}}}
	a, err := agentcore.Build(preset.Full(agentcore.Config{NativeProvider: provider, Model: "fixture", Tools: agentcore.NewToolSet(&echoTool{name: "ungranted"}), Policy: agentcore.NewAllowList()}, preset.Options{Spill: spill.NewMemorySpillStore(), NativeHistory: true})...)
	if err != nil {
		t.Fatal(err)
	}
	host, err := a.OpenPiTools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	raw, err := host.Definitions(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var definitions []struct{ Name string }
	if err := json.Unmarshal(raw, &definitions); err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, tool := range definitions {
		names[tool.Name] = true
	}
	for _, name := range []string{"job_list", "job_status", "job_wait", "job_cancel", "read_spill", "session_query"} {
		if !names[name] {
			t.Errorf("run-owned tool %s absent from native registry", name)
		}
	}
	if names["ungranted"] {
		t.Fatal("observation exemptions widened workload grants")
	}
}
