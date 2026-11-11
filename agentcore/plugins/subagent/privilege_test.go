package subagent_test

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/jobs"
	"github.com/2found/2ai/agentcore/plugins/subagent"
	"github.com/2found/2ai/ai"
)

// countingTool records whether it ever ran. A privilege test asserts on
// execution, not on what the model was told.
type countingTool struct {
	echoTool
	runs atomic.Int32
}

func (t *countingTool) Run(ctx context.Context, args string) (string, error) {
	t.runs.Add(1)
	return t.echoTool.Run(ctx, args)
}

// routedProvider answers the parent and the child from separate scripts, so an
// async child (which runs concurrently with its parent) cannot steal the
// parent's next reply. A request is the child's when its task marker appears.
func routedProvider(marker string, parent, child []ai.Message) (*ai.FallbackProvider, func() [][]string) {
	var mu sync.Mutex
	var childTools [][]string
	parentScript, childScript := ai.ScriptedStream(parent...), ai.ScriptedStream(child...)
	return nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
			raw, _ := json.Marshal(view.Messages())
			if !strings.Contains(string(raw), marker) || strings.Contains(string(raw), "parent prompt") {
				return parentScript(ctx, model, view, options)
			}
			var names []string
			for _, tool := range ai.GetCurrentTools(view.Messages()) {
				names = append(names, tool.Name)
			}
			mu.Lock()
			childTools = append(childTools, names)
			mu.Unlock()
			return childScript(ctx, model, view, options)
		}), func() [][]string {
			mu.Lock()
			defer mu.Unlock()
			return append([][]string(nil), childTools...)
		}
}

// A self-fork runs with exactly the parent's privileges: a tool the parent
// holds but its policy denies stays denied in the child, a tool the parent
// does not hold does not exist in the child, and the async path is governed
// the same way as the synchronous one.
func TestSelfForkCannotWidenPrivilege(t *testing.T) {
	for _, async := range []bool{false, true} {
		t.Run(map[bool]string{false: "sync", true: "async"}[async], func(t *testing.T) {
			secret := &countingTool{echoTool: echoTool{name: "secret"}}
			spawn := `{"task":"child-task: read the secret"}`
			parentScript := []ai.Message{nativeCall("p1", subagent.ToolSpawnSubagent, spawn), nativeAnswer("parent done")}
			if async {
				spawn = `{"task":"child-task: read the secret","async":true}`
				parentScript = []ai.Message{nativeCall("p1", subagent.ToolSpawnSubagent, spawn), nativeCall("p2", "job_wait", `{"id":"job_1"}`), nativeAnswer("parent done")}
			}
			provider, childTools := routedProvider("child-task", parentScript, []ai.Message{
				nativeCall("c1", "secret", `{"text":"leak"}`),
				nativeCall("c2", "ghost", `{"text":"leak"}`),
				nativeAnswer("child could not read it"),
			})
			agent, err := agentcore.New(agentcore.Config{
				NativeProvider: provider, Model: "faux-1",
				Tools:      agentcore.NewToolSet(&echoTool{name: "echo"}, secret),
				Policy:     agentcore.NewAllowList("echo", subagent.ToolSpawnSubagent),
				Extensions: []agentcore.ExtensionFactory{&subagent.Plugin{AllowAsync: true}, jobs.Plugin{}},
			})
			if err != nil {
				t.Fatal(err)
			}
			res, err := agent.Prompt(context.Background(), "parent prompt")
			if err != nil || res.Final != "parent done" {
				t.Fatalf("run: final=%q err=%v", res.Final, err)
			}
			if n := secret.runs.Load(); n != 0 {
				t.Fatalf("child executed a tool the parent's policy denies (%d runs)", n)
			}
			seen := childTools()
			if len(seen) == 0 {
				t.Fatal("the child never ran")
			}
			for _, names := range seen {
				for _, name := range names {
					if name == "secret" || name == "ghost" {
						t.Fatalf("child was offered %q, which the parent may not call", name)
					}
				}
			}
		})
	}
}

// A named delegate composed with a LARGER MaxDepth than the agent that handed
// it the task must not delegate deeper than that agent allows: the cap is the
// tightest of every delegating ancestor.
func TestDelegateCannotRaiseTheDepthCap(t *testing.T) {
	var delegateTools [][]string
	var mu sync.Mutex
	teammate := func(t *testing.T) *agentcore.Agent {
		t.Helper()
		script := ai.ScriptedStream(nativeCall("d1", subagent.ToolSpawnSubagent, `{"task":"grandchild"}`), nativeAnswer("teammate answer"), nativeAnswer("teammate answer"))
		a, err := agentcore.New(agentcore.Config{
			NativeProvider: nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
				var names []string
				for _, tool := range ai.GetCurrentTools(view.Messages()) {
					names = append(names, tool.Name)
				}
				mu.Lock()
				delegateTools = append(delegateTools, names)
				mu.Unlock()
				return script(ctx, model, view, options)
			}),
			Model:      "faux-1",
			Policy:     agentcore.NewAllowList(subagent.ToolSpawnSubagent),
			Extensions: []agentcore.ExtensionFactory{&subagent.Plugin{MaxDepth: 5}},
		})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}(t)
	var grandchildren atomic.Int32
	parent, _ := subagentAgent(t, &subagent.Plugin{MaxDepth: 1, Delegates: []subagent.Delegate{{
		Name: "teammate",
		Run: func(ctx context.Context, task string, _ agentcore.StreamSink) (string, agentcore.Usage, error) {
			if agentcore.DelegationDepth(ctx) > 1 {
				grandchildren.Add(1)
			}
			res, err := teammate.Prompt(ctx, task)
			return res.Final, res.Usage, err
		},
	}}},
		nativeCall("p1", subagent.ToolSpawnSubagent, `{"task":"ask the teammate","agent":"teammate"}`),
		nativeAnswer("parent done"),
	)
	res, err := parent.Prompt(context.Background(), "parent prompt")
	if err != nil || res.Final != "parent done" {
		t.Fatalf("run: final=%q err=%v", res.Final, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(delegateTools) == 0 {
		t.Fatal("the teammate never ran")
	}
	for _, names := range delegateTools {
		for _, name := range names {
			if name == subagent.ToolSpawnSubagent {
				t.Fatal("the teammate was offered spawn_subagent past the delegating agent's MaxDepth")
			}
		}
	}
	if grandchildren.Load() != 0 {
		t.Fatal("a grandchild was delegated past the cap")
	}
}

// A tighter cap downstream still applies: the ceiling only narrows.
func TestDelegateKeepsItsOwnTighterDepthCap(t *testing.T) {
	// Parent allows depth 3; the teammate allows only 1, so at depth 1 it is
	// already at its own cap and must not be offered the tool.
	offered := false
	teammate, err := agentcore.New(agentcore.Config{
		NativeProvider: nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
			for _, tool := range ai.GetCurrentTools(view.Messages()) {
				offered = offered || tool.Name == subagent.ToolSpawnSubagent
			}
			return ai.ScriptedStream(nativeAnswer("teammate answer"))(ctx, model, view, options)
		}),
		Model:      "faux-1",
		Policy:     agentcore.NewAllowList(subagent.ToolSpawnSubagent),
		Extensions: []agentcore.ExtensionFactory{&subagent.Plugin{MaxDepth: 1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parent, _ := subagentAgent(t, &subagent.Plugin{MaxDepth: 3, Delegates: []subagent.Delegate{{
		Name: "teammate",
		Run: func(ctx context.Context, task string, _ agentcore.StreamSink) (string, agentcore.Usage, error) {
			res, err := teammate.Prompt(ctx, task)
			return res.Final, res.Usage, err
		},
	}}},
		nativeCall("p1", subagent.ToolSpawnSubagent, `{"task":"ask","agent":"teammate"}`),
		nativeAnswer("parent done"),
	)
	if res, err := parent.Prompt(context.Background(), "parent prompt"); err != nil || res.Final != "parent done" {
		t.Fatalf("run: final=%q err=%v", res.Final, err)
	}
	if offered {
		t.Fatal("the teammate's own tighter cap was widened by the parent's")
	}
}
