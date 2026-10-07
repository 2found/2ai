package subagent_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/subagent"
	"github.com/2found/2ai/ai"
)

// The plugin binds governed forks to the native session host. Journal recovery,
// completed-child reattachment and interrupted-child resume are exercised by
// internal/runtime/pi_children_integration_test.go against the actual host.
func durableParent(t *testing.T, settings subagent.Plugin) *agentcore.Agent {
	t.Helper()
	a, err := agentcore.New(agentcore.Config{NativeProvider: nativeProvider(ai.ScriptedStream()), Model: "test", Session: agentcore.NewMemorySessionStore(), SessionID: "parent", Policy: agentcore.NewAllowList(subagent.ToolSpawnSubagent), Extensions: []agentcore.ExtensionFactory{settings}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func executeSpawn(t *testing.T, host *agentcore.PiToolHost, scope, args string) agentcore.PiToolOutcome {
	t.Helper()
	ctx := agentcore.WithToolInvocationScope(context.Background(), scope)
	params, _ := json.Marshal(map[string]any{"toolCallId": "reused-call", "toolName": subagent.ToolSpawnSubagent, "args": json.RawMessage(args)})
	_, audit, err := host.Execute(ctx, params, nil)
	if err != nil {
		t.Fatal(err)
	}
	return audit
}

func TestDurableChildIdentityUsesPhysicalInvocation(t *testing.T) {
	var sessions []string
	settings := subagent.Plugin{RunFork: func(ctx context.Context, child *agentcore.Agent, req subagent.ForkRequest, _ agentcore.StreamSink) (agentcore.RunResult, error) {
		key, ok := agentcore.IdempotencyKey(ctx)
		if !ok || req.SessionID != "parent/"+key || child.SessionID() != req.SessionID || !child.IsDurable() || agentcore.DelegationDepth(ctx) != 1 {
			t.Errorf("fork lost native identity/depth: key=%q request=%+v child=%s", key, req, child.SessionID())
		}
		sessions = append(sessions, req.SessionID)
		return agentcore.RunResult{Final: "child answer", StopReason: "stop"}, nil
	}}
	host, err := durableParent(t, settings).OpenPiTools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	for _, scope := range []string{"effect-one", "effect-two", "effect-one"} {
		if audit := executeSpawn(t, host, scope, `{"task":"inspect"}`); audit.Trace.Error != "" {
			t.Fatal(audit.Trace.Error)
		}
	}
	if len(sessions) != 3 || sessions[0] == sessions[1] || sessions[0] != sessions[2] {
		t.Fatalf("provider ID confused new invocation with replay: %v", sessions)
	}
}

func TestDurableChildRequiresNativeHostAndRecordedInvocation(t *testing.T) {
	for _, mode := range []string{"missing host", "missing invocation"} {
		t.Run(mode, func(t *testing.T) {
			calls := 0
			settings := subagent.Plugin{}
			if mode == "missing invocation" {
				settings.RunFork = func(context.Context, *agentcore.Agent, subagent.ForkRequest, agentcore.StreamSink) (agentcore.RunResult, error) {
					calls++
					return agentcore.RunResult{}, nil
				}
			}
			parent := durableParent(t, settings)
			ext, err := settings.BeginRun(context.Background(), agentcore.RunInfo{Agent: parent, Durable: true})
			if err != nil {
				t.Fatal(err)
			}
			tool := ext.(agentcore.ToolContributor).Tools()[0]
			_, err = tool.Run(context.Background(), `{"task":"inspect"}`)
			if err == nil || calls != 0 || !strings.Contains(err.Error(), "requires") {
				t.Fatalf("unsafe durable fallback: calls=%d err=%v", calls, err)
			}
		})
	}
}

func TestDurableSchemaCorrectionPassesOriginalNativeCheckpoint(t *testing.T) {
	const opaque = `{"messages":[{"role":"assistant","opaque":"preserve-me"}]}`
	var firstID string
	calls := 0
	settings := subagent.Plugin{RunFork: func(_ context.Context, _ *agentcore.Agent, req subagent.ForkRequest, _ agentcore.StreamSink) (agentcore.RunResult, error) {
		calls++
		if calls == 1 {
			firstID = req.SessionID
			return agentcore.RunResult{Final: "not JSON", NativeState: json.RawMessage(opaque), NativeRevision: "test-revision", Messages: []agentcore.Message{{Role: agentcore.RoleAssistant, Content: "display-only"}}}, nil
		}
		if req.SessionID != firstID+"/retry" || req.Previous == nil || string(req.Previous.NativeState) != opaque || req.Previous.NativeRevision != "test-revision" || !strings.Contains(req.Prompt, "failed output_schema validation") {
			t.Errorf("correction rebuilt or lost native state: %+v", req)
		}
		return agentcore.RunResult{Final: `{"fruit":"banana"}`, StopReason: "stop"}, nil
	}}
	host, err := durableParent(t, settings).OpenPiTools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	audit := executeSpawn(t, host, "effect", `{"task":"name a fruit","output_schema":`+fruitSchema+`}`)
	if calls != 2 || audit.Trace.Error != "" {
		t.Fatalf("correction failed: calls=%d audit=%+v", calls, audit)
	}
}

func TestSubagentStorelessParentKeepsEphemeralChild(t *testing.T) {
	agent, _ := subagentAgent(t, &subagent.Plugin{},
		AssistantToolCall("c1", subagent.ToolSpawnSubagent, `{"task":"echo banana and report"}`),
		AssistantToolCall("c2", "echo", `{"text":"banana"}`),
		nativeAnswer("the echo returned: banana"), nativeAnswer("child reported: banana"))
	res, err := agent.Prompt(context.Background(), "delegate")
	if err != nil || res.Final != "child reported: banana" {
		t.Fatalf("ephemeral child: %q %v", res.Final, err)
	}
}
