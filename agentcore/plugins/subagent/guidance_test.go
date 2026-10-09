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

// Observe the real native provider boundary, not a reimplementation of the
// plugin. Guidance must follow the same gates as the capability it describes.
func TestDelegationGuidanceFollowsAdvertisedCapabilities(t *testing.T) {
	for _, tc := range []struct {
		name                      string
		installed, granted, named bool
		async, durable            bool
		depth                     int
	}{
		{name: "absent", granted: true},
		{name: "denied", installed: true, named: true, async: true},
		{name: "depth cap", installed: true, granted: true, named: true, async: true, depth: 1},
		{name: "self sync", installed: true, granted: true},
		{name: "self async", installed: true, granted: true, async: true},
		{name: "named sync", installed: true, granted: true, named: true},
		{name: "named async", installed: true, granted: true, named: true, async: true},
		{name: "durable named", installed: true, granted: true, named: true, async: true, durable: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var prompt string
			var offered []agentcore.ToolSchema
			provider := nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
				prompt = ai.GetCurrentSystemPrompt(view.Messages())
				raw, err := json.Marshal(ai.GetCurrentTools(view.Messages()))
				if err != nil {
					return nil, err
				}
				if err = json.Unmarshal(raw, &offered); err != nil {
					return nil, err
				}
				return ai.ScriptedStream(nativeAnswer("direct answer"))(ctx, model, view, options)
			})
			cfg := agentcore.Config{NativeProvider: provider, Model: "test", Definition: agentcore.AgentDefinition{Agents: "Host-authored mission."}}
			if tc.granted {
				cfg.Policy = agentcore.NewAllowList(subagent.ToolSpawnSubagent)
			}
			if tc.installed {
				plugin := subagent.Plugin{AllowAsync: tc.async}
				if tc.named {
					plugin.Delegates = []subagent.Delegate{{Name: "ledger", Description: "Inspect approved ledger entries", Run: func(context.Context, string, agentcore.StreamSink) (string, agentcore.Usage, error) {
						t.Error("advertising a delegate executed it")
						return "", agentcore.Usage{}, nil
					}}}
				}
				cfg.Extensions = []agentcore.ExtensionFactory{plugin}
			}
			if tc.durable {
				cfg.Session, cfg.SessionID = agentcore.NewMemorySessionStore(), "guidance"
			}
			a, err := agentcore.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			ctx := agentcore.WithDelegationDepth(context.Background(), tc.depth)
			if tc.durable {
				// Durable execution belongs to the consuming session host. Inspect
				// its actual policy-filtered definitions through the native bridge.
				host, err := a.OpenPiTools(ctx)
				if err != nil {
					t.Fatal(err)
				}
				defer host.Close()
				raw, err := host.Definitions(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if err = json.Unmarshal(raw, &offered); err != nil {
					t.Fatal(err)
				}
			} else if _, err = a.Prompt(ctx, "hello"); err != nil {
				t.Fatal(err)
			}
			if !tc.durable && (!strings.Contains(prompt, "Host-authored mission.") || strings.Contains(prompt, "Delegate one bounded task")) {
				t.Fatal("plugin rewrote the mission or injected an unconditional system instruction", prompt)
			}
			visible := tc.installed && tc.granted && tc.depth == 0
			if !visible {
				if len(offered) != 0 {
					t.Fatalf("unavailable delegation advertised: %+v", offered)
				}
				return
			}
			if len(offered) != 1 || offered[0].Name != subagent.ToolSpawnSubagent {
				t.Fatalf("wrong tools: %+v", offered)
			}
			schema := offered[0]
			for _, contract := range []string{"handle ordinary reasoning", "Collect required results", "uncertain external effect must not be repeated", "Delegation never grants new authority"} {
				if !strings.Contains(schema.Description, contract) {
					t.Errorf("missing orchestration contract %q", contract)
				}
			}
			props := schema.Parameters["properties"].(map[string]any)
			_, named := props["agent"]
			if named != tc.named || strings.Contains(schema.Description, "do not probe unrelated") != tc.named {
				t.Fatal("named guidance differs from roster availability")
			}
			if tc.named {
				if !strings.Contains(props["agent"].(map[string]any)["description"].(string), "ledger — Inspect approved ledger entries") {
					t.Fatal("authorized roster missing")
				}
			}
			_, async := props["async"]
			wantAsync := tc.async && !tc.durable
			if async != wantAsync || strings.Contains(schema.Description, "job_wait/status") != wantAsync {
				t.Fatal("async guidance differs from advertised capability")
			}
			contextDescription := props["context"].(map[string]any)["description"].(string)
			if !strings.Contains(contextDescription, "their host permits") {
				t.Fatal("named context incorrectly promises unrestricted forwarding")
			}
		})
	}
}

func TestDelegationGuidanceUsesCurrentRosterAfterCheckpoint(t *testing.T) {
	var state json.RawMessage
	for _, target := range []string{"old-target", "new-target", ""} {
		var offered []agentcore.ToolSchema
		provider := nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
			raw, err := json.Marshal(ai.GetCurrentTools(view.Messages()))
			if err != nil {
				return nil, err
			}
			if err = json.Unmarshal(raw, &offered); err != nil {
				return nil, err
			}
			return ai.ScriptedStream(nativeAnswer("answer"))(ctx, model, view, options)
		})
		plugin := subagent.Plugin{}
		if target != "" {
			plugin.Delegates = []subagent.Delegate{{Name: target, Description: "Current authorized destination", Run: func(context.Context, string, agentcore.StreamSink) (string, agentcore.Usage, error) {
				return "unused", agentcore.Usage{}, nil
			}}}
		}
		a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "test", Policy: agentcore.NewAllowList(subagent.ToolSpawnSubagent), Extensions: []agentcore.ExtensionFactory{plugin}})
		if err != nil {
			t.Fatal(err)
		}
		r, err := a.RunNative(context.Background(), agentcore.NativeRun{State: state, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "continue"}}})
		if err != nil {
			t.Fatal(err)
		}
		state = r.NativeState
		if len(offered) != 1 || strings.Count(offered[0].Description, "Delegate one bounded task") != 1 {
			t.Fatal("missing or duplicated guidance on resume")
		}
		props := offered[0].Parameters["properties"].(map[string]any)
		if target == "" {
			if _, ok := props["agent"]; ok {
				t.Fatal("removed roster survived checkpoint")
			}
		} else {
			desc := props["agent"].(map[string]any)["description"].(string)
			if !strings.Contains(desc, target) || (target == "new-target" && strings.Contains(desc, "old-target")) {
				t.Fatal("stale roster restored", desc)
			}
		}
	}
}
