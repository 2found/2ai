package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/goal"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry/export"
	"github.com/2found/2ai/telemetry/llm"
)

// A long native run exercises compaction, the goal gate, permission denials,
// attempt telemetry and opaque checkpoint resume together. The retired typed
// journal observer is not part of execution or the durability proof.
func TestNativeLongRunCompactionTelemetryAndResume(t *testing.T) {
	ctx := context.Background()
	const target = 90
	calls, summaries, effects, forbidden := 0, 0, 0, 0
	var records []llm.TraceRecord
	stream := func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		text := ""
		message := ai.Message{Role: "assistant", Model: "longrun", Provider: "fixture", StopReason: "stop", Usage: &ai.Usage{Input: 10, Output: 2, Cost: ai.UsageCost{Total: 0.001}}}
		if strings.Contains(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
			summaries++
			text = "Continue the long task, preserve the goal and verified work."
		} else {
			calls++
			if !strings.Contains(ai.GetCurrentSystemPrompt(view.Messages()), "Complete the long task") {
				t.Error("compaction lost goal instructions")
			}
			switch {
			case calls <= target:
				name := "work"
				if calls == 3 {
					name = "wire_money"
				}
				message.StopReason = "toolUse"
				message.Content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: fmt.Sprintf("call-%d", calls), Name: name, Arguments: json.RawMessage(`{}`)})
			case calls == target+1:
				text = "Work is finished."
			default:
				text = "Work is verified.\nSTATUS: DONE"
			}
		}
		if text != "" {
			message.Content = ai.BlockContent(ai.ContentBlock{Type: "text", Text: text})
		}
		return ai.ScriptedStream(message)(ctx, model, view, options)
	}
	tools := agentcore.NewToolSet(
		agentcore.StringTool{ToolName: "work", Description: "perform bounded work", Properties: agentcore.StringProperties(), Required: []string{}, Execute: func(context.Context, map[string]string) (string, error) {
			effects++
			return fmt.Sprintf("verified work %d %s", effects, strings.Repeat("x", 900)), nil
		}},
		agentcore.StringTool{ToolName: "wire_money", Description: "forbidden effect", Properties: agentcore.StringProperties(), Required: []string{}, Execute: func(context.Context, map[string]string) (string, error) { forbidden++; return "sent", nil }},
	)
	limits := agentcore.DefaultLimits()
	limits.MaxTurns, limits.MaxToolCalls, limits.MaxContextTokens = 120, 120, 4000
	compact := agentcore.DefaultCompactionSettings()
	compact.KeepRecentTokens = 1500
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"longrun","provider":"fixture"}`), Stream: stream}}}
	build := func(condition string) *agentcore.Agent {
		a, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{NativeProvider: provider, Model: "longrun", Tools: tools, Policy: agentcore.NewAllowList("work"), Limits: &limits, Compaction: &compact}), goal.Until(condition))
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	backend := export.New(func(batch export.Batch) {
		llm.RecordBatch(batch, llm.SinkFunc(func(r llm.TraceRecord) { records = append(records, r) }), llm.Metadata{PricingKnown: true})
	})
	result, err := build("Complete the long task").RunNative(ctx, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "Work the long task to completion."}}, Telemetry: backend})
	if err != nil {
		t.Fatal(err)
	}
	if calls != target+2 || effects != target-1 || forbidden != 0 || summaries == 0 || !strings.Contains(result.Final, "STATUS: DONE") {
		t.Fatalf("long run: calls=%d effects=%d forbidden=%d summaries=%d final=%q", calls, effects, forbidden, summaries, result.Final)
	}
	if len(records) != calls+summaries || result.Usage.InputTokens != (calls+summaries)*10 || result.Usage.CostUSD <= 0 {
		t.Fatalf("attempt telemetry/accounting: records=%d calls=%d summaries=%d usage=%+v", len(records), calls, summaries, result.Usage)
	}
	var checkpoint struct {
		Goal     string
		Messages json.RawMessage
		Summary  json.RawMessage
	}
	if err := json.Unmarshal(result.NativeState, &checkpoint); err != nil || checkpoint.Goal != "Complete the long task" || len(checkpoint.Summary) == 0 {
		t.Fatalf("checkpoint lost goal/summary: %+v %v", checkpoint, err)
	}
	before := effects
	resumed, err := build("").RunNative(ctx, agentcore.NativeRun{State: result.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "Confirm the completed work."}}})
	if err != nil {
		t.Fatal(err)
	}
	if effects != before || !strings.Contains(resumed.Final, "STATUS: DONE") {
		t.Fatalf("resume replayed effects/lost gate: effects=%d final=%q", effects, resumed.Final)
	}
}
