package agentcore_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/goal"
	"github.com/2found/2ai/agentcore/plugins/preset"
	"github.com/2found/2ai/agentcore/plugins/subagent"
	"github.com/2found/2ai/agentcore/plugins/todo"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry"
)

type nativeReadTool struct{ calls *int }

func (nativeReadTool) Name() string { return "read" }
func (nativeReadTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{Name: "read", Parameters: map[string]any{"type": "object"}}
}
func (t nativeReadTool) Run(context.Context, string) (string, error) {
	*t.calls++
	return "evidence", nil
}

type nativeProgressTool struct {
	name    string
	partial string
}

func (t nativeProgressTool) Name() string { return t.name }
func (t nativeProgressTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{Name: t.name, Parameters: map[string]any{"type": "object"}}
}
func (t nativeProgressTool) Run(context.Context, string) (string, error) { return "finished", nil }
func (t nativeProgressTool) RunStreaming(_ context.Context, _ string, emit func(string)) (string, error) {
	emit(t.partial)
	return "finished", nil
}

func nativeReply(ctx context.Context, message *ai.Message) (*ai.AssistantMessageEventStream, error) {
	stream := ai.NewAssistantMessageEventStreamFor(ctx)
	typ := "done"
	event := ai.AssistantMessageEvent{Type: typ, Reason: message.StopReason, Message: message}
	if message.StopReason == "error" {
		event = ai.AssistantMessageEvent{Type: "error", Reason: "error", Error: message}
	}
	stream.Push(event)
	stream.End()
	return stream, nil
}

func TestPublicNativeRunFallbackToolsAndCheckpoint(t *testing.T) {
	ctx := context.Background()
	primary, secondary, effects := 0, 0, 0
	signature := "opaque-signature"
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"primary"}`), Stream: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
			primary++
			return nil, &agentcore.ProviderError{Status: 503, Message: "busy"}
		}},
		{Model: json.RawMessage(`{"id":"secondary"}`), Stream: func(ctx context.Context, _ json.RawMessage, view ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
			secondary++
			message := &ai.Message{Role: "assistant", Model: "secondary", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "finished"}), Usage: &ai.Usage{Input: 2, Output: 1}}
			if secondary == 1 {
				message.StopReason = "toolUse"
				message.Content = ai.BlockContent(ai.ContentBlock{Type: "thinking", Thinking: "inspect", ThinkingSignature: &signature}, ai.ContentBlock{Type: "toolCall", ID: "read-1", Name: "read", Arguments: json.RawMessage(`{}`)})
				message.Extra = map[string]json.RawMessage{"opaque": json.RawMessage(`{"keep":9007199254740993}`)}
			}
			if secondary == 3 {
				raw, _ := json.Marshal(view)
				if !strings.Contains(string(raw), "opaque-signature") || !strings.Contains(string(raw), "9007199254740993") {
					t.Error("checkpoint lost opaque provider history")
				}
			}
			return nativeReply(ctx, message)
		}},
	}}
	makeAgent := func() *agentcore.Agent {
		t.Helper()
		retry := agentcore.RetryPolicy{MaxAttempts: 1}
		a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "primary", Retry: &retry, Tools: agentcore.NewToolSet(nativeReadTool{&effects}), Policy: agentcore.NewAllowList("read")})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	recorder := telemetry.NewInMemory()
	first, err := makeAgent().RunNative(ctx, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "read and answer"}}, Telemetry: recorder.Context})
	if err != nil {
		t.Fatal(err)
	}
	if first.Final != "finished" || first.Turns != 2 || len(first.Tools) != 1 || effects != 1 || primary != 1 || secondary != 2 || first.Usage.InputTokens != 4 {
		t.Fatalf("native cycle: %+v calls=%d/%d effects=%d", first, primary, secondary, effects)
	}
	spans := recorder.GetSpans()
	if len(spans) != 4 {
		t.Fatalf("native spans: %+v", spans)
	}
	for _, span := range spans {
		if !span.Settled || (span.Name != "agentray.agent.run" && (span.ParentID == nil || *span.ParentID != spans[0].ID)) {
			t.Fatalf("unsettled/detached native span: %+v", span)
		}
	}
	second, err := makeAgent().RunNative(ctx, agentcore.NativeRun{State: first.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "continue"}}})
	if err != nil {
		t.Fatal(err)
	}
	if primary != 1 || secondary != 3 || effects != 1 || second.Final != "finished" {
		t.Fatal("checkpoint replayed work or forgot selection", primary, secondary, effects)
	}
}

func TestPublicNativeRunSurfacesOnlySubagentProgressPartials(t *testing.T) {
	calls := 0
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		calls++
		message := &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "finished"})}
		switch calls {
		case 1:
			message.StopReason = "toolUse"
			message.Content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "child", Name: "spawn_subagent", Arguments: json.RawMessage(`{}`)})
		case 2:
			message.StopReason = "toolUse"
			message.Content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "other", Name: "other_stream", Arguments: json.RawMessage(`{}`)})
		}
		return nativeReply(ctx, message)
	}}}}
	tools := agentcore.NewToolSet(
		nativeProgressTool{name: "spawn_subagent", partial: "[sub-agent] running web_search"},
		nativeProgressTool{name: "other_stream", partial: "private tool result"},
	)
	a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "test", Tools: tools, Policy: agentcore.NewAllowList("spawn_subagent", "other_stream")})
	if err != nil {
		t.Fatal(err)
	}
	var notes []string
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "delegate"}}, Sink: func(ev agentcore.StreamEvent) {
		if ev.Type == agentcore.StreamToolExecUpdate {
			notes = append(notes, ev.Note)
		}
	}})
	if err != nil || result.Final != "finished" {
		t.Fatalf("run: result=%+v err=%v", result, err)
	}
	if len(notes) != 1 || notes[0] != "[sub-agent] running web_search" {
		t.Fatalf("public progress notes = %#v, want only the safe subagent lifecycle note", notes)
	}
}

func TestPublicNativeRunDiscardsMalformedSavedSummary(t *testing.T) {
	requests := 0
	p := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: func(ctx context.Context, _ json.RawMessage, view ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
		requests++
		if requests == 2 {
			raw, _ := json.Marshal(view)
			if !strings.Contains(string(raw), "ORIGINAL-EVIDENCE") {
				t.Error("invalid summary replaced original history")
			}
		}
		return nativeReply(ctx, &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})})
	}}}}
	makeAgent := func() *agentcore.Agent {
		a, err := agentcore.New(agentcore.Config{NativeProvider: p, Model: "test"})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	first, err := makeAgent().Prompt(context.Background(), "ORIGINAL-EVIDENCE")
	if err != nil {
		t.Fatal(err)
	}
	var checkpoint map[string]json.RawMessage
	if err := json.Unmarshal(first.NativeState, &checkpoint); err != nil {
		t.Fatal(err)
	}
	checkpoint["summary"], _ = json.Marshal(map[string]any{"revision": first.NativeRevision, "prefix_count": 1, "prefix_digest": strings.Repeat("0", 64), "message": map[string]any{"role": "user", "content": "[Earlier work summary]\n<|open|>tools<|sep|>session_query", "agentrayContextSummary": "bad", "timestamp": 1}})
	state, _ := json.Marshal(checkpoint)
	discarded := false
	second, err := makeAgent().RunNative(context.Background(), agentcore.NativeRun{State: state, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "continue"}}, Sink: func(e agentcore.StreamEvent) {
		if e.Type == agentcore.StreamProgress && strings.Contains(e.Note, "Discarding") {
			discarded = true
		}
	}})
	if err != nil || second.Final != "done" || !discarded || requests != 2 {
		t.Fatal("bad summary did not recover", err, discarded, requests)
	}
}

func TestPublicNativeRunRejectsDisplayHistoryAndInvalidCheckpoint(t *testing.T) {
	calls := 0
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: func(ctx context.Context, _ json.RawMessage, _ ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
		calls++
		return nativeReply(ctx, &ai.Message{Role: "assistant", StopReason: "stop"})
	}}}}
	a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "test"})
	if err != nil {
		t.Fatal(err)
	}
	for _, run := range []agentcore.NativeRun{{State: json.RawMessage(`[]`)}, {Input: []agentcore.Message{{Role: agentcore.RoleAssistant, Content: "lossy projection"}}}} {
		if _, err := a.RunNative(context.Background(), run); err == nil {
			t.Fatal("accepted incompatible state")
		}
	}
	if calls != 0 {
		t.Fatal("invalid checkpoint reached provider")
	}
}

func TestPublicNativeSubagentCorrectionUsesCheckpoint(t *testing.T) {
	text := func(value string) ai.Message {
		return ai.Message{Role: "assistant", Model: "test", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: value}), Usage: &ai.Usage{Input: 1}}
	}
	script := ai.ScriptedStream(
		ai.Message{Role: "assistant", Model: "test", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "delegate", Name: subagent.ToolSpawnSubagent, Arguments: json.RawMessage(`{"task":"Return the answer","output_schema":{"type":"object","required":["answer"],"properties":{"answer":{"type":"string"}}}}`)})},
		text("invalid answer"), text(`{"answer":"corrected"}`), text("finished"))
	a, err := agentcore.New(agentcore.Config{NativeProvider: &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: script}}}, Model: "test", Policy: agentcore.NewAllowList(subagent.ToolSpawnSubagent), Extensions: []agentcore.ExtensionFactory{subagent.SelfOnly()}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := a.Prompt(context.Background(), "delegate")
	if err != nil {
		t.Fatal(err)
	}
	corrected := false
	for _, message := range result.Messages {
		if message.Role == agentcore.RoleTool && strings.Contains(message.Content, `"answer":"corrected"`) {
			corrected = true
		}
	}
	if len(result.Tools) != 1 || result.Tools[0].Error != "" || !corrected || result.Usage.InputTokens != 3 {
		t.Fatalf("native child correction failed: final=%q tools=%+v usage=%+v", result.Final, result.Tools, result.Usage)
	}
}

func TestPublicNativeGoalRevisionAndPlanBudget(t *testing.T) {
	ctx := context.Background()
	var calls int
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		calls++
		reply := ai.Message{Role: "assistant", StopReason: "toolUse"}
		switch calls {
		case 1:
			reply.Content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "plan", Name: "update_plan", Arguments: json.RawMessage(`{"items":[{"content":"check evidence","status":"in_progress"}]}`)})
		case 2:
			reply.Content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "goal", Name: "update_goal", Arguments: json.RawMessage(`{"goal":"verify actual evidence","reason":"initial objective was underspecified"}`)})
		default:
			if !strings.Contains(ai.GetCurrentSystemPrompt(view.Messages()), "verify actual evidence") {
				t.Error("revised goal missing")
			}
			reply.StopReason = "stop"
			reply.Content = ai.BlockContent(ai.ContentBlock{Type: "text", Text: "verified\nSTATUS: DONE"})
		}
		return ai.ScriptedStream(reply)(ctx, model, view, options)
	}}}}
	limits := agentcore.Limits{MaxTurns: 2, MaxToolCalls: 8, MaxContextTokens: 100000}
	makeAgent := func() *agentcore.Agent {
		cfg := agentcore.Config{NativeProvider: provider, Model: "test", Goal: "original", Limits: &limits, Policy: agentcore.NewAllowList("update_goal", "update_plan")}
		plugins := preset.Plugins(cfg)
		plugins = preset.Replace(plugins, goal.Plugin{Goal: "original", Revisable: true})
		plugins = append(plugins, todo.With(todo.NewStore()))
		a, err := agentcore.Build(plugins...)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	first, err := makeAgent().RunNative(ctx, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "verify"}}})
	if err != nil || first.StopReason != "stop" || calls != 3 {
		t.Fatalf("administrative turns consumed work budget: calls=%d reason=%s error=%v", calls, first.StopReason, err)
	}
	var checkpoint struct {
		Goal          string
		GoalRevisions []agentcore.GoalRevision
	}
	if err := json.Unmarshal(first.NativeState, &checkpoint); err != nil {
		t.Fatal(err)
	}
	if checkpoint.Goal != "verify actual evidence" || len(checkpoint.GoalRevisions) != 1 {
		t.Fatalf("revision not checkpointed: %+v", checkpoint)
	}
	second, err := makeAgent().RunNative(ctx, agentcore.NativeRun{State: first.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "continue"}}})
	if err != nil || second.StopReason != "stop" {
		t.Fatalf("resume: %s %v", second.StopReason, err)
	}
}
