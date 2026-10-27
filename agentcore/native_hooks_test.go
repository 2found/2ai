package agentcore_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

func TestNativeRunCallsStartAndRequestHooks(t *testing.T) {
	var sent string
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"m"}`), Stream: func(ctx context.Context, _ json.RawMessage, view ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
		raw, _ := json.Marshal(view)
		sent = string(raw)
		return nativeReply(ctx, &ai.Message{Role: "assistant", Model: "m", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"}), Usage: &ai.Usage{Input: 2, Output: 1}})
	}}}}
	var starts []agentcore.RunStart
	var requests []agentcore.ChatRequest
	var failures []string
	effects := 0
	hooks := agentcore.Hooks{
		BeforeAgentStart: []agentcore.BeforeAgentStartHook{func(_ context.Context, s agentcore.RunStart) agentcore.RunStart {
			starts = append(starts, s)
			return agentcore.RunStart{System: s.System + "\n\nSHAPED", Messages: append(s.Messages, agentcore.Message{Role: agentcore.RoleUser, Content: "injected"})}
		}},
		BeforeProviderRequest: []agentcore.ProviderRequestHook{
			func(_ context.Context, r agentcore.ChatRequest) agentcore.ChatRequest {
				requests = append(requests, r)
				return r
			},
			func(_ context.Context, r agentcore.ChatRequest) agentcore.ChatRequest {
				r.Messages = r.Messages[:1]
				return r
			},
		},
		OnError: func(source string, _ error) { failures = append(failures, source) },
	}
	a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "m", Tools: agentcore.NewToolSet(nativeReadTool{&effects}), Policy: agentcore.NewAllowList("read"), Hooks: hooks})
	if err != nil {
		t.Fatal(err)
	}
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "hello"}}, Task: "greet"})
	if err != nil || result.Final != "done" {
		t.Fatalf("run %+v %v", result, err)
	}
	if len(starts) != 1 || starts[0].Task != "greet" || starts[0].System == "" || len(starts[0].Messages) != 1 {
		t.Fatalf("start hook saw %+v", starts)
	}
	if !strings.Contains(sent, "SHAPED") || !strings.Contains(sent, "injected") {
		t.Fatalf("start edits not sent: %s", sent)
	}
	if len(requests) != 1 {
		t.Fatalf("request hook calls %d", len(requests))
	}
	r := requests[0]
	if r.Model != "m" || len(r.Messages) < 3 || r.Messages[0].Role != agentcore.RoleSystem || !strings.HasSuffix(r.Messages[0].Content, "SHAPED") || len(r.Tools) != 1 || r.Tools[0].Name != "read" {
		t.Fatalf("request projection %+v", r)
	}
	if len(failures) != 1 || failures[0] != "before_provider_request" || !strings.Contains(sent, "hello") {
		t.Fatalf("a rewrite must be reported, not applied: %v", failures)
	}
}
