package agentcore_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry"
)

func TestNativeCompactionRecoversRejectedRequestWithoutRepeatingTools(t *testing.T) {
	for _, alwaysReject := range []bool{false, true} {
		t.Run(map[bool]string{false: "recovers", true: "bounded rejection"}[alwaysReject], func(t *testing.T) {
			calls, summaries, effects := 0, 0, 0
			p := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test","contextWindow":8192}`), Stream: func(ctx context.Context, _ json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
				if strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
					summaries++
					if options["toolChoice"] != "none" {
						t.Error("summary tools enabled")
					}
					return nativeReply(ctx, &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "Goal: inspect evidence. Done: read returned evidence. Next: answer."})})
				}
				calls++
				if calls == 1 {
					return nativeReply(ctx, &ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "read", Name: "read", Arguments: json.RawMessage(`{}`)})})
				}
				if calls == 2 || alwaysReject {
					return nil, &agentcore.ProviderError{Status: 400, Message: "context_length_exceeded"}
				}
				raw, _ := json.Marshal(view)
				if !strings.Contains(string(raw), "Earlier work summary") {
					t.Error("retry did not receive compacted view")
				}
				return nativeReply(ctx, &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "finished"})})
			}}}}
			retry := agentcore.RetryPolicy{MaxAttempts: 1}
			a, err := agentcore.New(agentcore.Config{NativeProvider: p, Model: "test", Retry: &retry, Tools: agentcore.NewToolSet(nativeReadTool{&effects}), Policy: agentcore.NewAllowList("read")})
			if err != nil {
				t.Fatal(err)
			}
			safe, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{})
			if err != nil {
				t.Fatal(err)
			}
			ctx, err := telemetry.WithSafeObserver(context.Background(), safe, telemetry.SafeExecution{ExecutionID: "compaction-run"})
			if err != nil {
				t.Fatal(err)
			}
			result, err := a.RunNative(ctx, agentcore.NativeRun{Task: "Inspect evidence and answer", Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "Inspect evidence and answer"}}})
			if (!alwaysReject && err != nil) || (alwaysReject && err == nil) || calls != 3 || summaries != 1 || effects != 1 {
				t.Fatalf("recovery err=%v primary=%d summaries=%d effects=%d", err, calls, summaries, effects)
			}
			if !alwaysReject && result.Final != "finished" {
				t.Fatal(result.Final)
			}
			main, compact := 0, 0
			for _, r := range safe.Drain(256) {
				if r.Kind == "attempt_settled" {
					switch r.Category {
					case telemetry.CategoryMain:
						main++
					case telemetry.CategoryCompaction:
						compact++
					}
					if r.SpanID == nil {
						t.Fatal("compaction correlation missing")
					}
				}
			}
			if main != 3 || compact != 1 {
				t.Fatalf("compaction telemetry main=%d summary=%d", main, compact)
			}
		})
	}
}

func TestNativeCompactionProgressDoesNotClaimWholeSuccess(t *testing.T) {
	for _, fails := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "summary failure"}[fails], func(t *testing.T) {
			calls, summaries, effects := 0, 0, 0
			p := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test","contextWindow":8192}`), Stream: func(ctx context.Context, _ json.RawMessage, view ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
				if strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
					summaries++
					if fails {
						return nil, &agentcore.ProviderError{Status: 400, Message: "SECRET_PROVIDER_ERROR"}
					}
					return nativeReply(ctx, &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "Goal: answer. Done: read evidence. Next: finish."})})
				}
				calls++
				if calls == 1 {
					return nativeReply(ctx, &ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "read", Name: "read", Arguments: json.RawMessage(`{}`)})})
				}
				if calls == 2 {
					return nil, &agentcore.ProviderError{Status: 400, Message: "context_length_exceeded"}
				}
				return nativeReply(ctx, &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "finished"})})
			}}}}
			retry := agentcore.RetryPolicy{MaxAttempts: 1}
			a, err := agentcore.New(agentcore.Config{NativeProvider: p, Model: "test", Retry: &retry, Tools: agentcore.NewToolSet(nativeReadTool{&effects}), Policy: agentcore.NewAllowList("read")})
			if err != nil {
				t.Fatal(err)
			}
			var notes []string
			_, _ = a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "Inspect evidence"}}, Sink: func(ev agentcore.StreamEvent) {
				if ev.Type == agentcore.StreamProgress {
					notes = append(notes, ev.Note)
				}
			}})
			starts, settlements := 0, 0
			for _, note := range notes {
				if note == "Compacting context" {
					starts++
				}
				if note == "Context summary chunk settled; compaction may continue" {
					settlements++
				}
				if strings.Contains(note, "compaction finished") || strings.Contains(note, "SECRET") {
					t.Fatalf("false success or private data: %q", note)
				}
			}
			if summaries != 1 || starts != summaries || settlements != summaries || effects != 1 {
				t.Fatalf("summaries=%d starts=%d settlements=%d effects=%d", summaries, starts, settlements, effects)
			}
		})
	}
}
