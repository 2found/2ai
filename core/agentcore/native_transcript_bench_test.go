package agentcore_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/core/ai"
)

// transcriptBenchAgent is a plugin-free native agent whose scripted model
// answers every request with one short text reply, and a summary request with
// a summary. Its budget and window are large enough that a long benchmark
// checkpoint never compacts by itself. extensions are composed as given.
func transcriptBenchAgent(tb testing.TB, extensions ...agentcore.ExtensionFactory) *agentcore.Agent {
	tb.Helper()
	reply := ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "ok"})}
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{
		Model: json.RawMessage(`{"id":"bench","api":"scripted","provider":"scripted","contextWindow":100000000}`),
		Stream: func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
			if strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
				return ai.ScriptedStream(*summaryReply("<summary>1. Goal and intent: a long benchmark thread.</summary>"))(ctx, model, view, options)
			}
			return ai.ScriptedStream(reply)(ctx, model, view, options)
		},
	}}}
	limits := agentcore.Limits{MaxTurns: 4, MaxToolCalls: 4, MaxContextTokens: 50_000_000}
	a, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "bench", Limits: &limits, Extensions: extensions})
	if err != nil {
		tb.Fatal(err)
	}
	return a
}

// transcriptBenchState is a resumable checkpoint holding n messages: the
// head a real first run writes (system, user, assistant), then alternating
// user and assistant text of about 600 bytes, the shape of a long thread.
func transcriptBenchState(tb testing.TB, a *agentcore.Agent, n int) json.RawMessage {
	tb.Helper()
	first, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "start"}}})
	if err != nil {
		tb.Fatal(err)
	}
	var checkpoint map[string]json.RawMessage
	if err := json.Unmarshal(first.NativeState, &checkpoint); err != nil {
		tb.Fatal(err)
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(checkpoint["messages"], &messages); err != nil {
		tb.Fatal(err)
	}
	for i := len(messages); i < n; i++ {
		text := fmt.Sprintf("tin %d ", i) + strings.Repeat("Nội dung trao đổi dài. ", 25)
		message := ai.Message{Role: "user", Content: ai.TextContent(text), Timestamp: int64(1_700_000_000_000 + i)}
		if i%2 == 0 {
			message.Role, message.API, message.Provider, message.Model, message.StopReason = "assistant", "scripted", "scripted", "bench", "stop"
			message.Content = ai.BlockContent(ai.ContentBlock{Type: "text", Text: text})
			message.Usage = &ai.Usage{Input: float64(100 * i), Output: 50, TotalTokens: float64(100*i + 50)}
		}
		raw, err := json.Marshal(message)
		if err != nil {
			tb.Fatal(err)
		}
		messages = append(messages, raw)
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		tb.Fatal(err)
	}
	checkpoint["messages"] = encoded
	state, err := json.Marshal(checkpoint)
	if err != nil {
		tb.Fatal(err)
	}
	return state
}

// transcriptBenchCompacted is transcriptBenchState after a /compact: the
// checkpoint still holds all n messages, and its summary covers them, so a
// request carries the summary and the new turn only. This is the shape of a
// long thread in production, where the context budget keeps the request
// bounded while the stored transcript grows.
func transcriptBenchCompacted(tb testing.TB, a *agentcore.Agent, n int) json.RawMessage {
	tb.Helper()
	state := transcriptBenchState(tb, a, n)
	compacted, err := a.RunNative(context.Background(), agentcore.NativeRun{State: state, ControlOnly: true, Compact: &agentcore.ManualCompaction{}})
	if err != nil || len(compacted.Compactions) != 1 || compacted.Compactions[0].Failed {
		tb.Fatalf("compact: %v %+v", err, compacted.Compactions)
	}
	return compacted.NativeState
}

func runTranscriptBenchTurn(tb testing.TB, a *agentcore.Agent, state json.RawMessage) agentcore.RunResult {
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{State: state, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "one more question"}}})
	if err != nil || result.StopReason != "stop" {
		tb.Fatalf("run: %v (%s)", err, result.StopReason)
	}
	return result
}

// BenchmarkNativeRunAppendOneTurn is one resumed run that appends a user
// message and a reply to an n-message checkpoint, through the real RunNative
// path with a scripted model: state in, provider request, state out.
func BenchmarkNativeRunAppendOneTurn(b *testing.B) {
	for _, n := range []int{50, 500, 4367} {
		b.Run(fmt.Sprintf("messages=%d", n), func(b *testing.B) {
			a := transcriptBenchAgent(b)
			state := transcriptBenchState(b, a, n)
			b.SetBytes(int64(len(state)))
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				runTranscriptBenchTurn(b, a, state)
			}
		})
	}
}

// BenchmarkNativeRunAppendOneTurnCompacted is BenchmarkNativeRunAppendOneTurn
// on a compacted checkpoint: the stored transcript is n messages long, the
// request is not.
func BenchmarkNativeRunAppendOneTurnCompacted(b *testing.B) {
	for _, n := range []int{50, 500, 4367} {
		b.Run(fmt.Sprintf("messages=%d", n), func(b *testing.B) {
			a := transcriptBenchAgent(b)
			state := transcriptBenchCompacted(b, a, n)
			b.SetBytes(int64(len(state)))
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				runTranscriptBenchTurn(b, a, state)
			}
		})
	}
}

// BenchmarkNativeRequestViewDecode is the typed decode a request makes of an
// uncompacted n-message view, the part of BenchmarkNativeRunAppendOneTurn
// that belongs to the provider request rather than to the checkpoint.
func BenchmarkNativeRequestViewDecode(b *testing.B) {
	for _, n := range []int{50, 500, 4367} {
		b.Run(fmt.Sprintf("messages=%d", n), func(b *testing.B) {
			var checkpoint struct{ Messages json.RawMessage }
			if err := json.Unmarshal(transcriptBenchState(b, transcriptBenchAgent(b), n), &checkpoint); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				var messages []ai.Message
				if err := json.Unmarshal(checkpoint.Messages, &messages); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// observingExtension observes messages and does nothing with them, the way
// plugins that watch the transcript (todo, goal, session query) make a run
// project it.
type observingExtension struct{}

func (observingExtension) Name() string { return "observer" }
func (observingExtension) BeginRun(context.Context, agentcore.RunInfo) (agentcore.Extension, error) {
	return observingExtension{}, nil
}
func (observingExtension) ObserveMessages(context.Context, agentcore.ObservePhase, int, []agentcore.Message) {
}

// BenchmarkNativeRunAppendOneTurnObserved is BenchmarkNativeRunAppendOneTurnCompacted
// with one message observer installed: what a run still pays to project the
// stored transcript for observers at PhaseRestore.
func BenchmarkNativeRunAppendOneTurnObserved(b *testing.B) {
	for _, n := range []int{50, 500, 4367} {
		b.Run(fmt.Sprintf("messages=%d", n), func(b *testing.B) {
			a := transcriptBenchAgent(b, observingExtension{})
			state := transcriptBenchCompacted(b, a, n)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				runTranscriptBenchTurn(b, a, state)
			}
		})
	}
}
