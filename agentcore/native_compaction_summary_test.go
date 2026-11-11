package agentcore_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

// compactionRun drives a tool loop that outgrows a small context budget.
// classify sees every provider request and returns a reply for a summary
// request, or nil for a main request, which then calls the tool until turns
// main requests have been made.
func compactionRun(t *testing.T, turns int, settings agentcore.CompactionSettings, hooks agentcore.Hooks, classify func(view ai.TranscriptContext, options map[string]any) *ai.Message) (agentcore.RunResult, int, int, []string) {
	t.Helper()
	calls, effects := 0, 0
	signature := "OPAQUE-SIGNATURE"
	var mainViews []string
	stream := func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		if reply := classify(view, options); reply != nil {
			return ai.ScriptedStream(*reply)(ctx, model, view, options)
		}
		calls++
		raw, _ := json.Marshal(view.Messages())
		mainViews = append(mainViews, string(raw))
		message := ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "finished"}), Usage: &ai.Usage{Input: 10, Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}}
		if calls < turns {
			message.StopReason = "toolUse"
			message.Content = ai.BlockContent(
				ai.ContentBlock{Type: "thinking", Thinking: "private reasoning", ThinkingSignature: &signature},
				ai.ContentBlock{Type: "toolCall", ID: fmt.Sprintf("work-%d", calls), Name: "work", Arguments: json.RawMessage(`{}`)})
		}
		return ai.ScriptedStream(message)(ctx, model, view, options)
	}
	limits := agentcore.DefaultLimits()
	limits.MaxTurns, limits.MaxToolCalls, limits.MaxContextTokens = turns+10, turns+10, 4000
	settings.KeepRecentTokens = 1500
	a, err := agentcore.New(agentcore.Config{Model: "fixture", NativeProvider: &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"fixture","contextWindow":100000}`), Stream: stream}}}, Limits: &limits, Compaction: &settings, Hooks: hooks,
		Tools: agentcore.NewToolSet(agentcore.StringTool{ToolName: "work", Properties: agentcore.StringProperties(), Required: []string{}, Execute: func(context.Context, map[string]string) (string, error) {
			effects++
			return fmt.Sprintf("evidence-%d %s", effects, strings.Repeat("x", 900)), nil
		}}), Policy: agentcore.NewAllowList("work")})
	if err != nil {
		t.Fatal(err)
	}
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Task: "Complete the task.", Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "Complete the task."}}})
	if err != nil || result.Final != "finished" {
		t.Fatalf("run final=%q err=%v", result.Final, err)
	}
	return result, calls, effects, mainViews
}

func lastText(view ai.TranscriptContext) string {
	messages := view.Messages()
	return ai.ContentText(messages[len(messages)-1].Content)
}

func summaryReply(text string) *ai.Message {
	return &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: text}), Usage: &ai.Usage{Input: 50, Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}}
}

func TestNativeSummaryRequestIsRenderedTextWithAStructuredHandoff(t *testing.T) {
	summaries := 0
	settings := agentcore.DefaultCompactionSettings()
	settings.Instructions = "Keep every evidence number."
	settings.StandaloneSummary = true
	result, calls, effects, views := compactionRun(t, 21, settings, agentcore.Hooks{}, func(view ai.TranscriptContext, options map[string]any) *ai.Message {
		if !strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
			return nil
		}
		summaries++
		content := lastText(view)
		for _, want := range []string{"<conversation>", "USER: ", "Complete the task.", "ASSISTANT called work(", "TOOL work -> evidence-", "<summary>", "9. Next step", "Keep every evidence number."} {
			if !strings.Contains(content, want) {
				t.Errorf("summary request lost %q", want)
			}
		}
		for _, unwanted := range []string{"OPAQUE-SIGNATURE", "private reasoning", `"toolCallId"`, `"timestamp"`} {
			if strings.Contains(content, unwanted) {
				t.Errorf("summary request carried %q", unwanted)
			}
		}
		// A later compaction extends the earlier handoff instead of rereading
		// the work that handoff already stands for.
		if (summaries > 1) != strings.Contains(content, "PREVIOUS SUMMARY (stands") || summaries > 1 && strings.Contains(content, "evidence-1 ") {
			t.Errorf("summary %d did not build on the previous handoff", summaries)
		}
		if options["toolChoice"] != "none" || options["cacheRetention"] != "none" {
			t.Errorf("summary controls %v", options)
		}
		return summaryReply("<analysis>SCRATCHPAD</analysis>\n<summary>\n1. Goal and intent: complete the task.\n8. Current state: evidence gathered.\n</summary>")
	})
	if calls != 21 || effects != 20 || summaries == 0 || len(result.Compactions) != summaries {
		t.Fatalf("calls=%d effects=%d summaries=%d events=%d", calls, effects, summaries, len(result.Compactions))
	}
	for _, event := range result.Compactions {
		if event.Failed || event.Mode != "standalone" || event.Trigger != "auto" || event.PostTokens >= event.PreTokens || event.PrefixMessages == 0 || event.Usage.InputTokens != 50 {
			t.Fatalf("compaction event %+v", event)
		}
	}
	last := views[len(views)-1]
	if !strings.Contains(last, "[Earlier work summary]\\n1. Goal and intent: complete the task.") || strings.Contains(last, "SCRATCHPAD") {
		t.Fatal("the request after compaction must carry the handoff without its scratchpad")
	}
}

func TestNativeSharedCacheSummaryContinuesTheLiveConversation(t *testing.T) {
	for _, tc := range []struct {
		name, reply string
		toolCall    bool
		mode        string
	}{
		{name: "summary block", reply: "<analysis>notes</analysis><summary>1. Goal and intent: shared handoff.</summary>", mode: "shared"},
		{name: "model carried on in prose", reply: "Next I will read more evidence.", mode: "standalone"},
		{name: "model called a tool", toolCall: true, mode: "standalone"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			shared, standalone := 0, 0
			var mainSystem string
			settings := agentcore.DefaultCompactionSettings()
			result, calls, effects, views := compactionRun(t, 21, settings, agentcore.Hooks{}, func(view ai.TranscriptContext, options map[string]any) *ai.Message {
				system := ai.GetCurrentSystemPrompt(view.Messages())
				if strings.HasPrefix(system, "Summarize the conversation") {
					standalone++
					return summaryReply("<summary>1. Goal and intent: standalone handoff.</summary>")
				}
				if !strings.Contains(lastText(view), "about to be compacted") {
					mainSystem = system
					return nil
				}
				shared++
				raw, _ := json.Marshal(view.Messages())
				// The shared request is the live conversation plus one message:
				// same system prompt, native tool results, the main controls.
				if system != mainSystem || !strings.Contains(string(raw), `"toolResult"`) || !strings.Contains(string(raw), "OPAQUE-SIGNATURE") || options["toolChoice"] != nil || options["reasoning"] != nil {
					t.Errorf("shared request is not the live conversation: system=%v options=%v", system == mainSystem, options)
				}
				if tc.toolCall {
					return &ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "stray", Name: "work", Arguments: json.RawMessage(`{}`)})}
				}
				return summaryReply(tc.reply)
			})
			if calls != 21 || effects != 20 || shared == 0 {
				t.Fatalf("calls=%d effects=%d shared=%d: a summary reply must never run a tool", calls, effects, shared)
			}
			if (tc.mode == "shared") != (standalone == 0) {
				t.Fatalf("mode %s made %d standalone calls", tc.mode, standalone)
			}
			for _, event := range result.Compactions {
				if event.Failed || event.Mode != tc.mode {
					t.Fatalf("compaction event %+v, want mode %s", event, tc.mode)
				}
			}
			if last := views[len(views)-1]; !strings.Contains(last, tc.mode+" handoff.") || strings.Contains(last, "about to be compacted") {
				t.Fatal("the request after compaction must carry the handoff and not the summary instruction")
			}
		})
	}
}

func TestNativeBeforeCompactHookCanDeferASummary(t *testing.T) {
	asked, summaries, replaced := 0, 0, 0
	hooks := agentcore.Hooks{
		OnError: func(source string, _ error) {
			if source == "before_compact" {
				replaced++
			}
		},
		BeforeCompact: []agentcore.BeforeCompactHook{func(_ context.Context, request agentcore.CompactRequest) agentcore.CompactDecision {
			asked++
			if request.Budget != 4000 || len(request.Messages) < 3 || summaries > 0 && asked <= 3 {
				t.Errorf("hook request budget=%d messages=%d asked=%d summaries=%d", request.Budget, len(request.Messages), asked, summaries)
			}
			if asked <= 3 {
				return agentcore.CompactDecision{Skip: true}
			}
			// A native run keeps its transcript: a replacement is reported, not applied.
			return agentcore.CompactDecision{Messages: []agentcore.Message{{Role: agentcore.RoleUser, Content: "replacement"}}}
		}},
	}
	result, calls, effects, views := compactionRun(t, 21, agentcore.CompactionSettings{StandaloneSummary: true}, hooks, func(view ai.TranscriptContext, _ map[string]any) *ai.Message {
		if !strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
			return nil
		}
		summaries++
		return summaryReply("<summary>1. Goal and intent: deferred handoff.</summary>")
	})
	if calls != 21 || effects != 20 || asked <= 3 || summaries == 0 || summaries != asked-3 || replaced != summaries || len(result.Compactions) != summaries {
		t.Fatalf("asked=%d summaries=%d replaced=%d events=%d", asked, summaries, replaced, len(result.Compactions))
	}
	if last := views[len(views)-1]; strings.Contains(last, "replacement") || !strings.Contains(last, "deferred handoff.") {
		t.Fatal("the hook's replacement transcript must not reach the provider")
	}
}
