package agentcore_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/core/ai"
)

// workLoop is a provider that answers a summary request with summary and a
// main request with a call to "work", until calls main requests have been
// made; then it answers "finished". Each main view is recorded.
type workLoop struct {
	calls, turns int
	summaries    int
	summary      func(view ai.TranscriptContext) *ai.Message
	views        []string
	options      []map[string]any
}

func (w *workLoop) stream(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
	if strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
		w.summaries++
		reply := summaryReply("<summary>1. Goal and intent: complete the task.\n8. Current state: evidence gathered.</summary>")
		if w.summary != nil {
			reply = w.summary(view)
		}
		return ai.ScriptedStream(*reply)(ctx, model, view, options)
	}
	w.calls++
	raw, _ := json.Marshal(view.Messages())
	w.views, w.options = append(w.views, string(raw)), append(w.options, options)
	message := ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "finished"}), Usage: &ai.Usage{Input: 10}}
	if w.calls < w.turns {
		message.StopReason = "toolUse"
		message.Content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: fmt.Sprintf("work-%d", w.calls), Name: "work", Arguments: json.RawMessage(`{}`)})
	}
	return ai.ScriptedStream(message)(ctx, model, view, options)
}

func (w *workLoop) provider() *ai.FallbackProvider {
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"fixture","contextWindow":100000}`), Stream: w.stream}}}
}

// workAgent runs w with a "work" tool whose every result is about 230 tokens.
func workAgent(t *testing.T, w *workLoop, budget int, settings agentcore.CompactionSettings, configure func(*agentcore.Config)) *agentcore.Agent {
	t.Helper()
	effects := 0
	limits := agentcore.DefaultLimits()
	limits.MaxTurns, limits.MaxToolCalls, limits.MaxContextTokens = 50, 50, budget
	cfg := agentcore.Config{Model: "fixture", NativeProvider: w.provider(), Limits: &limits, Compaction: &settings, Policy: agentcore.NewAllowList("work"),
		Tools: agentcore.NewToolSet(agentcore.StringTool{ToolName: "work", Properties: agentcore.StringProperties(), Required: []string{}, Execute: func(context.Context, map[string]string) (string, error) {
			effects++
			return fmt.Sprintf("evidence-%d %s", effects, strings.Repeat("x", 900)), nil
		}})}
	if configure != nil {
		configure(&cfg)
	}
	a, err := agentcore.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func task(text string) []agentcore.Message {
	return []agentcore.Message{{Role: agentcore.RoleUser, Content: text}}
}

func stateMessages(t *testing.T, state json.RawMessage) []json.RawMessage {
	t.Helper()
	var checkpoint struct{ Messages []json.RawMessage }
	if err := json.Unmarshal(state, &checkpoint); err != nil {
		t.Fatal(err)
	}
	return checkpoint.Messages
}

// The checkpoint callback receives a resumable state after every settled turn,
// each one longer than the last.
func TestNativeCheckpointCallbackReceivesEachSettledTurn(t *testing.T) {
	w := &workLoop{turns: 3}
	var states []json.RawMessage
	result, err := workAgent(t, w, 100000, agentcore.DefaultCompactionSettings(), nil).RunNative(context.Background(), agentcore.NativeRun{Input: task("work"), Checkpoint: func(ctx context.Context, state json.RawMessage) error {
		if ctx.Err() != nil {
			t.Error("checkpoint context already done")
		}
		states = append(states, state)
		return nil
	}})
	if err != nil || result.Final != "finished" {
		t.Fatalf("run final=%q err=%v", result.Final, err)
	}
	if len(states) != 3 {
		t.Fatalf("%d checkpoints for 3 turns", len(states))
	}
	for i := 1; i < len(states); i++ {
		if len(stateMessages(t, states[i])) <= len(stateMessages(t, states[i-1])) {
			t.Fatalf("checkpoint %d did not grow", i)
		}
	}
	if got, want := len(stateMessages(t, states[2])), len(stateMessages(t, result.NativeState)); got != want {
		t.Fatalf("last checkpoint has %d messages, the result %d", got, want)
	}
	// An intermediate state resumes: the first turn's tool is not run again.
	resume := &workLoop{turns: 1}
	next, err := workAgent(t, resume, 100000, agentcore.DefaultCompactionSettings(), nil).RunNative(context.Background(), agentcore.NativeRun{State: states[0], Input: task("continue")})
	if err != nil || next.Final != "finished" || !strings.Contains(resume.views[0], "evidence-1") {
		t.Fatalf("resume from the first checkpoint: final=%q err=%v", next.Final, err)
	}
}

// A failing callback stops the run before another turn is paid for, and the
// run still returns the latest resumable state.
func TestNativeCheckpointCallbackErrorStopsTheRun(t *testing.T) {
	w := &workLoop{turns: 5}
	calls := 0
	result, err := workAgent(t, w, 100000, agentcore.DefaultCompactionSettings(), nil).RunNative(context.Background(), agentcore.NativeRun{Input: task("work"), Checkpoint: func(context.Context, json.RawMessage) error {
		if calls++; calls == 2 {
			return errors.New("disk full")
		}
		return nil
	}})
	if err == nil || !strings.Contains(err.Error(), "checkpoint: disk full") {
		t.Fatalf("err = %v, want the checkpoint failure", err)
	}
	if w.calls != 2 {
		t.Fatalf("%d requests: the run went on after its checkpoint failed", w.calls)
	}
	if result.NativeState == nil || result.NativeRevision == "" {
		t.Fatal("a failed run returned no resumable state")
	}
	resume := &workLoop{turns: 1}
	if _, err := workAgent(t, resume, 100000, agentcore.DefaultCompactionSettings(), nil).RunNative(context.Background(), agentcore.NativeRun{State: result.NativeState, Input: task("continue")}); err != nil || !strings.Contains(resume.views[0], "evidence-2") {
		t.Fatalf("the returned state does not resume after the second turn: %v", err)
	}
}

// A run cancelled inside a tool batch returns the last settled turn's state,
// not nothing.
func TestNativeCancelledRunReturnsTheLastResumableState(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := &workLoop{turns: 5}
	effects := 0
	a := workAgent(t, w, 100000, agentcore.DefaultCompactionSettings(), func(cfg *agentcore.Config) {
		cfg.Tools = agentcore.NewToolSet(agentcore.StringTool{ToolName: "work", Properties: agentcore.StringProperties(), Required: []string{}, Execute: func(ctx context.Context, _ map[string]string) (string, error) {
			if effects++; effects == 2 {
				cancel()
				<-ctx.Done()
				return "", ctx.Err()
			}
			return fmt.Sprintf("evidence-%d", effects), nil
		}})
	})
	result, _ := a.RunNative(ctx, agentcore.NativeRun{Input: task("work")})
	if result.NativeState == nil {
		t.Fatal("a cancelled run returned no resumable state")
	}
	resume := &workLoop{turns: 1}
	if _, err := workAgent(t, resume, 100000, agentcore.DefaultCompactionSettings(), nil).RunNative(context.Background(), agentcore.NativeRun{State: result.NativeState, Input: task("continue")}); err != nil || !strings.Contains(resume.views[0], "evidence-1") {
		t.Fatalf("the cancelled run's state does not resume: %v", err)
	}
}

// /compact summarizes now with the standing and the request's focus, and the
// next request reads the summary instead of the history.
func TestNativeManualCompactUsesItsFocus(t *testing.T) {
	settings := agentcore.DefaultCompactionSettings()
	settings.Instructions = "Keep every evidence number."
	settings.ClearToolResults.Disabled = true
	first := &workLoop{turns: 6}
	done, err := workAgent(t, first, 100000, settings, nil).RunNative(context.Background(), agentcore.NativeRun{Input: task("work")})
	if err != nil || first.summaries != 0 {
		t.Fatalf("setup run: %v summaries=%d", err, first.summaries)
	}

	var prompt string
	compact := &workLoop{summary: func(view ai.TranscriptContext) *ai.Message {
		prompt = lastText(view)
		return summaryReply("<summary>1. Goal and intent: manual handoff.</summary>")
	}}
	compacted, err := workAgent(t, compact, 100000, settings, nil).RunNative(context.Background(), agentcore.NativeRun{State: done.NativeState, ControlOnly: true, Compact: &agentcore.ManualCompaction{Instructions: "Focus on the open question."}})
	if err != nil {
		t.Fatal(err)
	}
	if compact.summaries != 1 || compact.calls != 0 {
		t.Fatalf("control-only compact made %d summaries and %d main requests", compact.summaries, compact.calls)
	}
	if !strings.Contains(prompt, "Keep every evidence number.") || !strings.Contains(prompt, "Focus on the open question.") {
		t.Fatal("the summary prompt lost the standing or the manual focus")
	}
	if len(compacted.Compactions) != 1 || compacted.Compactions[0].Trigger != "manual" || compacted.Compactions[0].Failed {
		t.Fatalf("compaction events %+v", compacted.Compactions)
	}

	next := &workLoop{turns: 1}
	if _, err := workAgent(t, next, 100000, settings, nil).RunNative(context.Background(), agentcore.NativeRun{State: compacted.NativeState, Input: task("continue")}); err != nil {
		t.Fatal(err)
	}
	if view := next.views[0]; !strings.Contains(view, "manual handoff.") || strings.Contains(view, "evidence-1 ") {
		t.Fatal("the request after /compact still carries the summarized history")
	}
}

func TestNativeManualCompactFailsWhenNothingCompacts(t *testing.T) {
	w := &workLoop{}
	_, err := workAgent(t, w, 100000, agentcore.DefaultCompactionSettings(), nil).RunNative(context.Background(), agentcore.NativeRun{ControlOnly: true, Compact: &agentcore.ManualCompaction{}})
	if err == nil || !strings.HasPrefix(err.Error(), "compact:") {
		t.Fatalf("err = %v, want a compact error on an empty history", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	first := &workLoop{turns: 4}
	done, err := workAgent(t, first, 100000, agentcore.DefaultCompactionSettings(), nil).RunNative(context.Background(), agentcore.NativeRun{Input: task("work")})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := workAgent(t, &workLoop{}, 100000, agentcore.DefaultCompactionSettings(), nil).RunNative(ctx, agentcore.NativeRun{State: done.NativeState, ControlOnly: true, Compact: &agentcore.ManualCompaction{}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want the cancellation", err)
	}
}

// A compaction provider answers the standalone summary; the run's own model
// never sees a summary request.
func TestNativeCompactionProviderAnswersSummaries(t *testing.T) {
	main := &workLoop{turns: 21}
	summarizer := &workLoop{}
	a := workAgent(t, main, 4000, agentcore.CompactionSettings{KeepRecentTokens: 1500}, func(cfg *agentcore.Config) {
		cfg.CompactionProvider = summarizer.provider()
	})
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: task("work")})
	if err != nil || result.Final != "finished" {
		t.Fatalf("run final=%q err=%v", result.Final, err)
	}
	if main.summaries != 0 || summarizer.summaries == 0 || summarizer.calls != 0 {
		t.Fatalf("main summaries=%d, compaction model summaries=%d calls=%d", main.summaries, summarizer.summaries, summarizer.calls)
	}
	if !strings.Contains(a.Describe(), "compaction_model") || !strings.Contains(a.Describe(), "fixture") {
		t.Fatal("Describe does not name the compaction model")
	}
	if _, err := agentcore.New(agentcore.Config{Model: "fixture", NativeProvider: main.provider(), CompactionProvider: &ai.FallbackProvider{}}); err == nil {
		t.Fatal("accepted a compaction provider without candidates")
	}
}

// The clearing boundary is part of NativeState: a resumed run clears the same
// results to the same bytes, and a corrupt boundary keeps the full results.
func TestNativeClearingBoundarySurvivesResume(t *testing.T) {
	settings := agentcore.DefaultCompactionSettings()
	settings.ClearToolResults = agentcore.ToolResultClearing{Trigger: 1000, Keep: 2, ClearAtLeast: 200}
	first := &workLoop{turns: 8}
	done, err := workAgent(t, first, 100000, settings, nil).RunNative(context.Background(), agentcore.NativeRun{Input: task("work")})
	if err != nil || first.summaries != 0 {
		t.Fatalf("run: %v summaries=%d", err, first.summaries)
	}
	var checkpoint map[string]json.RawMessage
	if err := json.Unmarshal(done.NativeState, &checkpoint); err != nil || len(checkpoint["cleared"]) == 0 {
		t.Fatal("NativeState lost the clearing boundary", err)
	}
	last := first.views[len(first.views)-1]
	if !strings.Contains(last, "Earlier tool result cleared to save context: work() returned") || strings.Contains(last, "evidence-1 ") || !strings.Contains(last, "evidence-7 ") {
		t.Fatal("the last request did not clear old results and keep recent ones")
	}

	resume := &workLoop{turns: 1}
	if _, err := workAgent(t, resume, 100000, settings, nil).RunNative(context.Background(), agentcore.NativeRun{State: done.NativeState, Input: task("continue")}); err != nil {
		t.Fatal(err)
	}
	if prefix := last[:strings.LastIndex(last, "evidence-6 ")]; !strings.HasPrefix(resume.views[0], prefix) {
		t.Fatal("a resumed run rewrote the cleared prefix: the prompt cache would miss")
	}

	checkpoint["cleared"] = json.RawMessage(`{"boundary":-1}`)
	corrupt, _ := json.Marshal(checkpoint)
	var notes []string
	again := &workLoop{turns: 1}
	off := settings
	off.ClearToolResults.Disabled = true
	if _, err := workAgent(t, again, 100000, off, nil).RunNative(context.Background(), agentcore.NativeRun{State: corrupt, Input: task("continue"), Sink: func(e agentcore.StreamEvent) { notes = append(notes, e.Note) }}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(notes, "\n"), "Discarding invalid tool-result clearing") || !strings.Contains(again.views[0], "evidence-1 ") {
		t.Fatal("a corrupt clearing boundary was not discarded")
	}
}

// reattacher restores a pinned file after each summary.
type reattacher struct {
	dropped *int
	text    string
	panics  bool
}

func (r reattacher) Name() string {
	if r.panics {
		return "broken-file"
	}
	return "pinned-file"
}
func (r reattacher) BeginRun(context.Context, agentcore.RunInfo) (agentcore.Extension, error) {
	return r, nil
}
func (r reattacher) ReattachAfterCompaction(_ context.Context, dropped []json.RawMessage, maxTokens int) string {
	if r.panics {
		panic("broken extension")
	}
	*r.dropped = len(dropped)
	if maxTokens <= 0 || maxTokens > 8192 {
		panic("reattach cap not passed")
	}
	if len(dropped) > 0 {
		dropped[0] = json.RawMessage(`{"role":"user","content":"MUTATED"}`)
	}
	return r.text
}

// A CompactionReattacher's text follows the summary, and a panicking one is
// reported and skipped.
func TestNativeReattacherFollowsTheSummary(t *testing.T) {
	dropped := 0
	var reported []string
	w := &workLoop{turns: 21}
	a := workAgent(t, w, 4000, agentcore.CompactionSettings{KeepRecentTokens: 1500}, func(cfg *agentcore.Config) {
		cfg.Extensions = []agentcore.ExtensionFactory{reattacher{panics: true, dropped: &dropped}, reattacher{dropped: &dropped, text: "PINNED main.go contents"}}
		cfg.Hooks.OnError = func(source string, _ error) { reported = append(reported, source) }
	})
	if _, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: task("work")}); err != nil {
		t.Fatal(err)
	}
	if w.summaries == 0 || dropped == 0 {
		t.Fatalf("summaries=%d dropped=%d", w.summaries, dropped)
	}
	last := w.views[len(w.views)-1]
	if !strings.Contains(last, "PINNED main.go contents") || !strings.Contains(last, "[Context reattached after the summary]") || strings.Contains(last, "MUTATED") {
		t.Fatal("the reattached context is missing after the summary, or a mutation leaked")
	}
	if !strings.Contains(strings.Join(reported, ","), "reattach[broken-file]") {
		t.Fatalf("a panicking reattacher was not reported: %v", reported)
	}
}
