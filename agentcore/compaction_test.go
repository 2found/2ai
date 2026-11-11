package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/ai"
)

// bigText returns a string of n bytes so token estimates cross thresholds.
func bigText(n int) string { return strings.Repeat("x", n) }

// TestLegacyCompactionEntryReplaysFully pins backward compatibility: a log
// written before the retained transcript existed has no Retained on its
// completion entries, and must still reduce to its full history rather than
// truncating to nothing. Old sessions stay resumable.
func TestLegacyCompactionEntryReplaysFully(t *testing.T) {
	log := []SessionEntry{
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "task"}},
		{Kind: EntryMessage, Message: &Message{Role: RoleAssistant, Content: "old work"}},
		{Kind: EntryCompaction},              // start
		{Kind: EntryCompaction, Final: true}, // completion, no Retained (legacy)
		{Kind: EntryMessage, Message: &Message{Role: RoleAssistant, Content: "new work"}},
	}
	rs := ReduceSession(log)
	want := []string{"task", "old work", "new work"}
	if len(rs.Messages) != len(want) {
		t.Fatalf("legacy log reduced to %d messages, want %d: %+v", len(rs.Messages), len(want), rs.Messages)
	}
	for i, w := range want {
		if rs.Messages[i].Content != w {
			t.Fatalf("message[%d] = %q, want %q", i, rs.Messages[i].Content, w)
		}
	}
	if rs.PendingCompaction {
		t.Fatal("a closed bracket must not leave PendingCompaction set")
	}
}

// TestCheckpointResetsThenAccumulates verifies the fold's shape directly: the
// most recent checkpoint replaces history, and entries appended after it chain
// onto that — including a later compaction, which resets again.
func TestCheckpointResetsThenAccumulates(t *testing.T) {
	log := []SessionEntry{
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "discarded by checkpoint 1"}},
		{Kind: EntryCompaction},
		{Kind: EntryCompaction, Final: true, Retained: []Message{
			{Role: RoleSystem, Content: "summary 1"},
			{Role: RoleAssistant, Content: "tail 1"},
		}},
		{Kind: EntryMessage, Message: &Message{Role: RoleAssistant, Content: "after checkpoint 1"}},
		{Kind: EntryCompaction},
		{Kind: EntryCompaction, Final: true, Retained: []Message{
			{Role: RoleSystem, Content: "summary 2"},
			{Role: RoleAssistant, Content: "tail 2"},
		}},
		{Kind: EntryMessage, Message: &Message{Role: RoleAssistant, Content: "after checkpoint 2"}},
	}
	rs := ReduceSession(log)
	want := []string{"summary 2", "tail 2", "after checkpoint 2"}
	if len(rs.Messages) != len(want) {
		t.Fatalf("reduced to %d messages, want %d: %+v", len(rs.Messages), len(want), rs.Messages)
	}
	for i, w := range want {
		if rs.Messages[i].Content != w {
			t.Fatalf("message[%d] = %q, want %q", i, rs.Messages[i].Content, w)
		}
	}
}

// TestPendingCompactionAfterCheckpointStillReruns covers the crash-mid-compaction
// case on top of an existing checkpoint: the unfinished start must still ask
// recovery to re-run compaction, and the history it hands over must be the last
// checkpoint plus the work done since — not the whole log.
func TestPendingCompactionAfterCheckpointStillReruns(t *testing.T) {
	log := []SessionEntry{
		{Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "discarded"}},
		{Kind: EntryCompaction},
		{Kind: EntryCompaction, Final: true, Retained: []Message{
			{Role: RoleSystem, Content: "summary"},
		}},
		{Kind: EntryMessage, Message: &Message{Role: RoleAssistant, Content: "since"}},
		{Kind: EntryCompaction}, // crash here: started, never completed
	}
	rs := ReduceSession(log)
	if !rs.PendingCompaction {
		t.Fatal("an unfinished compaction must reduce to PendingCompaction")
	}
	want := []string{"summary", "since"}
	if len(rs.Messages) != len(want) {
		t.Fatalf("reduced to %d messages, want %d: %+v", len(rs.Messages), len(want), rs.Messages)
	}
	for i, w := range want {
		if rs.Messages[i].Content != w {
			t.Fatalf("message[%d] = %q, want %q", i, rs.Messages[i].Content, w)
		}
	}
}

// TestRetainedTranscriptKeepsConversationalSystemMessages guards the subtle half
// of the write path. Only the run's OWN system prompt is excluded, because the
// run rebuilds it; a system message that is part of the conversation — a goal
// pin promoted into the head by an earlier compaction, or a caller's seed
// system message — has no other home and must survive.
func TestRetainedTranscriptKeepsConversationalSystemMessages(t *testing.T) {
	const system = "You are a helpful agent."
	const pinnedGoal = "[pinned goal]\nship it"

	t.Run("drops the run's own prompt", func(t *testing.T) {
		got := retainedTranscript([]Message{
			{Role: RoleSystem, Content: system},
			{Role: RoleSystem, Content: pinnedGoal},
			{Role: RoleAssistant, Content: "work"},
		}, system)
		if len(got) != 2 || got[0].Content != pinnedGoal {
			t.Fatalf("expected the prompt dropped and the goal pin kept: %+v", got)
		}
	})

	t.Run("keeps a leading system message that is not the prompt", func(t *testing.T) {
		got := retainedTranscript([]Message{
			{Role: RoleSystem, Content: pinnedGoal},
			{Role: RoleAssistant, Content: "work"},
		}, system)
		if len(got) != 2 || got[0].Content != pinnedGoal {
			t.Fatalf("a conversational system message was dropped: %+v", got)
		}
	})

	t.Run("snapshots rather than aliasing", func(t *testing.T) {
		live := []Message{
			{Role: RoleSystem, Content: system},
			{Role: RoleAssistant, Content: "original"},
		}
		got := retainedTranscript(live, system)
		live[1].Content = "mutated after the entry was written"
		if got[0].Content != "original" {
			t.Fatal("Retained aliases the live transcript; a durable entry must be a snapshot")
		}
	})
}

// The keep-recent window is clamped to half the budget: a consumer who sets
// Limits.MaxContextTokens below KeepRecentTokens (default 20k) would otherwise
// leave nothing old enough to summarize, and compaction would run every turn
// without shrinking anything.
func TestKeepRecentIsClampedToHalfTheBudget(t *testing.T) {
	if got := (CompactionSettings{}).keepRecent(6000); got != 3000 {
		t.Fatalf("keepRecent = %d, want 3000 (budget/2)", got)
	}
	if got := (CompactionSettings{KeepRecentTokens: 1500}).keepRecent(6000); got != 1500 {
		t.Fatalf("keepRecent = %d, want 1500 (explicit, under clamp)", got)
	}
	if got := (CompactionSettings{}).keepRecent(0); got != defaultKeepRecentTokens {
		t.Fatalf("keepRecent = %d, want default %d", got, defaultKeepRecentTokens)
	}
}

// sinkTool accepts a large content argument and returns a tiny confirmation —
// the bulk stays in the CALL, mirroring a full-file write tool.
type sinkTool struct{ calls int }

func (s *sinkTool) Name() string { return "write_file" }
func (s *sinkTool) Schema() ToolSchema {
	return ToolSchema{Name: "write_file", Description: "write a file", Parameters: map[string]any{"type": "object"}}
}
func (s *sinkTool) Run(_ context.Context, _ string) (string, error) {
	s.calls++
	return fmt.Sprintf("wrote #%d", s.calls), nil
}

// bigArgsStream emits tool calls whose ARGUMENTS carry the bulk (unique per
// call so no editing rule applies), and doubles as the compaction summarizer.
func bigArgsStream(target, argSize int, calls, summaries *int) ai.StreamFn {
	return func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		if strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
			*summaries++
			return nativeEmit(&ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "earlier milestones summarized"})})(ctx, model, view, options)
		}
		*calls++
		if *calls >= target {
			return nativeEmit(&ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "DONE"})})(ctx, model, view, options)
		}
		args := fmt.Sprintf(`{"path":"page.html","content":"%s-%d"}`, bigText(argSize), *calls)
		return nativeEmit(&ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(
			ai.ContentBlock{Type: "toolCall", ID: fmt.Sprintf("w%d", *calls), Name: "write_file", Arguments: json.RawMessage(args)},
		)})(ctx, model, view, options)
	}
}

func TestCompactionFiresWhenBulkIsInToolCallArguments(t *testing.T) {
	var calls, summaries int
	stream := bigArgsStream(20, 3000, &calls, &summaries) // ~750 est. tokens per call

	limits := DefaultLimits()
	limits.MaxTurns = 60
	limits.MaxToolCalls = 80
	limits.MaxContextTokens = 4000 // deliberately below default KeepRecentTokens

	agent, err := New(Config{
		NativeProvider: nativeCandidate("bigargs", stream),
		Model:          "bigargs",
		Tools:          NewToolSet(&sinkTool{}),
		Policy:         NewAllowList("write_file"),
		Limits:         &limits,
		// No Compaction override: the default 20k keep-recent window must be
		// clamped to the 4k budget or this run wedges (the pre-fix behavior).
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "build the page milestone by milestone")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	if res.Final != "DONE" {
		t.Fatalf("run did not finish (stop=%q final=%q turns=%d)", res.StopReason, res.Final, res.Turns)
	}
	if summaries == 0 {
		t.Fatalf("compaction never summarized: big tool-call arguments under a small budget must compact (turns=%d)", res.Turns)
	}
	// Bounded: 20 turns of ~750-token calls is ~15k tokens uncompacted; the
	// request views must have been folded down.
	if calls != 20 {
		t.Fatalf("expected exactly %d reasoning calls, got %d", 20, calls)
	}
}

// The bug this file exists for: before the window capped it, the loop compacted
// at a fixed 200k regardless of the model. Point a run at anything smaller and
// compaction never fires — the transcript grows past what the model can hold and
// the provider rejects it, which no retry or escalation can rescue because the
// transcript is simply too big.
func TestBudgetIsCappedByTheModelWindow(t *testing.T) {
	// A 32k model with the default (300k) configured ceiling must compact well
	// inside 32k, not at 300k.
	got := effectiveBudget(defaultContextTokenBudget, 32_000)
	if got >= 32_000 {
		t.Fatalf("budget %d does not fit a 32000-token window", got)
	}
}

// The other direction: the configured ceiling is a ceiling. A 1M-token model
// must not license a 1M-token transcript when the operator asked for less —
// that is the cost control.
func TestConfiguredBudgetStillCapsALargeWindow(t *testing.T) {
	if got := effectiveBudget(120_000, 1_000_000); got != 120_000 {
		t.Fatalf("effectiveBudget(120000, 1000000) = %d, want the configured 120000", got)
	}
}

// An unknown window (0) is "nobody could tell us", not "a window of zero". It
// must leave the configured budget alone rather than collapsing it, or every
// self-hosted endpoint would compact on the first turn.
func TestUnknownWindowLeavesTheConfiguredBudgetAlone(t *testing.T) {
	if got := effectiveBudget(150_000, 0); got != 150_000 {
		t.Fatalf("effectiveBudget(150000, 0) = %d, want 150000", got)
	}
	if got := effectiveBudget(0, 0); got != defaultContextTokenBudget {
		t.Fatalf("effectiveBudget(0, 0) = %d, want the default %d", got, defaultContextTokenBudget)
	}
}

// A context window holds the answer as well as the prompt. Budgeting the whole
// window means the loop only compacts once the input alone has filled it, with
// no room left to reply in — so the cap must reserve output headroom.
func TestBudgetReservesRoomToAnswerIn(t *testing.T) {
	const window = 200_000
	got := effectiveBudget(defaultContextTokenBudget, window)
	if got > window-outputHeadroomTokens {
		t.Fatalf("budget %d leaves less than %d tokens to answer in", got, outputHeadroomTokens)
	}
}

// A window smaller than the headroom must not produce a zero or negative
// budget: collapsing to zero would disable compaction on the *smallest* window — the exact inversion
// of what the cap is for.
func TestATinyWindowStillProducesAWorkingBudget(t *testing.T) {
	for _, window := range []int{1_000, 8_192, 16_385, 32_000} {
		got := effectiveBudget(defaultContextTokenBudget, window)
		if got <= 0 {
			t.Fatalf("window %d produced budget %d, which disables compaction", window, got)
		}
		if got >= window {
			t.Fatalf("window %d produced budget %d, which does not fit", window, got)
		}
	}
}

// The ladder is the reason the window lives on the rung. A run that escalates
// from a large-window model to a small-window one must re-derive its budget, or
// it carries the first rung's headroom onto a model that cannot hold it.
func TestEscalationRederivesTheBudgetForTheNewRung(t *testing.T) {
	big := effectiveBudget(defaultContextTokenBudget, 1_000_000)
	small := effectiveBudget(defaultContextTokenBudget, 32_000)
	if big <= 40_000 || small >= 32_000 {
		t.Fatalf("a 40k transcript must fit a 1M-token rung (%d) and compact on a 32k one (%d)", big, small)
	}
}

// End to end through a real run, and the pair below is the point: the SAME
// transcript must compact on a small-window model and not on an undeclared one.
// Without the wiring (Config → registry → agent → ladder → budget) every unit
// test above can pass while the loop still reads the raw configured ceiling, so
// only a run proves it.
func TestARunWithASmallWindowActuallyCompacts(t *testing.T) {
	if n := windowRunSummaries(t, 32_000); n == 0 {
		t.Fatal("a run on a 32k model never compacted; the window did not reach the loop")
	}
}

func TestTheSameRunWithNoDeclaredWindowDoesNotCompact(t *testing.T) {
	if n := windowRunSummaries(t, 0); n != 0 {
		t.Fatalf("a run with no declared window compacted %d times; the budget is not the configured one", n)
	}
}

// windowRunSummaries drives an identical bulky run at a given declared window
// and reports how many summarization calls the provider saw. Holding everything
// but the window fixed is what makes the pair above a controlled comparison
// rather than two independent assertions.
func windowRunSummaries(t *testing.T, window int) int {
	t.Helper()

	work := bulkTool{size: 60_000}
	var calls, summaries int
	stream := func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		if strings.HasPrefix(ai.GetCurrentSystemPrompt(view.Messages()), "Summarize the conversation") {
			summaries++
			return nativeEmit(&ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "earlier work summarized"})})(ctx, model, view, options)
		}
		calls++
		if calls >= 4 {
			return nativeEmit(&ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})})(ctx, model, view, options)
		}
		return nativeEmit(&ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(
			ai.ContentBlock{Type: "toolCall", ID: fmt.Sprintf("c%d", calls), Name: "work", Arguments: json.RawMessage(`{}`)},
		)})(ctx, model, view, options)
	}
	var model json.RawMessage
	if window > 0 {
		model = json.RawMessage(fmt.Sprintf(`{"id":"m","contextWindow":%d}`, window))
	} else {
		model = json.RawMessage(`{"id":"m"}`)
	}
	agent, err := New(Config{
		NativeProvider: &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: model, Stream: stream}}},
		Model:          "m",
		ContextWindow:  window,
		Tools:          NewToolSet(work),
		Policy:         NewAllowList("work"),
		Limits: &Limits{
			MaxTurns: 8, MaxToolCalls: 8,
			MaxToolResultLen: 64 * 1024,
			MaxContextTokens: defaultContextTokenBudget,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	return summaries
}

// bulkTool returns a payload large enough that a few calls cross a small
// window, so compaction is exercised for real rather than merely configured.
type bulkTool struct{ size int }

func (bulkTool) Name() string { return "work" }
func (bulkTool) Schema() ToolSchema {
	return ToolSchema{Name: "work", Description: "does work", Parameters: map[string]any{"type": "object"}}
}
func (b bulkTool) Run(context.Context, string) (string, error) {
	return strings.Repeat("x", b.size), nil
}

// A composition that says nothing about compaction still compacts and clears
// tool results, with the run's own models summarizing.
func TestDefaultCompactionIsDescribed(t *testing.T) {
	agent, err := New(Config{
		NativeProvider: scriptedNativeProvider(AssistantText("ok")),
		Model:          "test",
		Policy:         DenyAll{},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	desc := agent.Describe()
	for _, want := range []string{"keep_recent=20000", "clear_tool_results=on", "shared_cache=false"} {
		if !strings.Contains(desc, want) {
			t.Fatalf("Describe lacks %q:\n%s", want, desc)
		}
	}
}

// A skill read before a summary is reattached after it, newest reading first
// and each skill once; a failed read is not, and the room is never exceeded.
func TestReattachSkillsKeepsTheNewestReadingOfEachSkill(t *testing.T) {
	call := func(id, skill string) json.RawMessage {
		raw, _ := json.Marshal(map[string]any{"role": "assistant", "stopReason": "toolUse", "content": []any{map[string]any{"type": "toolCall", "id": id, "name": readSkillToolName, "arguments": map[string]any{"id": skill}}}})
		return raw
	}
	result := func(id, text string, isError bool) json.RawMessage {
		raw, _ := json.Marshal(map[string]any{"role": "toolResult", "toolCallId": id, "toolName": readSkillToolName, "isError": isError, "content": []any{map[string]any{"type": "text", "text": text}}})
		return raw
	}
	messages := []json.RawMessage{
		call("r1", "sql"), result("r1", "SQL steps v1", false),
		call("r2", "deploy"), result("r2", "no such skill", true),
		call("r3", "sql"), result("r3", "SQL steps v2", false),
		call("r4", "review"), result("r4", "Review steps", false),
	}
	got := reattachSkills(messages, 4096)
	if !strings.Contains(got, `[Skill "sql"`) || !strings.Contains(got, "SQL steps v2") || strings.Contains(got, "v1") || strings.Contains(got, "no such skill") || !strings.Contains(got, "Review steps") {
		t.Fatalf("reattached skills:\n%s", got)
	}
	if strings.Index(got, "Review steps") > strings.Index(got, "SQL steps v2") {
		t.Fatal("the most recent skill must come first")
	}
	if small := reattachSkills(messages, 120); len(small) > 120 || !strings.Contains(small, "Review steps") || strings.Contains(small, "SQL") {
		t.Fatalf("a skill that does not fit was not skipped:\n%s", small)
	}
	if reattachSkills(nil, 4096) != "" {
		t.Fatal("reattached a skill from no messages")
	}
}
