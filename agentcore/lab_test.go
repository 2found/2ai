package agentcore

import (
	"fmt"
	"strings"
	"testing"
)

// buildPrompt mirrors buildSystemPrompt's section layout so the fold parser is
// exercised against the real prompt shape.
func labPrompt(soul, mem, skillLine string) string {
	s := "# Identity\n" + soul + "\n\n"
	if mem != "" {
		s += "# Recalled memory\nThe following are durable facts from prior runs.\n- (fact) " + mem + "\n\n"
	}
	if skillLine != "" {
		s += "# Available skills\nYou have on-demand skills.\n" + skillLine + "\n"
	}
	return s
}

func TestFoldStepsBasicTurn(t *testing.T) {
	records := []TurnRecord{
		{
			Messages: []Message{
				{Role: RoleSystem, Content: labPrompt("You are a tester.", "user likes brevity", "- id: search — Search: find things")},
				{Role: RoleUser, Content: "do the thing"},
			},
			Response:   "calling search",
			ToolCalls:  []ToolCall{{ID: "c1", Name: "search", Arguments: `{"q":"x"}`}},
			Tools:      []string{"search", "read_skill"},
			StopReason: "tool_calls",
			TokensIn:   100, TokensOut: 20, CostUSD: 0.001,
		},
		{
			Messages: []Message{
				{Role: RoleSystem, Content: labPrompt("You are a tester.", "user likes brevity", "- id: search — Search: find things")},
				{Role: RoleUser, Content: "do the thing"},
				{Role: RoleAssistant, Content: "calling search", ToolCalls: []ToolCall{{ID: "c1", Name: "search", Arguments: `{"q":"x"}`}}},
				{Role: RoleTool, ToolCallID: "c1", Name: "search", Content: "found it"},
			},
			Response:   "done",
			StopReason: "stop",
			TokensIn:   140, TokensOut: 10, CostUSD: 0.0008,
		},
	}

	steps := FoldSteps(records)
	if len(steps) != 2 {
		t.Fatalf("want 2 steps, got %d", len(steps))
	}

	s0 := steps[0]
	if s0.Kind != LabStepTurn || s0.Turn != 1 {
		t.Fatalf("step0 kind/turn = %s/%d", s0.Kind, s0.Turn)
	}
	if s0.Persona != "You are a tester." {
		t.Fatalf("persona = %q", s0.Persona)
	}
	if len(s0.Memory) != 1 || s0.Memory[0] != "(fact) user likes brevity" {
		t.Fatalf("memory = %v", s0.Memory)
	}
	if len(s0.SkillsAdvertised) != 1 || s0.SkillsAdvertised[0].ID != "search" || s0.SkillsAdvertised[0].Name != "Search" {
		t.Fatalf("advertised = %+v", s0.SkillsAdvertised)
	}
	if len(s0.ToolCalls) != 1 || s0.ToolCalls[0].Result != "found it" {
		t.Fatalf("tool call result not paired across turns: %+v", s0.ToolCalls)
	}
	if s0.CumTokensIn != 100 || s0.CumCostUSD != 0.001 {
		t.Fatalf("cum after step0 = %d/%v", s0.CumTokensIn, s0.CumCostUSD)
	}

	s1 := steps[1]
	if s1.CumTokensIn != 240 || s1.CumTokensOut != 30 {
		t.Fatalf("cumulative tokens wrong: in=%d out=%d", s1.CumTokensIn, s1.CumTokensOut)
	}
}

func TestFoldStepsLoadedSkills(t *testing.T) {
	records := []TurnRecord{
		{
			Messages:  []Message{{Role: RoleSystem, Content: labPrompt("p", "", "- id: deep — Deep: deep skill")}},
			ToolCalls: []ToolCall{{ID: "r1", Name: readSkillToolName, Arguments: `{"id":"deep"}`}},
			Tools:     []string{readSkillToolName},
		},
		{
			Messages: []Message{{Role: RoleSystem, Content: labPrompt("p", "", "- id: deep — Deep: deep skill")}},
			Response: "ok",
		},
	}
	steps := FoldSteps(records)
	if len(steps[0].SkillsLoaded) != 1 || steps[0].SkillsLoaded[0] != "deep" {
		t.Fatalf("step0 loaded = %v", steps[0].SkillsLoaded)
	}
	// Loaded set carries forward.
	if len(steps[1].SkillsLoaded) != 1 || steps[1].SkillsLoaded[0] != "deep" {
		t.Fatalf("step1 loaded should carry forward: %v", steps[1].SkillsLoaded)
	}
	// Advertised but not loaded is visible as a gap the UI can render.
	if len(steps[0].SkillsAdvertised) != 1 {
		t.Fatalf("advertised = %+v", steps[0].SkillsAdvertised)
	}
}

func TestFoldStepsCompactionStep(t *testing.T) {
	sys := labPrompt("p", "", "")
	records := []TurnRecord{
		{
			Messages: []Message{{Role: RoleSystem, Content: sys}, {Role: RoleUser, Content: "hi"}},
			Response: "a", TokensIn: 50,
		},
		{
			// Compaction folded a summary message into this turn's context.
			Messages: []Message{
				{Role: RoleSystem, Content: sys},
				{Role: RoleSystem, Content: summaryMarker + "\nearlier: user asked many things"},
				{Role: RoleUser, Content: "more"},
			},
			Response: "b", TokensIn: 30,
		},
	}
	steps := FoldSteps(records)
	if len(steps) != 3 {
		t.Fatalf("want 3 steps (turn, compaction, turn), got %d", len(steps))
	}
	if steps[1].Kind != LabStepCompaction {
		t.Fatalf("step1 should be compaction, got %s", steps[1].Kind)
	}
	if steps[1].Summary != "earlier: user asked many things" {
		t.Fatalf("summary = %q", steps[1].Summary)
	}
	// Cumulative carries through the compaction step unchanged.
	if steps[1].CumTokensIn != 50 {
		t.Fatalf("cum at compaction = %d", steps[1].CumTokensIn)
	}
	if steps[2].CumTokensIn != 80 {
		t.Fatalf("cum after final turn = %d", steps[2].CumTokensIn)
	}
}

func TestFoldNoSecretsLeak(t *testing.T) {
	records := []TurnRecord{{
		Messages:  []Message{{Role: RoleSystem, Content: labPrompt("p", "", "")}},
		ToolCalls: []ToolCall{{ID: "c1", Name: "http", Arguments: `{"token":"{{cred:API_KEY}}"}`}},
	}}
	steps := FoldSteps(records)
	if steps[0].ToolCalls[0].Args != `{"token":"{{cred:API_KEY}}"}` {
		t.Fatalf("fold must preserve placeholder form, got %q", steps[0].ToolCalls[0].Args)
	}
}

// Replayed runs used to hardcode Allowed: true because the persisted LLM-call
// trace carried no gate outcome. That is worse than missing data: a blocked
// call rendered as allowed on the Lab. Empty ToolGates is the historical
// encoding and must stay allowed; a recorded denial must not.
func TestFoldStepsCarriesToolDenial(t *testing.T) {
	steps := FoldSteps([]TurnRecord{{
		Messages:  []Message{{Role: RoleSystem, Content: labPrompt("p", "", "")}},
		ToolCalls: []ToolCall{{ID: "c1", Name: "write_dashboard", Arguments: "{}"}},
		ToolGates: []ToolGate{{CallID: "c1", Allowed: false, Reason: "tool 'write_dashboard' is not permitted"}},
	}})
	if len(steps) != 1 || len(steps[0].ToolCalls) != 1 {
		t.Fatalf("want 1 step with 1 tool call, got %+v", steps)
	}
	tc := steps[0].ToolCalls[0]
	if tc.Allowed {
		t.Fatal("replayed denial rendered as allowed")
	}
	if tc.Error != "tool 'write_dashboard' is not permitted" {
		t.Fatalf("denial reason = %q", tc.Error)
	}
}

func TestFoldStepsLegacyGatesDefaultAllowed(t *testing.T) {
	steps := FoldSteps([]TurnRecord{{
		Messages:  []Message{{Role: RoleSystem, Content: labPrompt("p", "", "")}},
		ToolCalls: []ToolCall{{ID: "c1", Name: "search", Arguments: "{}"}},
	}})
	if !steps[0].ToolCalls[0].Allowed {
		t.Fatal("historical row without gates must not be rewritten as denied")
	}
}

// Live explain folds at StepGate, before persistTrace has written
// tool_gates_json. Without the overlay, FoldSteps treats empty gates as the
// historical Allowed:true default and a live denial renders as allowed until
// the run ends. ApplyLiveGates is the in-memory feed that closes that gap.
func TestApplyLiveGatesOverlaysInMemoryDenial(t *testing.T) {
	records := []TurnRecord{{
		Messages:  []Message{{Role: RoleSystem, Content: labPrompt("p", "", "")}},
		ToolCalls: []ToolCall{{ID: "c1", Name: "write_dashboard", Arguments: "{}"}},
	}}
	traces := []ToolTrace{{CallID: "c1", Tool: "write_dashboard", Allowed: false, Reason: "not permitted"}}
	steps := FoldSteps(ApplyLiveGates(records, traces))
	if len(steps) != 1 || len(steps[0].ToolCalls) != 1 {
		t.Fatalf("want 1 step with 1 tool call, got %+v", steps)
	}
	if steps[0].ToolCalls[0].Allowed {
		t.Fatal("live denial rendered as allowed")
	}
	if steps[0].ToolCalls[0].Error != "not permitted" {
		t.Fatalf("denial reason = %q", steps[0].ToolCalls[0].Error)
	}
}

func TestApplyLiveGatesLeavesPersistedGates(t *testing.T) {
	records := []TurnRecord{{
		Messages:  []Message{{Role: RoleSystem, Content: labPrompt("p", "", "")}},
		ToolCalls: []ToolCall{{ID: "c1", Name: "search", Arguments: "{}"}},
		ToolGates: []ToolGate{{CallID: "c1", Allowed: false, Reason: "persisted"}},
	}}
	traces := []ToolTrace{{CallID: "c1", Allowed: true}}
	got := ApplyLiveGates(records, traces)
	if len(got[0].ToolGates) != 1 || got[0].ToolGates[0].Allowed || got[0].ToolGates[0].Reason != "persisted" {
		t.Fatalf("persisted gates rewritten: %+v", got[0].ToolGates)
	}
}

func TestApplyLiveGatesEmptyTracesIsIdentity(t *testing.T) {
	records := []TurnRecord{{
		Messages:  []Message{{Role: RoleSystem, Content: labPrompt("p", "", "")}},
		ToolCalls: []ToolCall{{ID: "c1", Name: "search", Arguments: "{}"}},
	}}
	got := ApplyLiveGates(records, nil)
	if len(got) != 1 || len(got[0].ToolGates) != 0 {
		t.Fatalf("empty traces must not invent gates: %+v", got)
	}
}

// stepsWithCompactionsAt builds a folded step list of `turns` turns with a
// compaction step closing the span at each of the given turns, and cumulative
// accounting that advances one unit per step — enough to check that a chapter's
// own spend is differenced, not copied.
func stepsWithCompactionsAt(turns int, compactAt ...int) []LabStep {
	mark := map[int]bool{}
	for _, t := range compactAt {
		mark[t] = true
	}
	var steps []LabStep
	cumIn, cumOut, cumCost := 0, 0, 0.0
	add := func(turn int, kind LabStepKind, summary string, calls int) {
		cumIn += 10
		cumOut += 2
		cumCost += 0.5
		tc := make([]LabToolCall, calls)
		steps = append(steps, LabStep{
			Index: len(steps), Turn: turn, Kind: kind, Summary: summary,
			ToolCalls: tc, CumTokensIn: cumIn, CumTokensOut: cumOut, CumCostUSD: cumCost,
		})
	}
	for turn := 1; turn <= turns; turn++ {
		if mark[turn] {
			add(turn, LabStepCompaction,
				fmt.Sprintf("## Goal\nFinish the audit\n## Progress\n### Done\n- [x] shards through turn %d reconciled", turn), 0)
		}
		add(turn, LabStepTurn, "", 1)
	}
	return steps
}

// TestRunChaptersDivideAtCompactions is the core claim: a long run comes back
// as a handful of navigable spans instead of thousands of undifferentiated
// steps, and each span says which steps it covers so a client can turn a
// chapter into a page request without a second lookup.
func TestRunChaptersDivideAtCompactions(t *testing.T) {
	steps := stepsWithCompactionsAt(20, 6, 13)
	chapters := RunChapters(steps)

	if len(chapters) != 3 {
		t.Fatalf("two compactions must yield three chapters, got %d", len(chapters))
	}
	// The spans must tile the step list exactly: no step in two chapters, none
	// in none. A reader who opened every chapter must have seen the whole run.
	if chapters[0].FirstStep != 0 {
		t.Fatalf("first chapter starts at step %d", chapters[0].FirstStep)
	}
	for i := 1; i < len(chapters); i++ {
		if chapters[i].FirstStep != chapters[i-1].LastStep+1 {
			t.Fatalf("chapters %d and %d do not tile: %d then %d",
				i-1, i, chapters[i-1].LastStep, chapters[i].FirstStep)
		}
	}
	if last := chapters[len(chapters)-1].LastStep; last != len(steps)-1 {
		t.Fatalf("chapters end at step %d, run has %d steps", last, len(steps))
	}
	total := 0
	for _, c := range chapters {
		total += c.Steps
	}
	if total != len(steps) {
		t.Fatalf("chapters cover %d steps, run has %d", total, len(steps))
	}
}

// TestRunChaptersCloseOnTheirOwnSummary pins the direction. A compaction
// summarizes what came BEFORE it, so it has to close its chapter rather than
// open the next one — a list that labelled each span with the summary written
// at its start would describe every chapter by the work preceding it.
func TestRunChaptersCloseOnTheirOwnSummary(t *testing.T) {
	steps := stepsWithCompactionsAt(20, 6, 13)
	chapters := RunChapters(steps)

	if got := chapters[0].Summary; got == "" {
		t.Fatal("the first chapter must carry the summary that closed it")
	}
	for i, want := range []int{6, 13} {
		if !strings.Contains(chapters[i].Summary, fmt.Sprintf("turn %d", want)) {
			t.Fatalf("chapter %d closed with the wrong summary: %q", i, chapters[i].Summary)
		}
		// The compaction step is the last step OF the chapter it summarizes, not
		// the first step of the one after. Getting this off by one puts each
		// chapter's closing account one step outside the span it describes.
		if last := steps[chapters[i].LastStep]; last.Kind != LabStepCompaction {
			t.Fatalf("chapter %d ends on a %v step; a summarized chapter ends on its compaction",
				i, last.Kind)
		}
		if last := steps[chapters[i].LastStep]; last.Summary != chapters[i].Summary {
			t.Fatalf("chapter %d carries a summary from step %d, not from the step that closed it",
				i, chapters[i].LastStep)
		}
	}
	// The last chapter is the stretch after the final compaction. It has no
	// summary because none was ever written for it, and claiming otherwise
	// would attribute an earlier chapter's account to later work.
	if s := chapters[2].Summary; s != "" {
		t.Fatalf("the final chapter must not borrow a summary: %q", s)
	}
}

// TestRunChaptersAccountPerChapter checks the numbers are differenced out of
// the cumulative totals rather than reported as running totals — otherwise
// every chapter would appear to cost more than the last no matter what
// happened in it.
func TestRunChaptersAccountPerChapter(t *testing.T) {
	steps := stepsWithCompactionsAt(20, 6, 13)
	chapters := RunChapters(steps)

	sumIn, sumCalls := 0, 0
	for _, c := range chapters {
		if c.TokensIn <= 0 {
			t.Fatalf("chapter %d reports %d input tokens", c.Index, c.TokensIn)
		}
		sumIn += c.TokensIn
		sumCalls += c.ToolCalls
	}
	last := steps[len(steps)-1]
	if sumIn != last.CumTokensIn {
		t.Fatalf("chapter tokens sum to %d, run total is %d", sumIn, last.CumTokensIn)
	}
	if sumCalls != 20 {
		t.Fatalf("chapter tool calls sum to %d, run made 20", sumCalls)
	}
}

// TestRunChaptersOnARunThatNeverCompacted covers the short-run case, which is
// most runs: one chapter spanning everything, so a client renders the same way
// without special-casing.
func TestRunChaptersOnARunThatNeverCompacted(t *testing.T) {
	chapters := RunChapters(stepsWithCompactionsAt(5))
	if len(chapters) != 1 {
		t.Fatalf("want one chapter, got %d", len(chapters))
	}
	if chapters[0].FirstStep != 0 || chapters[0].LastStep != 4 || chapters[0].Steps != 5 {
		t.Fatalf("the single chapter must span the run: %+v", chapters[0])
	}
	if chapters[0].Title == "" {
		t.Fatal("even an unsummarized chapter needs a label")
	}
}

func TestRunChaptersOnAnEmptyRun(t *testing.T) {
	if got := RunChapters(nil); len(got) != 0 {
		t.Fatalf("an empty run has no chapters, got %d", len(got))
	}
}

// TestChapterTitleReadsTheSummary checks the label comes from the model's own
// words. The checkpoint format is mostly headings and list markers, so a naive
// "first line" would title every chapter "## Goal".
func TestChapterTitleReadsTheSummary(t *testing.T) {
	cases := []struct {
		name    string
		summary string
		want    string
	}{
		{"skips headings", "## Goal\nReconcile the ledger\n## Progress", "Reconcile the ledger"},
		{"strips list markers", "## Progress\n### Done\n- [x] shards 1-40 audited", "shards 1-40 audited"},
		{"falls back to the first real line", "just a plain summary", "just a plain summary"},
		{"labels an unsummarized chapter", "", "The run"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := chapterTitle(tc.summary, 0); got != tc.want {
				t.Fatalf("title = %q, want %q", got, tc.want)
			}
		})
	}
}
