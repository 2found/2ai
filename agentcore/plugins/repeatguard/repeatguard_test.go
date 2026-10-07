package repeatguard

import (
	"context"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
)

func mustGuard(t *testing.T, p *Plugin) *guardRun {
	t.Helper()
	g, err := p.resolve()
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if g == nil {
		t.Fatal("expected a guard")
	}
	return g
}

func call(name, args string) agentcore.ToolCall {
	return agentcore.ToolCall{Name: name, Arguments: args}
}

func TestRepeatGuard_FiresAtEachThresholdOnce(t *testing.T) {
	g := mustGuard(t, &Plugin{Thresholds: []int{3, 5}})
	c := call("run_sql", `{"q":"select 1"}`)

	if got := g.observe([]agentcore.ToolCall{c}); got != "" {
		t.Fatalf("call 1 must not nudge, got %q", got)
	}
	if got := g.observe([]agentcore.ToolCall{c}); got != "" {
		t.Fatalf("call 2 must not nudge, got %q", got)
	}
	third := g.observe([]agentcore.ToolCall{c})
	if third != repeatReminderShort {
		t.Fatalf("call 3 must deliver the short nudge, got %q", third)
	}
	if got := g.observe([]agentcore.ToolCall{c}); got != "" {
		t.Fatalf("call 4 is between thresholds, got %q", got)
	}
	fifth := g.observe([]agentcore.ToolCall{c})
	if !strings.Contains(fifth, "consecutive_calls: 5") || !strings.Contains(fifth, "run_sql") {
		t.Fatalf("call 5 must deliver the detailed nudge, got %q", fifth)
	}
	// Past the last threshold it goes quiet rather than nagging every turn.
	for i := 0; i < 5; i++ {
		if got := g.observe([]agentcore.ToolCall{c}); got != "" {
			t.Fatalf("beyond the last threshold must be silent, got %q", got)
		}
	}
}

func TestRepeatGuard_ArgumentOrderDoesNotBreakTheChain(t *testing.T) {
	g := mustGuard(t, &Plugin{Thresholds: []int{3}})
	// Same call, keys shuffled — models do this constantly. Treating these as
	// different calls would make the guard miss most real loops.
	seq := []string{
		`{"a":1,"b":{"x":true,"y":2}}`,
		`{"b":{"y":2,"x":true},"a":1}`,
		`{"a":1,"b":{"x":true,"y":2}}`,
	}
	var last string
	for _, args := range seq {
		last = g.observe([]agentcore.ToolCall{call("t", args)})
	}
	if last != repeatReminderShort {
		t.Fatalf("key order must not break the chain; got %q", last)
	}
}

func TestRepeatGuard_DifferentArgumentsResetTheChain(t *testing.T) {
	g := mustGuard(t, &Plugin{Thresholds: []int{3}})
	g.observe([]agentcore.ToolCall{call("t", `{"q":1}`)})
	g.observe([]agentcore.ToolCall{call("t", `{"q":1}`)})
	g.observe([]agentcore.ToolCall{call("t", `{"q":2}`)}) // different -> reset to 1
	if got := g.observe([]agentcore.ToolCall{call("t", `{"q":2}`)}); got != "" {
		t.Fatalf("count should be 2 after the reset, got %q", got)
	}
	if got := g.observe([]agentcore.ToolCall{call("t", `{"q":2}`)}); got != repeatReminderShort {
		t.Fatalf("the new chain should nudge at 3, got %q", got)
	}
}

func TestRepeatGuard_ExcludedToolIsTransparentNotAReset(t *testing.T) {
	// This is the whole point of exclusion: a bookkeeping call interleaved into
	// a loop must not launder it.
	g := mustGuard(t, &Plugin{Thresholds: []int{3}, Exclude: []string{"update_plan"}})
	c := call("run_sql", `{"q":1}`)
	plan := call("update_plan", `{"items":[]}`)

	g.observe([]agentcore.ToolCall{c})
	g.observe([]agentcore.ToolCall{plan})
	g.observe([]agentcore.ToolCall{c})
	g.observe([]agentcore.ToolCall{plan})
	if got := g.observe([]agentcore.ToolCall{c}); got != repeatReminderShort {
		t.Fatalf("interleaved excluded calls must not reset the chain, got %q", got)
	}
}

func TestRepeatGuard_IncludeLimitsTracking(t *testing.T) {
	g := mustGuard(t, &Plugin{Thresholds: []int{3}, Include: []string{"run_*"}, Exclude: []string{}})
	// An untracked tool never nudges no matter how often it repeats.
	for i := 0; i < 10; i++ {
		if got := g.observe([]agentcore.ToolCall{call("web_fetch", `{}`)}); got != "" {
			t.Fatalf("web_fetch is outside Include, got %q", got)
		}
	}
	c := call("run_sql", `{"q":1}`)
	g.observe([]agentcore.ToolCall{c})
	g.observe([]agentcore.ToolCall{c})
	if got := g.observe([]agentcore.ToolCall{c}); got != repeatReminderShort {
		t.Fatalf("included tool should nudge, got %q", got)
	}
}

func TestRepeatGuard_CountsWithinOneBatch(t *testing.T) {
	// A model that emits the same call three times in ONE assistant turn is
	// looping just as hard as one that spreads them over three turns.
	g := mustGuard(t, &Plugin{Thresholds: []int{3}})
	c := call("t", `{"q":1}`)
	if got := g.observe([]agentcore.ToolCall{c, c, c}); got != repeatReminderShort {
		t.Fatalf("expected a nudge from a single batch, got %q", got)
	}
}

func TestRepeatGuard_ResetClearsTheChain(t *testing.T) {
	g := mustGuard(t, &Plugin{Thresholds: []int{3}})
	c := call("t", `{"q":1}`)
	g.observe([]agentcore.ToolCall{c})
	g.observe([]agentcore.ToolCall{c})
	g.reset() // a steer or follow-up arrived
	if got := g.observe([]agentcore.ToolCall{c}); got != "" {
		t.Fatalf("reset must clear the count, got %q", got)
	}
}

func TestRepeatGuard_ArgumentPreviewIsBoundedButDetectionIsNot(t *testing.T) {
	long := strings.Repeat("z", 4000)
	g := mustGuard(t, &Plugin{Thresholds: []int{2, 3}, ArgumentsPreviewChars: 50})
	c := call("write", `{"body":"`+long+`"}`)
	g.observe([]agentcore.ToolCall{c})
	g.observe([]agentcore.ToolCall{c}) // threshold 2 -> short form
	detailed := g.observe([]agentcore.ToolCall{c})
	if !strings.Contains(detailed, "more chars)") {
		t.Fatalf("preview should report omission, got %q", detailed)
	}
	if len(detailed) > 1000 {
		t.Fatalf("detailed reminder is unbounded (%d bytes) — it rides into every later request", len(detailed))
	}
	// Detection still used the FULL argument string: a call differing only past
	// the preview cap must break the chain.
	other := call("write", `{"body":"`+long+`DIFFERENT"}`)
	g.reset()
	g.observe([]agentcore.ToolCall{c})
	if got := g.observe([]agentcore.ToolCall{other}); got != "" {
		t.Fatalf("a call differing beyond the preview cap must reset, got %q", got)
	}
}

func TestRepeatGuard_NonJSONArgumentsStillChain(t *testing.T) {
	g := mustGuard(t, &Plugin{Thresholds: []int{3}})
	c := call("t", "not json at all")
	g.observe([]agentcore.ToolCall{c})
	g.observe([]agentcore.ToolCall{c})
	if got := g.observe([]agentcore.ToolCall{c}); got != repeatReminderShort {
		t.Fatalf("unparseable arguments must still chain by raw text, got %q", got)
	}
}

func TestRepeatGuard_RejectsBadThresholds(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   []int
	}{
		{"below two", []int{1}},
		{"zero", []int{0, 3}},
		{"negative", []int{-2}},
		{"duplicate", []int{3, 3}},
	} {
		if _, err := (Plugin{Thresholds: tc.in}).resolve(); err == nil {
			t.Fatalf("%s: expected a configuration error", tc.name)
		}
	}
	// And composing it surfaces the error rather than installing a guard that
	// never fires — the whole reason validation is eager.
	if _, err := agentcore.BuildRegistry(Plugin{Thresholds: []int{1}}); err == nil {
		t.Fatal("composition must reject an invalid repeat-guard threshold")
	}
}

func TestRepeatGuard_DefaultsApplied(t *testing.T) {
	g := mustGuard(t, &Plugin{})
	if len(g.thresholds) != len(DefaultThresholds) || g.thresholds[0] != 3 {
		t.Fatalf("thresholds = %v, want %v", g.thresholds, DefaultThresholds)
	}
	if g.previewChars != defaultArgumentsPreviewChars {
		t.Fatalf("previewChars = %d", g.previewChars)
	}
	// Nothing is excluded by default: this plugin names no other capability's
	// tools. Bookkeeping transparency now comes from the run's predicate, which
	// TestRepeatGuard_BookkeepingToolIsTransparent covers.
	if len(g.exclude) != 0 {
		t.Fatalf("DefaultExclude should be empty, got %v", g.exclude)
	}
}

// An extension the loop holds must never be nil: "off" is expressed by not
// composing the plugin at all, so BeginRun always returns a working guard.
func TestRepeatGuard_BeginRunAlwaysYieldsAGuard(t *testing.T) {
	ext, err := Plugin{}.BeginRun(context.Background(), agentcore.RunInfo{})
	if err != nil || ext == nil {
		t.Fatalf("BeginRun = %v, %v; want a guard", ext, err)
	}
	g := ext.(*guardRun)
	if got := g.observe([]agentcore.ToolCall{call("t", "{}")}); got != "" {
		t.Fatalf("a single call must not nudge, got %q", got)
	}
}

// New material in the conversation clears the chain: the model now knows
// something it did not, so its earlier repetition is a different chain of
// reasoning. Only external input counts — replaying history must not reset it.
func TestRepeatGuard_ExternalInputResetsTheChain(t *testing.T) {
	g := mustGuard(t, &Plugin{Thresholds: []int{3}})
	c := call("run_sql", `{"q":"select 1"}`)
	g.observe([]agentcore.ToolCall{c, c})

	g.ObserveMessages(context.Background(), agentcore.PhaseRebase, 1, nil)
	if got := g.observe([]agentcore.ToolCall{c}); got == "" {
		t.Fatal("a rebase is not new information; the chain must survive it")
	}

	g.ObserveMessages(context.Background(), agentcore.PhaseExternalInput, 1, nil)
	for i := 0; i < 2; i++ {
		if got := g.observe([]agentcore.ToolCall{c}); got != "" {
			t.Fatalf("chain restarted, call %d must not nudge; got %q", i+1, got)
		}
	}
}

// The batch decision is what the loop actually consumes.
func TestRepeatGuard_InterceptBatchEmitsOneUserMessage(t *testing.T) {
	g := mustGuard(t, &Plugin{Thresholds: []int{2}})
	c := call("run_sql", `{"q":"select 1"}`)
	if d := g.InterceptBatch(context.Background(), []agentcore.ToolCall{c}); len(d.AdditionalContexts) != 0 {
		t.Fatalf("below threshold must add nothing, got %+v", d)
	}
	d := g.InterceptBatch(context.Background(), []agentcore.ToolCall{c})
	if len(d.AdditionalContexts) != 1 || d.AdditionalContexts[0].Role != agentcore.RoleUser {
		t.Fatalf("want one synthetic user message, got %+v", d.AdditionalContexts)
	}
}

func TestMatchToolPattern(t *testing.T) {
	cases := []struct {
		pattern, name string
		want          bool
	}{
		{"run_sql", "run_sql", true},
		{"run_sql", "run_sqlx", false},
		{"run_*", "run_sql", true},
		{"run_*", "web_fetch", false},
		{"*_sql", "run_sql", true},
		{"*_sql", "run_sqlx", false},
		{"*", "anything", true},
		{"mcp_*", "run_sql", false},
		{"a*c", "abc", true},
		{"a*c", "abd", false},
	}
	for _, c := range cases {
		if got := matchToolPattern(c.pattern, c.name); got != c.want {
			t.Errorf("matchToolPattern(%q, %q) = %v, want %v", c.pattern, c.name, got, c.want)
		}
	}
}

// TestRepeatGuard_BookkeepingToolIsTransparent replaces what used to be a
// hardcoded update_plan exclusion. The guard now asks the RUN whether a tool is
// administrative, so a planner from any package is transparent to the chain and
// this plugin depends on none of them.
//
// The property is the strong one: a bookkeeping call interleaved into a repeat
// loop must be invisible, NOT a reset — otherwise a stuck agent could launder a
// real loop by touching its plan between attempts.
func TestRepeatGuard_BookkeepingToolIsTransparent(t *testing.T) {
	ext, err := Plugin{Thresholds: []int{3}}.BeginRun(context.Background(), agentcore.RunInfo{
		Bookkeeping: func(name string) bool { return name == "some_planner" },
	})
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	g := ext.(*guardRun)

	c := call("run_sql", `{"q":1}`)
	plan := call("some_planner", `{"items":[]}`)

	g.observe([]agentcore.ToolCall{c})
	g.observe([]agentcore.ToolCall{plan})
	g.observe([]agentcore.ToolCall{c})
	g.observe([]agentcore.ToolCall{plan})
	if got := g.observe([]agentcore.ToolCall{c}); got != repeatReminderShort {
		t.Fatalf("an interleaved bookkeeping call laundered a real loop, got %q", got)
	}
}

// A run that supplies no predicate must still work: the guard simply tracks
// every tool. An extension must never assume the loop filled an optional field.
func TestRepeatGuard_NilBookkeepingPredicateIsSafe(t *testing.T) {
	ext, err := Plugin{Thresholds: []int{3}}.BeginRun(context.Background(), agentcore.RunInfo{})
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	if !ext.(*guardRun).tracks("anything") {
		t.Fatal("a nil predicate must not make every tool transparent")
	}
}
