package agentcore

import (
	"context"
	"strings"
	"testing"

	"github.com/2found/2ai/core/ai"
)

// These cover the composition MECHANISM: seams, priorities, reversibility,
// diagnostics. The plugins themselves live in feature/plugins/* and are
// covered there — this file must keep passing even if every plugin package is
// deleted, which is the point of the split.

// --- test doubles ---------------------------------------------------------

// seamPlugin claims a named seam, so conflicts can be provoked without a real
// capability.
type seamPlugin struct {
	name string
	seam string
	err  error
}

func (p seamPlugin) Name() string { return p.name }

func (p seamPlugin) Register(r *Registry) error {
	if p.err != nil {
		return p.err
	}
	if p.seam != "" {
		return r.claim(p.seam)
	}
	return nil
}

// hookPlugin appends its label whenever its before-hook fires.
type hookPlugin struct {
	name     string
	priority Priority
	log      *[]string
}

func (p hookPlugin) Name() string { return p.name }

func (p hookPlugin) Register(r *Registry) error {
	r.AddHooks(p.priority, Hooks{
		Before: []BeforeToolCall{
			func(ctx context.Context, call ToolCall) Decision {
				*p.log = append(*p.log, p.name)
				return Allowed()
			},
		},
	})
	return nil
}

// funcPlugin adapts a closure into a Plugin.
type funcPlugin struct {
	name string
	fn   func(*Registry) error
}

func (p funcPlugin) Name() string               { return p.name }
func (p funcPlugin) Register(r *Registry) error { return p.fn(r) }
func modelOf(m string) Plugin {
	return funcPlugin{name: "model", fn: func(r *Registry) error {
		if err := r.SetModel(&FauxProvider{}, m); err != nil {
			return err
		}
		return r.SetNativeProvider(scriptedNativeProvider(AssistantText("done")))
	}}
}

// --- composition ----------------------------------------------------------

func TestBuild_MinimalComposition(t *testing.T) {
	a, err := Build(modelOf("m"))
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	// Every default an agent needs to be safe comes from the registry, not from
	// Config — a plugin-built agent is as governed as a Config-built one.
	if _, ok := a.policy.(DenyAll); !ok {
		t.Fatalf("an agent with no policy plugin must deny by default, got %T", a.policy)
	}
	if a.tools == nil {
		t.Fatal("tools must default to an empty set, not nil")
	}
	if a.limits.MaxTurns == 0 {
		t.Fatal("limits must default to DefaultLimits()")
	}
}

func TestBuild_RejectsAnAgentWithNoModel(t *testing.T) {
	if _, err := Build(seamPlugin{name: "noop"}); err == nil {
		t.Fatal("a composition with no model must not build")
	}
}

func TestBuild_SkipsNilPlugins(t *testing.T) {
	if _, err := Build(nil, modelOf("m"), nil); err != nil {
		t.Fatalf("a nil entry must be skipped, not fatal: %v", err)
	}
}

// --- seams ----------------------------------------------------------------

func TestRegistry_SeamHasExactlyOneProvider(t *testing.T) {
	r := newRegistry()
	if err := r.register(seamPlugin{name: "first", seam: "policy"}); err != nil {
		t.Fatalf("first claim: %v", err)
	}
	err := r.register(seamPlugin{name: "second", seam: "policy"})
	if err == nil {
		t.Fatal("the second claim on a seam must fail, not silently overwrite")
	}
	// A conflict you cannot attribute is a conflict you cannot fix.
	if !strings.Contains(err.Error(), "first") || !strings.Contains(err.Error(), "second") {
		t.Fatalf("conflict must name both plugins, got: %v", err)
	}
}

func TestRegistry_EverySetterClaimsItsSeam(t *testing.T) {
	// Two plugins fighting over the same knob is a configuration error whichever
	// knob it is, so the claim is checked across a representative spread rather
	// than only on the seams that were convenient to test.
	cases := []struct {
		seam string
		set  func(*Registry) error
	}{
		{"goal", func(r *Registry) error { return r.SetGoal("g") }},
		{"limits", func(r *Registry) error { return r.SetLimits(DefaultLimits()) }},
		{"memory", func(r *Registry) error { return r.SetMemory(nil) }},
		{"budget_gate", func(r *Registry) error { return r.SetBudgetGate(nil) }},
		{"prompt_cache", func(r *Registry) error { return r.SetPromptCache("k", "") }},
		{"session", func(r *Registry) error { return r.SetSession(NewMemorySessionStore(), "s", false) }},
		{"compaction", func(r *Registry) error { return r.SetCompaction(DefaultCompactionSettings()) }},
		{"compaction_model", func(r *Registry) error { return r.SetCompactionModel(scriptedNativeProvider()) }},
	}
	for _, tc := range cases {
		r := newRegistry()
		r.current = "a"
		if err := tc.set(r); err != nil {
			t.Fatalf("%s: first set: %v", tc.seam, err)
		}
		r.current = "b"
		if err := tc.set(r); err == nil {
			t.Fatalf("%s: a second plugin must not be able to claim it", tc.seam)
		}
	}
}

// A compaction model without candidates could never answer a summary.
func TestRegistry_CompactionModelNeedsCandidates(t *testing.T) {
	for _, p := range []*ai.FallbackProvider{nil, {}} {
		if err := newRegistry().SetCompactionModel(p); err == nil {
			t.Fatalf("accepted compaction model %v", p)
		}
	}
}

func TestRegistry_DuplicatePluginNameIsAnError(t *testing.T) {
	r := newRegistry()
	if err := r.register(seamPlugin{name: "dup"}); err != nil {
		t.Fatalf("first: %v", err)
	}
	if err := r.register(seamPlugin{name: "dup"}); err == nil {
		t.Fatal("registering the same plugin name twice must fail")
	}
}

func TestRegistry_FailedPluginLeavesNothingBehind(t *testing.T) {
	r := newRegistry()
	if err := r.register(seamPlugin{name: "boom", err: context.Canceled}); err == nil {
		t.Fatal("expected the plugin error to propagate")
	}
	if len(r.Plugins()) != 0 {
		t.Fatalf("a failed plugin must not stay listed: %v", r.Plugins())
	}
	// Its effects are rolled back too, so a corrected retry works.
	if err := r.register(seamPlugin{name: "boom", seam: "policy"}); err != nil {
		t.Fatalf("retry after a failed registration: %v", err)
	}
}

// TestRegistry_PartialPluginRollsBack: a plugin that claims two seams and then
// fails must leave neither claimed, or a retry can never succeed.
func TestRegistry_PartialPluginRollsBack(t *testing.T) {
	r := newRegistry()
	half := funcPlugin{name: "half", fn: func(r *Registry) error {
		if err := r.SetGoal("g"); err != nil {
			return err
		}
		return context.Canceled
	}}
	if err := r.register(half); err == nil {
		t.Fatal("expected failure")
	}
	if r.Provider("goal") != "" {
		t.Fatal("a failed plugin must not leave a seam claimed")
	}
	if r.goal != "" {
		t.Fatalf("a failed plugin must not leave its value behind, got %q", r.goal)
	}
}

// --- reversibility --------------------------------------------------------

func TestRegistry_UnloadReversesEverything(t *testing.T) {
	r, err := BuildRegistry(
		modelOf("m"),
		funcPlugin{name: "policy", fn: func(r *Registry) error {
			r.AddHooks(PriorityGate, Hooks{Before: []BeforeToolCall{func(context.Context, ToolCall) Decision { return Allowed() }}})
			return r.SetPolicy(NewAllowList("echo"))
		}},
		funcPlugin{name: "tools", fn: func(r *Registry) error { return r.AddTools(&echoTool{name: "echo"}) }},
	)
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	if r.Provider("policy") != "policy" {
		t.Fatalf("policy seam owner = %q", r.Provider("policy"))
	}
	if len(r.mergedHooks().Before) != 1 {
		t.Fatal("the policy plugin's hook did not register")
	}

	if err := r.Unload("policy"); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	// Seam released, hook withdrawn, previous value restored — all three, or the
	// plugin was a patch rather than a unit.
	if r.Provider("policy") != "" {
		t.Fatal("unload must release the seam so another plugin can claim it")
	}
	if len(r.mergedHooks().Before) != 0 {
		t.Fatal("unload must remove the plugin's hooks")
	}
	if _, ok := r.policy.(DenyAll); !ok {
		t.Fatalf("unload must restore the previous provider, got %T", r.policy)
	}
	// Unloading one plugin must not disturb another.
	if len(r.tools.Names()) != 1 {
		t.Fatalf("unload leaked into another plugin: %v", r.tools.Names())
	}
	// And the freed seam really is claimable again.
	if err := r.register(funcPlugin{name: "policy2", fn: func(r *Registry) error { return r.SetPolicy(DenyAll{}) }}); err != nil {
		t.Fatalf("re-registering into the freed seam: %v", err)
	}
}

func TestRegistry_UnloadWithdrawsTools(t *testing.T) {
	r, err := BuildRegistry(
		modelOf("m"),
		funcPlugin{name: "tools", fn: func(r *Registry) error { return r.AddTools(&echoTool{name: "a"}) }},
	)
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	if err := r.Unload("tools"); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	if len(r.tools.Names()) != 0 {
		t.Fatalf("unloading a tools plugin must withdraw its tools: %v", r.tools.Names())
	}
}

// contributor registers one extension, one tool and one hook under its name,
// so unload order can be provoked across every additive registration.
func contributor(name string, began *int) Plugin {
	return funcPlugin{name: name, fn: func(r *Registry) error {
		r.AddExtension(countingExtension{name: name + "_ext", began: began})
		r.AddHooks(PriorityDefault, Hooks{Before: []BeforeToolCall{func(context.Context, ToolCall) Decision { return Allowed() }}})
		return r.AddTools(&echoTool{name: name + "_tool"})
	}}
}

// Unload used to restore a snapshot taken when the plugin registered, so
// unloading an EARLIER plugin also erased every later plugin's tools and
// extensions, and a hook added after an unload could reuse a live hook's id.
func TestRegistry_UnloadIsOrderIndependent(t *testing.T) {
	began := 0
	r, err := BuildRegistry(contributor("a", &began), contributor("b", &began))
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	if err := r.Unload("a"); err != nil { // first in, first out: not LIFO
		t.Fatalf("Unload: %v", err)
	}
	if got := strings.Join(r.tools.Names(), ","); got != "b_tool" {
		t.Fatalf("tools after unloading a = %q, want b_tool", got)
	}
	if got := strings.Join(extensionNames(values(r.extensions)), ","); got != "b_ext" {
		t.Fatalf("extensions after unloading a = %q, want b_ext", got)
	}
	if n := len(r.mergedHooks().Before); n != 1 {
		t.Fatalf("hooks after unloading a = %d, want 1", n)
	}
	// A plugin registered after the unload must not share an id with b's hook.
	if err := r.register(contributor("c", &began)); err != nil {
		t.Fatalf("register c: %v", err)
	}
	if err := r.Unload("c"); err != nil {
		t.Fatalf("Unload c: %v", err)
	}
	if n := len(r.mergedHooks().Before); n != 1 {
		t.Fatalf("unloading c removed b's hook: %d hooks left", n)
	}
	if got := strings.Join(r.tools.Names(), ","); got != "b_tool" {
		t.Fatalf("tools after unloading c = %q, want b_tool", got)
	}
}

func TestRegistry_ToolNameCollisionIsAnError(t *testing.T) {
	_, err := BuildRegistry(
		funcPlugin{name: "first", fn: func(r *Registry) error { return r.AddTools(&echoTool{name: "echo"}) }},
		funcPlugin{name: "second", fn: func(r *Registry) error { return r.AddTools(&echoTool{name: "echo"}) }},
	)
	if err == nil || !strings.Contains(err.Error(), `"first"`) || !strings.Contains(err.Error(), `"echo"`) {
		t.Fatalf("a second plugin silently replaced a tool: err = %v", err)
	}
	if _, err := BuildRegistry(funcPlugin{name: "dup", fn: func(r *Registry) error {
		return r.AddTools(&echoTool{name: "x"}, &echoTool{name: "x"})
	}}); err == nil {
		t.Fatal("one registration naming a tool twice must fail")
	}
}

// An extension tool that shadows a composed tool is a configuration error at
// run start, never a silent swap of which code runs under that name.
func TestExtensionToolCollisionFailsTheRun(t *testing.T) {
	began := 0
	a, err := Build(modelOf("m"), countingExtension{name: "ext", began: &began},
		funcPlugin{name: "tools", fn: func(r *Registry) error { return r.AddTools(&echoTool{name: "ext_tool"}) }})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if _, err := a.OpenTools(context.Background()); err == nil || !strings.Contains(err.Error(), "ext_tool") {
		t.Fatalf("colliding extension tool was accepted: %v", err)
	}
}

func TestRegistry_UnloadUnknownPluginErrors(t *testing.T) {
	r := newRegistry()
	if err := r.Unload("nope"); err == nil {
		t.Fatal("unloading a plugin that was never loaded must fail loudly")
	}
}

// --- ordering -------------------------------------------------------------

func TestRegistry_HooksRunInPriorityOrderNotListOrder(t *testing.T) {
	var log []string
	// Registered late → default → gate: the reverse of the order they must run
	// in. If ordering came from the plugin list, this fails.
	r, err := BuildRegistry(
		hookPlugin{name: "late", priority: PriorityLate, log: &log},
		hookPlugin{name: "consumer", priority: PriorityDefault, log: &log},
		hookPlugin{name: "gate", priority: PriorityGate, log: &log},
	)
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	for _, h := range r.mergedHooks().Before {
		h(context.Background(), ToolCall{Name: "x"})
	}
	if got := strings.Join(log, ","); got != "gate,consumer,late" {
		t.Fatalf("hooks ran in %q, want gate,consumer,late", got)
	}
}

func TestRegistry_TiesBreakByRegistrationOrder(t *testing.T) {
	var log []string
	r, err := BuildRegistry(
		hookPlugin{name: "first", priority: PriorityDefault, log: &log},
		hookPlugin{name: "second", priority: PriorityDefault, log: &log},
	)
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	for _, h := range r.mergedHooks().Before {
		h(context.Background(), ToolCall{Name: "x"})
	}
	if strings.Join(log, ",") != "first,second" {
		t.Fatalf("equal priorities must be stable in registration order, got %v", log)
	}
}

// TestPolicyGate_RunsBeforeConsumerHooks: a consumer hook registered EARLIER
// in the list, even at PriorityGate, still cannot see or approve a call the
// policy refuses, because the policy is part of the executor.
func TestPolicyGate_RunsBeforeConsumerHooks(t *testing.T) {
	var saw []string
	a, err := Build(modelOf("m"),
		funcPlugin{name: "consumer", fn: func(r *Registry) error { // listed FIRST
			r.AddHooks(PriorityGate, Hooks{Before: []BeforeToolCall{
				func(ctx context.Context, call ToolCall) Decision {
					saw = append(saw, call.Name)
					return Allowed()
				},
			}})
			return nil
		}},
		PolicyAllowList("ok"),
	)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if d, _ := a.gate(context.Background(), ToolCall{Name: "forbidden"}); d.Allow {
		t.Fatal("the policy did not refuse an unlisted tool")
	}
	if d, _ := a.gate(context.Background(), ToolCall{Name: "ok"}); !d.Allow {
		t.Fatalf("an allowed call was refused: %+v", d)
	}
	if strings.Join(saw, ",") != "ok" {
		t.Fatalf("consumer hook saw %v, want only the allowed call", saw)
	}
}

// A composition with no policy plugin still enforces the default-deny policy
// at execution, not only in the advertised tool list.
func TestNoPolicyPluginStillDeniesExecution(t *testing.T) {
	ran := false
	a, err := Build(modelOf("m"), funcPlugin{name: "tools", fn: func(r *Registry) error {
		return r.AddTools(funcTool{name: "danger", run: func(context.Context, string) (string, error) { ran = true; return "", nil }})
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	host, err := a.OpenTools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	_, audit, err := host.Execute(context.Background(), nativeHostCall("danger", `{}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if ran || audit.Trace.Allowed {
		t.Fatalf("an unadvertised tool ran without a policy: ran=%v trace=%+v", ran, audit.Trace)
	}
}

func TestRegistry_HookThrowIsNotDowngraded(t *testing.T) {
	r, err := BuildRegistry(
		funcPlugin{name: "strict", fn: func(r *Registry) error {
			r.AddHooks(PriorityDefault, Hooks{ErrorPolicy: HookThrow})
			return nil
		}},
		funcPlugin{name: "relaxed", fn: func(r *Registry) error {
			r.AddHooks(PriorityDefault, Hooks{ErrorPolicy: HookContinue})
			return nil
		}},
	)
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	if r.mergedHooks().ErrorPolicy != HookThrow {
		t.Fatal("a plugin that treats hook failure as fatal must not be silently downgraded")
	}
}

// --- Config path ----------------------------------------------------------

func TestApplyConfig_LeavesUnsetSeamsFree(t *testing.T) {
	// A Config that says nothing about the goal must not claim the goal seam,
	// or an extra plugin could never contribute one.
	r, err := BuildRegistry(configPlugin{cfg: Config{Provider: &FauxProvider{}, Model: "m"}})
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	if r.Provider("goal") != "" {
		t.Fatal("an unset Config field must leave its seam unclaimed")
	}
	if err := r.register(funcPlugin{name: "extra", fn: func(r *Registry) error { return r.SetGoal("added later") }}); err != nil {
		t.Fatalf("a plugin must be able to fill a seam Config left free: %v", err)
	}
	if r.goal != "added later" {
		t.Fatalf("goal = %q", r.goal)
	}
}

// --- diagnostics ----------------------------------------------------------

func TestRegistry_DescribeNamesSeamOwners(t *testing.T) {
	r, err := BuildRegistry(
		modelOf("m"),
		PolicyAllowList("echo"),
		funcPlugin{name: "tools", fn: func(r *Registry) error { return r.AddTools(&echoTool{name: "echo"}) }},
	)
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	out := r.Describe()
	for _, want := range []string{"plugins:", "seam model", "seam policy", "tools: echo"} {
		if !strings.Contains(out, want) {
			t.Fatalf("Describe() missing %q:\n%s", want, out)
		}
	}
}

func TestAgentDescribe_ReportsPresenceNotSecrets(t *testing.T) {
	a, err := New(Config{
		Provider: &FauxProvider{}, Model: "m",
		Session: NewMemorySessionStore(), SessionID: "s1",
		PromptCacheKey: "ck",
		RefreshKey:     func(context.Context, string) (string, error) { return "super-secret-token", nil },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	out := a.Describe()
	if strings.Contains(out, "super-secret-token") {
		t.Fatal("Describe must never render a credential")
	}
	for _, want := range []string{"refresh_key:", "session:", "policy:"} {
		if !strings.Contains(out, want) {
			t.Fatalf("Describe() missing %q:\n%s", want, out)
		}
	}
}

// --- ejection -------------------------------------------------------------

// countingExtension records whether the loop ever asked it to begin a run.
type countingExtension struct {
	name  string
	began *int
}

func (e countingExtension) Name() string { return e.name }

func (e countingExtension) BeginRun(context.Context, RunInfo) (Extension, error) {
	*e.began++
	return e, nil
}

func (e countingExtension) Tools() []Tool { return []Tool{&echoTool{name: "ext_tool"}} }

func (e countingExtension) Register(r *Registry) error {
	r.AddExtension(e)
	return nil
}

// TestUnloadEjectsAnExtensionCompletely is the "plug and eject" property at the
// registry level: after Unload the extension is gone from the composition, and
// an agent built afterwards never calls it — not even once, which is the part a
// name-check alone would miss.
func TestUnloadEjectsAnExtensionCompletely(t *testing.T) {
	began := 0
	ext := countingExtension{name: "ejectable", began: &began}

	r, err := BuildRegistry(modelOf("m"), ext)
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	if !strings.Contains(r.Describe(), "ejectable") {
		t.Fatalf("extension did not register:\n%s", r.Describe())
	}

	if err := r.Unload("ejectable"); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	if strings.Contains(r.Describe(), "ejectable") {
		t.Fatalf("extension survived its own removal:\n%s", r.Describe())
	}

	agent, err := r.Agent()
	if err != nil {
		t.Fatalf("Agent: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if began != 0 {
		t.Fatalf("an unloaded extension was still driven %d time(s)", began)
	}
}

// TestExtensionToolIsRunScopedNotComposeScoped pins where a contributed tool
// comes from: the run, not the registry. Registering an extension must not
// mutate the shared base ToolSet, or two agents built from one registry would
// see each other's tools.
func TestExtensionToolIsRunScopedNotComposeScoped(t *testing.T) {
	began := 0
	r, err := BuildRegistry(modelOf("m"), countingExtension{name: "ext", began: &began})
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	if strings.Contains(r.Describe(), "ext_tool") {
		t.Fatalf("an extension tool leaked into the composed tool set:\n%s", r.Describe())
	}
	agent, err := r.Agent()
	if err != nil {
		t.Fatalf("Agent: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if began != 1 {
		t.Fatalf("extension began %d times, want exactly 1 per run", began)
	}
}

// TestDefaultLimitsCannotStarveTheirOwnTurns guards the pairing between the two
// ceilings, not their absolute values. A turn that calls a tool spends one of
// each, so MaxToolCalls below MaxTurns makes the tool ceiling the real turn
// ceiling — the run dies of a limit nobody was tuning, and the reason reported
// is the wrong one.
func TestDefaultLimitsCannotStarveTheirOwnTurns(t *testing.T) {
	d := DefaultLimits()
	if d.MaxToolCalls < d.MaxTurns {
		t.Fatalf("MaxToolCalls (%d) < MaxTurns (%d): every turn may call a tool, so the tool ceiling would bind first", d.MaxToolCalls, d.MaxTurns)
	}
}

// TestDefaultLimitsClearTheMeasuredFloor pins the finding that motivated the
// raise: a data agent answering one real question spent more than 12 turns on
// schema discovery, exploratory queries and a correction before writing the
// answer, and failed 2 of 3 first runs. Anything back at or below that floor is
// a regression to a measured-broken configuration, whatever the reasoning.
func TestDefaultLimitsClearTheMeasuredFloor(t *testing.T) {
	const failedAtTurns = 12
	for _, tc := range []struct {
		name string
		got  int
		min  int
	}{
		{"turns", DefaultLimits().MaxTurns, failedAtTurns * 2},
		{"tool calls", DefaultLimits().MaxToolCalls, failedAtTurns * 2},
	} {
		if tc.got < tc.min {
			t.Errorf("default max %s = %d, want >= %d (12 turns was measured insufficient for a data agent)", tc.name, tc.got, tc.min)
		}
	}
}

// TestDefaultLimitsLeaveTheContextKnobsAlone: the turn raise was not licence to
// move the byte/token budgets with it — those are sized by the model's window
// and the transcript, not by how many steps an answer takes.
func TestDefaultLimitsLeaveTheContextKnobsAlone(t *testing.T) {
	d := DefaultLimits()
	if d.MaxToolResultLen != defaultMaxToolResultBytes {
		t.Errorf("MaxToolResultLen = %d, want the package default %d", d.MaxToolResultLen, defaultMaxToolResultBytes)
	}
	if d.MaxContextTokens != defaultContextTokenBudget {
		t.Errorf("MaxContextTokens = %d, want the package default %d", d.MaxContextTokens, defaultContextTokenBudget)
	}
}
