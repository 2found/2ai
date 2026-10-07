package goal_test

// The agent revising its own objective.
//
// A long autonomous run discovers things, and some of what it discovers is that
// the requirement it was given was wrong — impossible as stated, or resting on
// an assumption the work disproved. Before update_goal, an agent in that
// position had two honest moves: grind against the condition until MaxTurns, or
// declare BLOCKED and hand back nothing. Both waste the run.
//
// These tests hold the four things that have to be true for the tool to be worth
// the authority it transfers: the gate must actually enforce the NEW condition,
// the model must be told the new contract rather than left reading the old one,
// the change must reach the durable log so a resume comes back on the current
// objective, and every revision must carry the model's reason so a narrowing is
// reviewable afterwards.

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/2found/2ai/ai"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/goal"
)

// --- harness ------------------------------------------------------------------

type scriptProvider struct {
	mu      sync.Mutex
	turns   int
	systems []string   // the system message shown on each turn
	tools   [][]string // the tool names offered to the model on each turn
	script  func(turn int) ai.Message
}

// offered reports whether the model was ever shown a schema by this name. It is
// the only honest answer to "is the tool available": what the agent was built
// with is configuration, what reaches the provider is the contract.
func (p *scriptProvider) offered(name string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, turn := range p.tools {
		for _, n := range turn {
			if n == name {
				return true
			}
		}
	}
	return false
}

func (p *scriptProvider) Stream(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
	p.mu.Lock()
	p.turns++
	p.systems = append(p.systems, ai.GetCurrentSystemPrompt(view.Messages()))
	var names []string
	for _, tool := range ai.GetCurrentTools(view.Messages()) {
		names = append(names, tool.Name)
	}
	p.tools = append(p.tools, names)
	message := p.script(p.turns)
	p.mu.Unlock()
	return ai.ScriptedStream(message)(ctx, model, view, options)
}

const (
	origGoal  = "Every shard in the ledger corpus audited and one report filed per region"
	newGoal   = "Every shard in the ledger corpus audited, with CLEARING-7742 escalated rather than reconciled"
	theReason = "CLEARING-7742 spans three corpora, so it cannot be reconciled from the ledger alone"
)

// reviseCall is the tool call the scripted model makes to change the objective.
func reviseCall(id string) ai.Message {
	return nativeCall(id, goal.ToolName,
		fmt.Sprintf(`{"goal":%q,"reason":%q}`, newGoal, theReason))
}

// --- the tests ------------------------------------------------------------------

// The load-bearing one: after a revision, the gate must hold the run to the NEW
// condition and the model must be reading it. A gate still enforcing the old
// condition while the prompt states the new one — or the reverse — is worse than
// no revision at all, because the run is then held to a contract nothing tells
// it about.
func TestRevisedGoalIsWhatTheModelReadsAndTheGateEnforces(t *testing.T) {
	prov := &scriptProvider{}
	prov.script = func(turn int) ai.Message {
		switch turn {
		case 1:
			return reviseCall("r1")
		case 2:
			// Finishing with no sentinel: the gate must still be armed.
			return nativeAnswer("Escalated CLEARING-7742.")
		default:
			return nativeAnswer("Escalated CLEARING-7742.\n" + goal.Done)
		}
	}

	plugin, trail := goal.UntilRevisable(origGoal)
	limits := agentcore.DefaultLimits()
	limits.MaxTurns = 10
	limits.MaxToolCalls = 10

	agent, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{
		NativeProvider: nativeProvider(prov.Stream), Model: "test",
		Tools:  agentcore.NewToolSet(),
		Policy: agentcore.NewAllowList(goal.ToolName),
		Limits: &limits,
	}), plugin)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "Audit the ledger corpus.")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	prov.mu.Lock()
	systems := prov.systems
	prov.mu.Unlock()

	if len(systems) < 3 {
		t.Fatalf("want at least 3 turns, got %d — the gate never re-opened the sentinel-free finish", len(systems))
	}

	// Turn 1 read the original contract; the revision had not happened yet.
	if !strings.Contains(systems[0], origGoal) {
		t.Fatalf("the first turn did not state the original condition:\n%s", systems[0])
	}

	// Every turn after the revision reads the NEW contract, and only it. Both
	// halves matter: a prompt that gained the new condition while keeping the old
	// one hands the model a contradiction to resolve on its own.
	for i, sys := range systems[1:] {
		if !strings.Contains(sys, newGoal) {
			t.Fatalf("turn %d still states the superseded condition — the system prompt was not "+
				"rebuilt, so the model is working to a contract the gate no longer enforces:\n%s", i+2, sys)
		}
		if strings.Contains(sys, origGoal) {
			t.Fatalf("turn %d states BOTH conditions: the revision was appended rather than "+
				"replacing, leaving the model to guess which one counts:\n%s", i+2, sys)
		}
	}

	// The gate was still armed after the revision: the sentinel-free finish on
	// turn 2 was rejected and the run continued.
	if res.StopReason == goal.StopReasonStalled {
		t.Fatalf("run stalled: %q", res.Final)
	}
	if !strings.Contains(res.Final, goal.Done) {
		t.Fatalf("run ended without the sentinel: %q", res.Final)
	}

	// The trail is the accountability half, and it must carry the model's own
	// justification — "the goal changed" is not reviewable, "the goal changed
	// because X" is.
	revs := trail.Revisions()
	if len(revs) != 2 {
		t.Fatalf("want the original plus one revision in the trail, got %d: %+v", len(revs), revs)
	}
	if revs[0].Goal != origGoal || revs[1].Goal != newGoal {
		t.Fatalf("the trail does not record the run's conditions in order: %+v", revs)
	}
	if revs[1].Reason != theReason {
		t.Fatalf("the revision was recorded without the model's reason, so a narrowing is not "+
			"reviewable afterwards: %q", revs[1].Reason)
	}
}

// The checkpoint retains both the current condition and the revision's reason
// and previous condition. Resuming it must not create a duplicate revision.
func TestRevisedGoalIsDurable(t *testing.T) {
	makeAgent := func(stream ai.StreamFn) *agentcore.Agent {
		t.Helper()
		plugin, _ := goal.UntilRevisable(origGoal)
		a, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{NativeProvider: nativeProvider(stream), Model: "test", Policy: agentcore.NewAllowList(goal.ToolName)}), plugin)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	result, err := makeAgent(ai.ScriptedStream(reviseCall("r1"), nativeAnswer("done\n"+goal.Done))).Prompt(context.Background(), "Audit the ledger corpus.")
	if err != nil {
		t.Fatal(err)
	}
	var saved struct {
		Goal          string
		GoalRevisions []agentcore.GoalRevision
	}
	if err := json.Unmarshal(result.NativeState, &saved); err != nil {
		t.Fatal(err)
	}
	want := []agentcore.GoalRevision{{Previous: origGoal, Goal: newGoal, Reason: theReason}}
	if saved.Goal != newGoal || !reflect.DeepEqual(saved.GoalRevisions, want) {
		t.Fatalf("revision not checkpointed: %+v", saved)
	}
	resumed := makeAgent(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		prompt := ai.GetCurrentSystemPrompt(view.Messages())
		if !strings.Contains(prompt, newGoal) || strings.Contains(prompt, origGoal) {
			t.Errorf("resumed stale condition: %s", prompt)
		}
		return ai.ScriptedStream(nativeAnswer("still done\n"+goal.Done))(ctx, model, view, options)
	})
	next, err := resumed.RunNative(context.Background(), agentcore.NativeRun{State: result.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "continue"}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(next.NativeState, &saved); err != nil {
		t.Fatal(err)
	}
	if saved.Goal != newGoal || !reflect.DeepEqual(saved.GoalRevisions, want) {
		t.Fatalf("resume added a revision: %+v", saved)
	}
}

// Recovering a revised goal is not revising it again.
//
// A resumed run is built from a fresh plugin holding the ORIGINAL condition, and
// the loop hands it the log's current one instead. Bringing the store up to date
// is right; routing that through Update is not, because Update marks the
// condition pending and pending is what the loop drains. The first turn back
// would then write an EntryGoal the log already holds and emit "goal updated:
// <the same goal>" — telling a watching human the agent moved the finish line
// when all that happened was a restart. The log is the audit trail for exactly
// the move this tool makes possible, so a spurious entry in it is not cosmetic.
func TestResumingOnARevisedGoalDoesNotRecordAnotherRevision(t *testing.T) {
	// The resume composition the plugin documents: no configured Goal, so the
	// loop's recovered condition is the one in force, and a caller-supplied Store
	// that still holds what the run STARTED with — this process was not around
	// when the agent revised it.
	st := goal.NewStore(origGoal)
	p := goal.Plugin{Revisable: true, Store: st}

	ext, err := p.BeginRun(context.Background(), agentcore.RunInfo{Goal: newGoal})
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	reviser, ok := ext.(interface{ ReviseGoal() (string, bool) })
	if !ok {
		t.Fatalf("the gate no longer implements GoalReviser (%T); the loop would never "+
			"drain a revision at all", ext)
	}

	if got := st.Goal(); got != newGoal {
		t.Fatalf("the resumed gate is enforcing %q, not the condition the log recovered (%q)", got, newGoal)
	}
	if g, pending := reviser.ReviseGoal(); pending {
		t.Fatalf("the resume left %q marked as a pending revision, so the first turn back "+
			"would append an EntryGoal the log already holds and emit \"goal updated\" "+
			"to the viewer for a change that never happened", g)
	}

	// The adoption is still in the trail — recovering is worth recording, it just
	// is not a revision — and a genuine revision after it still drains.
	if n := len(st.Revisions()); n != 2 {
		t.Fatalf("want the original plus the recovered condition in the trail, got %d", n)
	}
	if !st.Update("Every shard audited and the CLEARING-7742 escalation acknowledged", "the escalation bounced") {
		t.Fatalf("a real revision after a resume was rejected")
	}
	if _, pending := reviser.ReviseGoal(); !pending {
		t.Fatalf("a real revision after a resume never reached the loop, so it would never " +
			"be recorded or enforced")
	}
}

// The tool is authority, so it is not handed out by default. A plain Until() run
// must not advertise it — both because the composition did not ask for it, and
// because a schema on every turn is a standing token cost.
func TestTheToolIsOptIn(t *testing.T) {
	for _, tc := range []struct {
		name   string
		plugin goal.Plugin
		want   bool
	}{
		{"Until is not revisable", goal.Until(origGoal), false},
		{"UntilRevisable is", func() goal.Plugin { p, _ := goal.UntilRevisable(origGoal); return p }(), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prov := &scriptProvider{}
			prov.script = func(int) ai.Message {
				return nativeAnswer("done\n" + goal.Done)
			}
			limits := agentcore.DefaultLimits()
			agent, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{
				NativeProvider: nativeProvider(prov.Stream), Model: "test",
				Tools:  agentcore.NewToolSet(),
				Policy: agentcore.NewAllowList(goal.ToolName),
				Limits: &limits,
			}), tc.plugin)
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			if _, err := agent.Prompt(context.Background(), "go"); err != nil {
				t.Fatalf("Prompt: %v", err)
			}
			if got := prov.offered(goal.ToolName); got != tc.want {
				t.Fatalf("update_goal reached the provider = %v, want %v (schemas seen: %v)",
					got, tc.want, prov.tools)
			}
		})
	}
}

// A reason is the whole audit trail, so the tool refuses without one rather than
// recording an unexplained narrowing.
func TestRevisionRequiresAReason(t *testing.T) {
	prov := &scriptProvider{}
	prov.script = func(turn int) ai.Message {
		if turn == 1 {
			return nativeCall("r1", goal.ToolName,
				fmt.Sprintf(`{"goal":%q}`, newGoal))
		}
		return nativeAnswer("done\n" + goal.Done)
	}

	plugin, trail := goal.UntilRevisable(origGoal)
	limits := agentcore.DefaultLimits()
	agent, err := agentcore.Build(agentcore.ConfigPlugin(agentcore.Config{
		NativeProvider: nativeProvider(prov.Stream), Model: "test",
		Tools:  agentcore.NewToolSet(),
		Policy: agentcore.NewAllowList(goal.ToolName),
		Limits: &limits,
	}), plugin)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got := trail.Goal(); got != origGoal {
		t.Fatalf("a reasonless revision was accepted: the condition is now %q", got)
	}
	if revs := trail.Revisions(); len(revs) != 1 {
		t.Fatalf("a reasonless revision reached the trail: %+v", revs)
	}
}

// Restating the same condition is not a change. Letting it through would write
// an EntryGoal and rebuild the system prompt — invalidating the provider's cache
// for the whole prefix — in exchange for nothing, once per attempt, on a model
// prone to repeating itself.
func TestRestatingTheSameGoalChangesNothing(t *testing.T) {
	st := goal.NewStore(origGoal)
	if st.Update(origGoal, "no real change") {
		t.Fatal("an identical restatement was accepted as a revision")
	}
	if revs := st.Revisions(); len(revs) != 1 {
		t.Fatalf("the no-op restatement was recorded: %+v", revs)
	}
	if !st.Update(newGoal, theReason) {
		t.Fatal("a real revision was rejected")
	}
}
