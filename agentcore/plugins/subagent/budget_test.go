package subagent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/2found/2ai/ai"
	"strings"
	"sync"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/subagent"
)

// The spawn budget has to fit the run it is in.
//
// It used to be a fixed 8, calibrated against DefaultLimits (12 turns, 24 tool
// calls) where 8 children is already a third of everything the run may do. On a
// long autonomous run — thousands of turns, thousands of tool calls — that same
// 8 is not a budget but a cliff: the ninth spawn and every one after it comes
// back "sub-agent budget exhausted — finish the remaining work yourself", and
// the agent spends the remaining thousands of turns doing in its own context
// exactly the noisy work delegation exists to keep out of it. Nothing fails
// loudly; the run just gets worse.
//
// So the default is now a SHARE of the run's tool budget. These tests pin both
// ends of that: the share still yields the historical 8 at DefaultLimits, and
// it grows with a longer run.

// budgetProbe scripts a parent that attempts a fixed number of spawns and then
// finishes, and answers every child immediately. Routing is by content: a child
// run is seeded with the task text, so a request whose first non-system message
// is a child task belongs to a child.
type budgetProbe struct {
	mu       sync.Mutex
	attempts int // spawns the parent will try before finishing
	served   int // parent turns served
}

func (p *budgetProbe) Stream(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
	message := nativeAnswer(childAnswer)
	if agentcore.DelegationDepth(ctx) == 0 {
		p.mu.Lock()
		p.served++
		if p.served > p.attempts {
			message = nativeAnswer("parent done")
		} else {
			message = AssistantToolCall(fmt.Sprintf("s%d", p.served), subagent.ToolSpawnSubagent, fmt.Sprintf(`{"task":%q}`, fmt.Sprintf("%s %d", childTaskPrefix, p.served)))
		}
		p.mu.Unlock()
	}
	return ai.ScriptedStream(message)(ctx, model, view, options)
}

const (
	childTaskPrefix = "child task"
	childAnswer     = "child done"
)

// spawnOutcomes runs one agent that attempts `attempts` spawns under the given
// limits and plugin settings, and reports how many children actually ran versus
// how many were refused by the budget.
func spawnOutcomes(t *testing.T, limits agentcore.Limits, settings subagent.Plugin, attempts int) (ran, blocked int) {
	t.Helper()
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: nativeProvider((&budgetProbe{attempts: attempts}).Stream),
		Model:          "faux-1",
		Tools:          agentcore.NewToolSet(&echoTool{name: "echo"}),
		Policy:         agentcore.NewAllowList("echo", subagent.ToolSpawnSubagent),
		Limits:         &limits,
		Extensions:     []agentcore.ExtensionFactory{settings},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "fan out")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	// Classify strictly. A spawn that failed for any OTHER reason — a provider
	// error, the loop's repeated-failure circuit breaker — must not be scored as
	// a child that ran, or a broken probe reads as a generous budget.
	for _, m := range res.Messages {
		if m.Role != agentcore.RoleTool || m.Name != subagent.ToolSpawnSubagent {
			continue
		}
		switch {
		case m.Content == childAnswer:
			ran++
		case strings.Contains(m.Content, "budget exhausted"):
			blocked++
		default:
			t.Fatalf("spawn failed for an unrelated reason: %q", m.Content)
		}
	}
	if ran+blocked != attempts {
		t.Fatalf("script did not complete: %d spawn results for %d attempts (stop=%q)",
			ran+blocked, attempts, res.StopReason)
	}
	return ran, blocked
}

// TestSpawnBudget covers both ends of the derived default and the escape hatch.
//
// Each case attempts exactly one spawn past its budget. That is deliberate: the
// loop disables a tool that keeps failing, so a script that kept asking after
// the budget ran out would collect circuit-breaker refusals rather than budget
// refusals, and the counts would stop meaning what the case names claim.
func TestSpawnBudget(t *testing.T) {
	cases := []struct {
		name         string
		maxToolCalls int
		settings     subagent.Plugin
		want         int
		why          string
	}{{
		// The default end: whatever the library's tool budget currently is, a
		// third of it is what delegation gets. Both sides are computed, because
		// pinning the number here froze the delegation budget at the tool budget
		// of the day the case was written — raising DefaultLimits then failed a
		// test about subagents.
		name:         "delegation tracks the default tool budget",
		maxToolCalls: agentcore.DefaultLimits().MaxToolCalls,
		want:         max(agentcore.DefaultLimits().MaxToolCalls/3, 8),
		why:          "a third of the default tool budget, floored at 8",
	}, {
		// The reason for the change. Before, this run delegated 8 tasks and did
		// the other 42 inline, in its own context.
		name:         "scales with a long run",
		maxToolCalls: 150,
		want:         50,
		why:          "a third of the run's tool budget",
	}, {
		// The other direction: a deliberately tool-starved run must not lose
		// delegation entirely, which a bare share would do (12/3 = 4, floored).
		name:         "floor for a tool-starved run",
		maxToolCalls: 12,
		want:         8,
		why:          "the floor, since the share would yield 4",
	}, {
		// The escape hatch. Without this, the derived default could quietly
		// override an explicit cap set for cost reasons.
		name:         "explicit setting wins",
		maxToolCalls: 600, // would derive 200
		settings:     subagent.Plugin{MaxPerRun: 3},
		want:         3,
		why:          "the number the consumer named",
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			limits := agentcore.DefaultLimits()
			limits.MaxTurns = 4 * (tc.want + 2)
			limits.MaxToolCalls = tc.maxToolCalls

			ran, blocked := spawnOutcomes(t, limits, tc.settings, tc.want+1)
			if ran != tc.want {
				t.Fatalf("spawned %d children, want %d (%s)", ran, tc.want, tc.why)
			}
			if blocked != 1 {
				t.Fatalf("the budget must still bite one past it: %d blocked, want 1", blocked)
			}
		})
	}
}
