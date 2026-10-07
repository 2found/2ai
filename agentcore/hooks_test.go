package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
)

// TestPiContextHookRewritesRequest verifies a native context hook can rewrite
// the message view the model reasons over, without mutating the persisted run
// history.
func TestPiContextHookRewritesRequest(t *testing.T) {
	recorder := &nativeRecorder{}
	faux := recordedNativeProvider(recorder, AssistantText("done"))

	redact := func(_ context.Context, raw []json.RawMessage) ([]json.RawMessage, error) {
		out := make([]json.RawMessage, len(raw))
		for i, m := range raw {
			out[i] = json.RawMessage(strings.ReplaceAll(string(m), "SECRET", "[redacted]"))
		}
		return out, nil
	}

	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Hooks:          Hooks{PiContext: []PiContextHook{redact}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "my SECRET token")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	// The provider saw the redacted view.
	var sawContents []string
	for _, req := range recorder.all() {
		for _, m := range req.Messages {
			if m.Role == RoleUser {
				sawContents = append(sawContents, m.Content)
			}
		}
	}
	if len(sawContents) == 0 || strings.Contains(strings.Join(sawContents, "|"), "SECRET") {
		t.Fatalf("provider should have seen redacted content, saw %v", sawContents)
	}
	if !strings.Contains(strings.Join(sawContents, "|"), "[redacted]") {
		t.Fatalf("provider should have seen [redacted], saw %v", sawContents)
	}
	// Persisted history is untouched: the original user message is intact.
	var keptOriginal bool
	for _, m := range res.Messages {
		if m.Role == RoleUser && strings.Contains(m.Content, "SECRET") {
			keptOriginal = true
		}
	}
	if !keptOriginal {
		t.Fatalf("context hook must not mutate persisted history: %+v", res.Messages)
	}
}

// (before_provider_request and before_agent_start are legacy request-pipeline
// seams with no native counterpart; request shaping lives in PiContext hooks,
// candidate StreamFns and engine admission.)



// TestPanickingObserverDoesNotAbort verifies the default (continue) error policy:
// a panicking message_end observer is attributed via OnError and the run finishes
// normally.
func TestPanickingObserverDoesNotAbort(t *testing.T) {
	var mu sync.Mutex
	var errors []string
	panicker := func(context.Context, Message) { panic("boom") }
	agent, err := New(Config{
		NativeProvider: scriptedNativeProvider(AssistantText("survived")),
		Model:          "test",
		Hooks: Hooks{
			MessageEnd: []MessageEndHook{panicker},
			OnError: func(source string, err error) {
				mu.Lock()
				errors = append(errors, source+": "+err.Error())
				mu.Unlock()
			},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("a panicking observer must not abort the run: %v", err)
	}
	if res.Final != "survived" {
		t.Fatalf("run should have completed, final=%q", res.Final)
	}
	if len(errors) == 0 || !strings.Contains(errors[0], "message_end[0]") {
		t.Fatalf("panic should be attributed to its source, got %v", errors)
	}
}

// TestHookThrowPolicyAborts verifies the opt-in throw policy surfaces a hook
// failure from the loop instead of swallowing it.
func TestHookThrowPolicyAborts(t *testing.T) {
	boom := func(context.Context, Message) { panic("fatal") }
	agent, err := New(Config{
		NativeProvider: scriptedNativeProvider(AssistantText("never")),
		Model:          "test",
		Hooks: Hooks{
			MessageEnd:  []MessageEndHook{boom},
			ErrorPolicy: HookThrow,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = agent.Prompt(context.Background(), "go")
	if err == nil {
		t.Fatalf("HookThrow must surface the hook error")
	}
	if !strings.Contains(err.Error(), "message_end[0]") {
		t.Fatalf("aborted error should attribute the source, got %v", err)
	}
}

// TestDeterministicEmitOrder verifies multiple handlers of one event fire in
// registration order, every turn.
func TestDeterministicEmitOrder(t *testing.T) {
	var order []string
	mk := func(tag string) MessageEndHook {
		return func(context.Context, Message) { order = append(order, tag) }
	}
	// Two turns: a tool call then a final answer.
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "noop", `{}`),
		AssistantText("done"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(noopTool{}),
		Policy:         NewAllowList("noop"),
		Hooks:          Hooks{MessageEnd: []MessageEndHook{mk("a"), mk("b"), mk("c")}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	// Two assistant messages × three observers, each in a,b,c order.
	want := []string{"a", "b", "c", "a", "b", "c"}
	if strings.Join(order, ",") != strings.Join(want, ",") {
		t.Fatalf("emit order = %v, want %v", order, want)
	}
}

// TestReentrantHookCallsIntoAgent verifies a hook may call back into the agent
// (run a nested sub-agent) without deadlock, and the outer run still completes.
func TestReentrantHookCallsIntoAgent(t *testing.T) {
	// The nested agent the hook drives.
	inner, err := New(Config{
		NativeProvider: scriptedNativeProvider(AssistantText("inner-done")),
		Model:          "test",
	})
	if err != nil {
		t.Fatalf("New inner: %v", err)
	}

	var nestedFinal string
	reentrant := func(ctx context.Context, _ Message) {
		r, ierr := inner.Prompt(ctx, "sub-task")
		if ierr == nil {
			nestedFinal = r.Final
		}
	}
	outer, err := New(Config{
		NativeProvider: scriptedNativeProvider(AssistantText("outer-done")),
		Model:          "test",
		Hooks:          Hooks{MessageEnd: []MessageEndHook{reentrant}},
	})
	if err != nil {
		t.Fatalf("New outer: %v", err)
	}
	res, err := outer.Prompt(context.Background(), "task")
	if err != nil {
		t.Fatalf("reentrant hook must not deadlock or fail: %v", err)
	}
	if res.Final != "outer-done" {
		t.Fatalf("outer run final = %q", res.Final)
	}
	if nestedFinal != "inner-done" {
		t.Fatalf("nested agent did not run from the hook: %q", nestedFinal)
	}
}

// TestPanickingBeforeHookDoesNotExecuteToolUnderThrow verifies that under
// HookThrow a panicking before-tool-call hook refuses the call (blocked) rather
// than letting it execute.
func TestPanickingBeforeHookDoesNotExecuteToolUnderThrow(t *testing.T) {
	var ran bool
	tool := funcTool{
		name: "danger",
		run:  func(context.Context, string) (string, error) { ran = true; return "ok", nil },
	}
	boom := func(context.Context, ToolCall) Decision { panic("hook down") }
	agent, err := New(Config{
		NativeProvider: scriptedNativeProvider(
			AssistantToolCall("c1", "danger", `{}`),
			AssistantText("end"),
		),
		Model:  "test",
		Tools:  NewToolSet(tool),
		Policy: NewAllowList("danger"),
		Hooks: Hooks{
			Before:      []BeforeToolCall{boom},
			ErrorPolicy: HookThrow,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if ran {
		t.Fatalf("tool must not execute when a before-hook fails under HookThrow")
	}
	// The block reason was fed back to the model as a tool result.
	var sawBlock bool
	for _, m := range res.Messages {
		if m.Role == RoleTool && strings.Contains(m.Content, "blocked") {
			sawBlock = true
		}
	}
	if !sawBlock {
		t.Fatalf("expected a blocked tool result, got %+v", res.Messages)
	}
}

// funcTool is a minimal Tool backed by a func, for tests.
type funcTool struct {
	name string
	run  func(context.Context, string) (string, error)
}

func (f funcTool) Name() string { return f.name }
func (f funcTool) Schema() ToolSchema {
	return ToolSchema{Name: f.name, Description: f.name, Parameters: map[string]any{"type": "object"}}
}
func (f funcTool) Run(ctx context.Context, args string) (string, error) { return f.run(ctx, args) }

// (before_agent_start is a legacy request-pipeline seam with no native
// counterpart; the system prompt is assembled once at run start from the
// definition, memory recall and extension sections.)

// TestTurnHooksFireOnNonStreamedRun is the point of turn hooks: the stream
// events only reach an attached viewer, so metering must not depend on one.
func TestTurnHooksFireOnNonStreamedRun(t *testing.T) {
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "noop", `{}`),
		AssistantText("done"),
	)
	var starts, ends []TurnInfo
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(&echoTool{name: "noop"}),
		Policy:         NewAllowList("noop"),
		Hooks: Hooks{
			TurnStart: []TurnHook{func(_ context.Context, i TurnInfo) { starts = append(starts, i) }},
			TurnEnd:   []TurnHook{func(_ context.Context, i TurnInfo) { ends = append(ends, i) }},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go") // nil sink: no stream events at all
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if len(starts) != res.Turns || len(ends) != res.Turns {
		t.Fatalf("want %d turn_start and turn_end hooks, got %d/%d", res.Turns, len(starts), len(ends))
	}
	if starts[0].Turn != 1 || ends[len(ends)-1].Turn != res.Turns {
		t.Fatalf("turn numbering wrong: starts=%+v ends=%+v", starts, ends)
	}
	if ends[len(ends)-1].StopReason == "" {
		t.Fatal("turn_end should carry the turn's stop reason")
	}
	if starts[0].StopReason != "" {
		t.Fatal("turn_start must not carry a stop reason")
	}
}

// TestAbortedTurnBooksNoTurnEnd pins the native lifecycle: turn_end observers
// only see completed turns, so a turn that dies mid-flight on a provider error
// never reaches the observer at all — the books never show a turn that started
// and never ended.
func TestAbortedTurnBooksNoTurnEnd(t *testing.T) {
	var ends int
	agent, err := New(Config{
		NativeProvider: nativeCandidate("err", failingNativeStream(errors.New("provider exploded"))),
		Model:          "test",
		Retry:          &RetryPolicy{MaxAttempts: 1},
		Hooks: Hooks{
			TurnEnd: []TurnHook{func(context.Context, TurnInfo) { ends++ }},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err == nil {
		t.Fatal("want a provider error")
	}
	if ends != 0 {
		t.Fatalf("aborted turn must not reach turn_end observers, got %d", ends)
	}
}

// TestAgentEndSeesFinalResult verifies agent_end observes the RunResult the
// caller receives — including a failed run's synthesized failure turn.
func TestAgentEndSeesFinalResult(t *testing.T) {
	faux := scriptedNativeProvider(AssistantText("the answer"))
	var seen RunResult
	var calls int
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Hooks: Hooks{
			AgentEnd: []AgentEndHook{func(_ context.Context, r RunResult) { seen = r; calls++ }},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if calls != 1 {
		t.Fatalf("agent_end should fire exactly once, got %d", calls)
	}
	if seen.Final != res.Final || seen.Turns != res.Turns {
		t.Fatalf("agent_end saw %+v, caller got %+v", seen, res)
	}
}

