package agentcore

import (
	"context"
	"strings"
	"testing"
)

type budgetParallelEcho struct{ *echoTool }

func (*budgetParallelEcho) Parallel() bool { return true }

// toolCallWithCost scripts a tool-calling turn that also reports model cost, so a
// test can drive accumulated Usage across turns.
func toolCallWithCost(id, name, args string, cost float64) ChatResponse {
	r := AssistantToolCall(id, name, args)
	r.Usage = Usage{CostUSD: cost}
	return r
}

// TestBudgetGateGracefulStop proves the mid-run ceiling (#4b): a run whose own
// turns push accumulated cost over the cap gets exactly one tool-free wrap-up
// turn and then stops with StopReason "budget_exhausted" — it does not keep
// spending, and the model is told to summarize (tools are stripped for that
// turn).
func TestBudgetGateGracefulStop(t *testing.T) {
	work := &echoTool{name: "do_work"}
	faux := scriptedNativeProvider(
		toolCallWithCost("w1", "do_work", `{"step":1}`, 0.30), // usage after: 0.30
		toolCallWithCost("w2", "do_work", `{"step":2}`, 0.30), // usage after: 0.60 (>= cap)
		// The finalizing turn: a well-behaved model, seeing no tools, writes a
		// wrap-up. (Faux replays blindly; the loop having stripped schemas is what a
		// real provider would honor.)
		AssistantText("Completed step 1 and 2; next I would do step 3."),
		// Guard turn: must never be reached — if the run kept going it would call
		// work a third time and the assertion on work.called would fail.
		toolCallWithCost("w3", "do_work", `{"step":3}`, 0.30),
	)
	limits := DefaultLimits()
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(work),
		Policy:         NewAllowList("do_work"),
		Limits:         &limits,
		BudgetGate: func(_ context.Context, u Usage) bool {
			return u.CostUSD >= 0.50
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "do the work")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.StopReason != "budget_exhausted" {
		t.Fatalf("stop reason = %q, want budget_exhausted (turns=%d final=%q)", res.StopReason, res.Turns, res.Final)
	}
	if work.called != 2 {
		t.Fatalf("work called %d times; run must stop after the cap, not run turn 3", work.called)
	}
	if !strings.Contains(res.Final, "Completed step") {
		t.Fatalf("final answer should be the wrap-up summary, got %q", res.Final)
	}
	// The steer message must have been injected before the wrap-up turn.
	foundSteer := false
	for _, m := range res.Messages {
		if m.Role == RoleUser && strings.Contains(m.Content, "run budget for this period has been exhausted") {
			foundSteer = true
		}
	}
	if !foundSteer {
		t.Fatal("budget-exhausted steer message was not injected into history")
	}
}

// TestBudgetGateInactiveWhenUnderCap confirms an uncapped-enough run finishes
// normally: the gate never trips, no steer is injected, and the model reaches its
// own final answer.
func TestBudgetGateInactiveWhenUnderCap(t *testing.T) {
	work := &echoTool{name: "do_work"}
	faux := scriptedNativeProvider(
		toolCallWithCost("w1", "do_work", `{"step":1}`, 0.10),
		AssistantText("done"),
	)
	limits := DefaultLimits()
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(work),
		Policy:         NewAllowList("do_work"),
		Limits:         &limits,
		BudgetGate: func(_ context.Context, u Usage) bool {
			return u.CostUSD >= 100.0
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "do the work")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "done" {
		t.Fatalf("final = %q want normal completion", res.Final)
	}
	if res.StopReason == "budget_exhausted" {
		t.Fatal("budget gate tripped under cap")
	}
}

// The defect these guard: a run that reached a ceiling returned
// lastAssistantText(), which is empty for the ordinary case where every turn was
// a tool call. The caller was handed nothing after paying for every token the run
// spent, and the chat surface rendered that as "the agent never answered".
//
// Every ceiling now ends the same way the budget one already did — one tool-free
// turn where the model says what it found.

// countSteers reports how many injected wrap-up steers are in the history. The
// steer must appear once: two ceilings tripping in one run (a tool budget that
// runs out on the last permitted turn) must not stack instructions on the model.
func countSteers(messages []Message) int {
	n := 0
	for _, m := range messages {
		if m.Role != RoleUser {
			continue
		}
		if strings.Contains(m.Content, "Do not call any more tools") {
			n++
		}
	}
	return n
}

// TestMaxTurnsEndsWithAnAnswerNotSilence is the reported failure: twelve turns of
// real work, then "Something went wrong. Try again."
func TestMaxTurnsEndsWithAnAnswerNotSilence(t *testing.T) {
	work := &echoTool{name: "run_sql"}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "run_sql", `{"q":1}`),
		AssistantToolCall("c2", "run_sql", `{"q":2}`),
		AssistantToolCall("c3", "run_sql", `{"q":3}`),
		// The borrowed turn. A model handed an empty tool list answers in text.
		AssistantText("Signup to activation is the weakest step; I would check the email step next."),
		// Never reached: a fourth work call would mean the run walked past its cap.
		AssistantToolCall("c4", "run_sql", `{"q":4}`),
	)
	limits := DefaultLimits()
	limits.MaxTurns = 3
	agent, err := New(Config{NativeProvider: faux, Model: "test", Tools: NewToolSet(work), Policy: NewAllowList("run_sql"), Limits: &limits})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "what is my weakest step")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final == "" {
		t.Fatal("run hit its turn cap and returned no answer at all — the caller pays for the tokens either way")
	}
	if !strings.Contains(res.Final, "weakest step") {
		t.Fatalf("final = %q, want the wrap-up the model wrote", res.Final)
	}
	if res.StopReason != "max_turns" {
		t.Fatalf("stop reason = %q, want max_turns — the wrap-up must not disguise which ceiling stopped the run", res.StopReason)
	}
	if work.called != 3 {
		t.Fatalf("tool called %d times, want 3: the wrap-up turn is tool-free and must not buy extra work", work.called)
	}
	if n := countSteers(res.Messages); n != 1 {
		t.Fatalf("steer injected %d times, want exactly 1", n)
	}
}

// TestMaxToolCallsEndsWithAnAnswerNotSilence covers the other work ceiling. It
// used to return from inside the batch guard, so the blocked tool results were
// the last thing in the history and nothing ever read them.
func TestMaxToolCallsEndsWithAnAnswerNotSilence(t *testing.T) {
	work := &echoTool{name: "run_sql"}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "run_sql", `{"q":1}`),
		AssistantToolCall("c2", "run_sql", `{"q":2}`),
		// Native admission removes tools before this next model turn.
		AssistantText("Two queries in, activation looks like the gap. I ran out of queries before confirming it."),
	)
	limits := DefaultLimits()
	limits.MaxToolCalls = 2
	agent, err := New(Config{NativeProvider: faux, Model: "test", Tools: NewToolSet(work), Policy: NewAllowList("run_sql"), Limits: &limits})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "what is my weakest step")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if !strings.Contains(res.Final, "activation looks like the gap") {
		t.Fatalf("final = %q, want the wrap-up written after the tool budget ran out", res.Final)
	}
	if res.StopReason != "max_tool_calls" {
		t.Fatalf("stop reason = %q, want max_tool_calls", res.StopReason)
	}
	if work.called != 2 {
		t.Fatalf("tool called %d times, want 2 — the blocked batch must not execute", work.called)
	}
}

func TestMaxToolCallsIsHardCapForParallelBatch(t *testing.T) {
	work := &budgetParallelEcho{echoTool: &echoTool{name: "parallel_work"}}
	batch := ChatResponse{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "c1", Name: work.Name(), Arguments: `{"q":1}`},
			{ID: "c2", Name: work.Name(), Arguments: `{"q":2}`},
		}},
		StopReason: "tool_calls",
	}
	limits := DefaultLimits()
	limits.MaxToolCalls = 1
	agent, err := New(Config{
		NativeProvider: scriptedNativeProvider(batch, AssistantText("one call completed; the other hit the cap")),
		Model:          "test", Tools: NewToolSet(work), Policy: NewAllowList(work.Name()), Limits: &limits,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	result, err := agent.Prompt(context.Background(), "run both")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if work.called != 1 {
		t.Fatalf("parallel executions = %d, want hard cap of 1", work.called)
	}
	allowed, exhausted := 0, 0
	for _, trace := range result.Tools {
		if trace.Allowed {
			allowed++
		}
		if trace.Reason == "tool-call budget exhausted" {
			exhausted++
		}
	}
	if allowed != 1 || exhausted != 1 {
		t.Fatalf("traces = %+v, want one execution and one budget rejection", result.Tools)
	}
}

// TestWrapUpIsHonestAboutHavingNothing pins the instruction, not just the
// mechanism. A summary that invents a conclusion from a run that never reached
// one is worse than the silence it replaced, so the steer has to tell the model
// it may say it could not answer.
func TestWrapUpIsHonestAboutHavingNothing(t *testing.T) {
	for _, reason := range []string{"max_turns", "max_tool_calls"} {
		steer := finalizeSteer(reason)
		if !strings.Contains(steer, "Do not call any more tools") {
			t.Fatalf("%s steer must strip the model of tool use in words as well as schema: %q", reason, steer)
		}
		if !strings.Contains(steer, "not enough to answer") {
			t.Fatalf("%s steer must license an honest 'I could not answer': %q", reason, steer)
		}
	}
	if finalizeSteer("budget_exhausted") != budgetExhaustedSteer {
		t.Fatal("an unrecognised reason must fall back to the budget steer, not to empty text")
	}
}

// TestWrapUpBorrowsExactlyOneTurn is the containment guard. The wrap-up strips
// the tool schemas, but a provider is free to ignore that; if the grant were
// expressed against the turn budget it would renew forever under bookkeeping
// refunds, which is a hang rather than a wasted turn.
func TestWrapUpBorrowsExactlyOneTurn(t *testing.T) {
	work := &echoTool{name: "run_sql"}
	script := make([]ChatResponse, 0, 40)
	for i := 0; i < 40; i++ {
		script = append(script, AssistantToolCall("c", "run_sql", `{"q":1}`))
	}
	faux := scriptedNativeProvider(script...)
	limits := DefaultLimits()
	limits.MaxTurns = 4
	limits.MaxToolCalls = 100
	agent, err := New(Config{NativeProvider: faux, Model: "test", Tools: NewToolSet(work), Policy: NewAllowList("run_sql"), Limits: &limits})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "never stop")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Turns != limits.MaxTurns+1 {
		t.Fatalf("turns = %d, want %d — the wrap-up is one borrowed turn, not an extension", res.Turns, limits.MaxTurns+1)
	}
	if res.StopReason != "max_turns" {
		t.Fatalf("stop reason = %q, want max_turns", res.StopReason)
	}
}

// TestCleanFinishOnTheLastTurnKeepsItsOwnAnswer is why the wrap-up borrows a turn
// instead of reserving one. Reserving would spend every run's last working turn
// buying an answer only overrunning runs need, and would overwrite a complete
// answer with a summary of itself.
func TestCleanFinishOnTheLastTurnKeepsItsOwnAnswer(t *testing.T) {
	work := &echoTool{name: "run_sql"}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "run_sql", `{"q":1}`),
		AssistantToolCall("c2", "run_sql", `{"q":2}`),
		AssistantText("Activation is the weakest step, at 24%."),
	)
	limits := DefaultLimits()
	limits.MaxTurns = 3
	agent, err := New(Config{NativeProvider: faux, Model: "test", Tools: NewToolSet(work), Policy: NewAllowList("run_sql"), Limits: &limits})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "what is my weakest step")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "Activation is the weakest step, at 24%." {
		t.Fatalf("final = %q — a run that finished on its last turn must keep its own answer", res.Final)
	}
	if res.StopReason == "max_turns" {
		t.Fatal("a clean finish on the last permitted turn is not a ceiling stop")
	}
	if n := countSteers(res.Messages); n != 0 {
		t.Fatalf("steer injected %d times into a run that finished on its own", n)
	}
}
