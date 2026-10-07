package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/ai"
)

// echoTool is a trivial tool that returns its arguments, recording invocation.
type echoTool struct {
	name   string
	called int
}

func (e *echoTool) Name() string { return e.name }
func (e *echoTool) Schema() ToolSchema {
	return ToolSchema{Name: e.name, Description: "echo", Parameters: map[string]any{"type": "object"}}
}
func (e *echoTool) Run(_ context.Context, args string) (string, error) {
	e.called++
	return "ran " + e.name + " with " + args, nil
}

// TestLoopRunsPermittedTool drives a faux two-turn script: turn 1 calls the
// tool, turn 2 produces the final answer. Verifies the tool executed and the
// trace recorded it as allowed.
func TestLoopRunsPermittedTool(t *testing.T) {
	tool := &echoTool{name: "run_query"}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "run_query", `{"sql":"select 1"}`),
		AssistantText("done: the query returned 1"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(tool),
		Policy:         NewAllowList("run_query"),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "run a query")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if tool.called != 1 {
		t.Fatalf("expected tool called once, got %d", tool.called)
	}
	if res.Final != "done: the query returned 1" {
		t.Fatalf("unexpected final: %q", res.Final)
	}
	if len(res.Tools) != 1 || !res.Tools[0].Allowed {
		t.Fatalf("expected 1 allowed tool trace, got %+v", res.Tools)
	}
}

// TestPermissionGateBlocks verifies the beforeToolCall permission gate denies a
// tool the policy doesn't permit, feeds the reason back to the model (not a
// silent failure), and records allowed=false.
func TestPermissionGateBlocks(t *testing.T) {
	tool := &echoTool{name: "write_dashboard"}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "write_dashboard", `{}`),
		AssistantText("understood, I cannot write dashboards"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(tool),
		Policy:         NewAllowList(), // permits nothing
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "build a dashboard")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if tool.called != 0 {
		t.Fatalf("blocked tool must not execute, got %d calls", tool.called)
	}
	if len(res.Tools) != 1 || res.Tools[0].Allowed {
		t.Fatalf("expected 1 blocked trace, got %+v", res.Tools)
	}
	// The denial surfaces through the policy-filtered advertisement: the tool is
	// never offered, and a forced call reads back as not found.
	var sawDenial bool
	for _, m := range res.Messages {
		if m.Role == RoleTool && strings.Contains(m.Content, "not found") {
			sawDenial = true
		}
	}
	if !sawDenial {
		t.Fatal("block reason was not returned to the model")
	}
}

// TestPermittedToolsFiltersSchemas verifies a disabled tool never reaches the
// model's advertised schema list.
func TestPermittedToolsFiltersSchemas(t *testing.T) {
	recorder := &nativeRecorder{}
	faux := recordedNativeProvider(recorder, AssistantText("hi"))
	agent, _ := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(&echoTool{name: "a"}, &echoTool{name: "b"}),
		Policy:         NewAllowList("a"),
	})
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	req := recorder.all()[0]
	if len(req.Tools) != 1 || req.Tools[0] != "a" {
		t.Fatalf("expected only tool 'a' advertised, got %+v", req.Tools)
	}
}

// TestPromptStreamEmitsTokens drives the streaming path: turn 1 calls a tool,
// turn 2 answers. It asserts the streamed tokens concatenate to the final
// answer and that a tool StreamEvent fired for the executed call (parity with
// the persisted trace).
func TestPromptStreamEmitsTokens(t *testing.T) {
	tool := &echoTool{name: "run_query"}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "run_query", `{"sql":"select 1"}`),
		AssistantText("the query returned exactly one"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(tool),
		Policy:         NewAllowList("run_query"),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var tokens strings.Builder
	var toolEvents int
	res, err := agent.PromptStream(context.Background(), "run a query", func(ev StreamEvent) {
		switch ev.Type {
		case StreamToken:
			tokens.WriteString(ev.Token)
		case StreamTool:
			toolEvents++
			if ev.Tool == nil || ev.Tool.Tool != "run_query" {
				t.Errorf("unexpected tool event: %+v", ev.Tool)
			}
		}
	})
	if err != nil {
		t.Fatalf("PromptStream: %v", err)
	}
	if res.Final != "the query returned exactly one" {
		t.Fatalf("unexpected final: %q", res.Final)
	}
	// Streamed tokens must reconstruct the final answer exactly.
	if tokens.String() != res.Final {
		t.Fatalf("streamed tokens %q != final %q", tokens.String(), res.Final)
	}
	if toolEvents != 1 {
		t.Fatalf("expected 1 tool stream event, got %d", toolEvents)
	}
	if tool.called != 1 {
		t.Fatalf("expected tool executed once, got %d", tool.called)
	}
}

// barrierTool is a parallel-eligible tool that signals arrival then blocks until
// released, so a test can prove two such tools in one turn ran concurrently:
// under sequential execution the first would never see its peer arrive and would
// time out.
type barrierTool struct {
	name    string
	arrived chan<- string
	release <-chan struct{}
}

func (b *barrierTool) Name() string   { return b.name }
func (b *barrierTool) Parallel() bool { return true }
func (b *barrierTool) Schema() ToolSchema {
	return ToolSchema{Name: b.name, Description: "barrier", Parameters: map[string]any{"type": "object"}}
}
func (b *barrierTool) Run(_ context.Context, _ string) (string, error) {
	b.arrived <- b.name
	select {
	case <-b.release:
		return "ran " + b.name, nil
	case <-time.After(2 * time.Second):
		return "", errors.New("timeout: peer never ran concurrently")
	}
}

// TestParallelToolsRunConcurrently verifies that when every tool call in a turn
// targets a ParallelTool, the batch executes concurrently and results are still
// applied in the model's original order.
func TestParallelToolsRunConcurrently(t *testing.T) {
	arrived := make(chan string, 2)
	release := make(chan struct{})
	a := &barrierTool{name: "qa", arrived: arrived, release: release}
	b := &barrierTool{name: "qb", arrived: arrived, release: release}

	// Once both tools have arrived, release them — proving concurrency.
	go func() {
		<-arrived
		<-arrived
		close(release)
	}()

	twoCalls := ChatResponse{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "c1", Name: "qa", Arguments: "{}"},
			{ID: "c2", Name: "qb", Arguments: "{}"},
		}},
		StopReason: "tool_calls",
	}
	faux := scriptedNativeProvider(twoCalls, AssistantText("both done"))
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(a, b),
		Policy:         NewAllowList("qa", "qb"),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "run both")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if len(res.Tools) != 2 {
		t.Fatalf("expected 2 tool traces, got %d", len(res.Tools))
	}
	// Order preserved despite concurrent execution.
	if res.Tools[0].Tool != "qa" || res.Tools[1].Tool != "qb" {
		t.Fatalf("tool order not preserved: %+v", res.Tools)
	}
	for _, tr := range res.Tools {
		if tr.Error != "" {
			t.Fatalf("tool %q errored (ran sequentially?): %s", tr.Tool, tr.Error)
		}
	}
}

// prepareTool rewrites its arguments before validation/execution and records
// what Run actually received.
type prepareTool struct {
	gotArgs string
}

func (p *prepareTool) Name() string { return "prep" }
func (p *prepareTool) Schema() ToolSchema {
	return ToolSchema{Name: "prep", Description: "prep", Parameters: map[string]any{"type": "object"}}
}
func (p *prepareTool) PrepareArguments(raw string) string { return `{"normalized":true}` }
func (p *prepareTool) Run(_ context.Context, args string) (string, error) {
	p.gotArgs = args
	return "ok", nil
}

// TestPrepareArgumentsNormalizes verifies ArgPreparer runs before execution and
// the normalized args reach both the tool and the persisted trace.
func TestPrepareArgumentsNormalizes(t *testing.T) {
	tool := &prepareTool{}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "prep", `{"raw":1}`),
		AssistantText("done"),
	)
	agent, _ := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(tool),
		Policy:         NewAllowList("prep"),
	})
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if tool.gotArgs != `{"normalized":true}` {
		t.Fatalf("tool did not receive normalized args: %q", tool.gotArgs)
	}
}

// (The per-turn RefreshKey/KeyUpdater seam is not part of native execution;
// credential refresh lives in the candidate StreamFn itself.)

// TestCancelledContextStopsBeforeProvider verifies an already-cancelled context
// aborts the run before any provider call.
func TestCancelledContextStopsBeforeProvider(t *testing.T) {
	recorder := &nativeRecorder{}
	faux := recordedNativeProvider(recorder, AssistantText("should not be reached"))
	agent, _ := New(Config{NativeProvider: faux, Model: "test"})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := agent.Prompt(ctx, "go"); err == nil {
		t.Fatalf("expected a cancellation error")
	}
	if recorder.count() != 0 {
		t.Fatalf("provider must not be called on a cancelled context, got %d calls", recorder.count())
	}
}


// TestEscalationFallsBackOnError verifies that when the primary candidate
// errors the fallback ladder retries the turn on the next candidate, succeeds,
// and uses that candidate's model.
func TestEscalationFallsBackOnError(t *testing.T) {
	primaryCalls := 0
	failing := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		primaryCalls++
		return nil, errors.New("provider down")
	}
	recorder := &nativeRecorder{}
	ladder := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"lite-model"}`), Stream: failing},
		{Model: json.RawMessage(`{"id":"pro-model"}`), Stream: recorder.wrap(scriptedNativeStream(AssistantText("recovered")))},
	}}
	retry := RetryPolicy{MaxAttempts: 1}
	agent, err := New(Config{
		NativeProvider: ladder,
		Model:          "lite-model",
		Retry:          &retry,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "recovered" {
		t.Fatalf("expected fallback answer, got %q", res.Final)
	}
	if primaryCalls != 1 {
		t.Fatalf("expected primary tried once, got %d", primaryCalls)
	}
	got := recorder.all()
	if len(got) != 1 || got[0].Model != "pro-model" {
		t.Fatalf("expected fallback rung used with its own model, got %+v", got)
	}
}

// TestEscalationExhaustedReturnsError verifies that when every candidate fails
// the run surfaces the error rather than looping forever, having tried each
// candidate once.
func TestEscalationExhaustedReturnsError(t *testing.T) {
	var bad1Calls, bad2Calls int
	bad1 := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		bad1Calls++
		return nil, errors.New("provider down")
	}
	bad2 := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		bad2Calls++
		return nil, errors.New("provider down")
	}
	ladder := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"a"}`), Stream: bad1},
		{Model: json.RawMessage(`{"id":"b"}`), Stream: bad2},
	}}
	retry := RetryPolicy{MaxAttempts: 1}
	agent, _ := New(Config{
		NativeProvider: ladder,
		Model:          "a",
		Retry:          &retry,
	})
	if _, err := agent.Prompt(context.Background(), "go"); err == nil {
		t.Fatal("expected an error when the whole ladder fails")
	}
	if bad1Calls != 1 || bad2Calls != 1 {
		t.Fatalf("expected each rung tried once, got %d and %d", bad1Calls, bad2Calls)
	}
}

// panicTool always panics, simulating a buggy tool (nil deref, etc.).
type panicTool struct{ name string }

func (p *panicTool) Name() string { return p.name }
func (p *panicTool) Schema() ToolSchema {
	return ToolSchema{Name: p.name, Description: "panics", Parameters: map[string]any{"type": "object"}}
}
func (p *panicTool) Run(context.Context, string) (string, error) {
	panic("boom")
}

// TestPanickingToolDegradesToError verifies a tool that panics is recovered into
// an ordinary error result — the run finishes and the model sees the failure
// rather than the whole run crashing.
func TestPanickingToolDegradesToError(t *testing.T) {
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "kaboom", `{}`),
		AssistantText("handled the failure"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(&panicTool{name: "kaboom"}),
		Policy:         NewAllowList("kaboom"),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "handled the failure" {
		t.Fatalf("expected run to continue past the panic, got %q", res.Final)
	}
	if len(res.Tools) != 1 || !strings.Contains(res.Tools[0].Error, "panicked") {
		t.Fatalf("expected a recovered panic trace, got %+v", res.Tools)
	}
	var sawError bool
	for _, m := range res.Messages {
		if m.Role == RoleTool && strings.Contains(m.Content, "error:") && strings.Contains(m.Content, "panicked") {
			sawError = true
		}
	}
	if !sawError {
		t.Fatal("panic error was not fed back to the model")
	}
}

// flakyTool always errors, counting executions, so a test can prove the circuit
// breaker stops executing it after repeated failures.
type flakyTool struct {
	name   string
	called int
}

func (f *flakyTool) Name() string { return f.name }
func (f *flakyTool) Schema() ToolSchema {
	return ToolSchema{Name: f.name, Description: "flaky", Parameters: map[string]any{"type": "object"}}
}
func (f *flakyTool) Run(context.Context, string) (string, error) {
	f.called++
	return "", errors.New("always fails")
}


func TestTruncateBytesIsRuneSafe(t *testing.T) {
	s := strings.Repeat("é", 100) // 2 bytes each
	out := truncateBytes(s, 50)
	if len(out) > 50 {
		t.Fatalf("truncate exceeded budget: %d", len(out))
	}
	if !strings.HasSuffix(out, "[truncated]") {
		t.Fatalf("expected truncation marker, got %q", out)
	}
}

func TestDeniedAbortedMatchesWireLiteral(t *testing.T) {
	if ToolDenialAborted != "aborted" {
		t.Fatalf("wire value changed: %q (historical rows carry \"aborted\")", ToolDenialAborted)
	}
	if !(ToolTrace{Reason: "aborted"}).DeniedAborted() {
		t.Fatal("historical reason=aborted must classify as abort denial")
	}
	if (ToolTrace{Reason: "tool-call budget exhausted"}).DeniedAborted() {
		t.Fatal("other denials must not classify as abort")
	}
}


// TestSteeringInjectedBeforeNextTurn verifies a steering message queued during
// turn 1 appears in the turn-2 request, ahead of the model's reasoning.
func TestSteeringInjectedBeforeNextTurn(t *testing.T) {
	// Turn 1: model calls a (permitted) no-op tool so the loop continues; turn 2:
	// final answer. Steering is queued once, drained on turn 2.
	steeringRecorder := &nativeRecorder{}
	faux := recordedNativeProvider(steeringRecorder,
		AssistantToolCall("c1", "noop", `{}`),
		AssistantText("ok"),
	)
	var delivered bool
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(noopTool{}),
		Policy:         NewAllowList("noop"),
		GetSteeringMessages: func(context.Context) []Message {
			if delivered {
				return nil
			}
			delivered = true
			return []Message{{Role: RoleUser, Content: "STEER: prefer option B"}}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	// The second recorded request must contain the steering message.
	recorded := steeringRecorder.all()
	if len(recorded) < 2 {
		t.Fatalf("expected at least 2 turns, got %d", len(recorded))
	}
	var seen bool
	for _, m := range recorded[1].Messages {
		if strings.Contains(m.Content, "STEER: prefer option B") {
			seen = true
		}
	}
	if !seen {
		t.Fatalf("steering message not present in turn-2 request: %+v", recorded[1].Messages)
	}
}

// TestFollowUpRestartsLoop verifies a follow-up queued after the final answer
// restarts the loop instead of ending the run.
func TestFollowUpRestartsLoop(t *testing.T) {
	followRecorder := &nativeRecorder{}
	faux := recordedNativeProvider(followRecorder,
		AssistantText("first answer"),
		AssistantText("second answer"),
	)
	var sent bool
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		GetFollowUpMessages: func(context.Context) []Message {
			if sent {
				return nil
			}
			sent = true
			return []Message{{Role: RoleUser, Content: "now do the next thing"}}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "start")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "second answer" {
		t.Fatalf("loop did not restart on follow-up: final=%q turns=%d", res.Final, res.Turns)
	}
	if res.Turns != 2 {
		t.Fatalf("expected 2 turns after one follow-up, got %d", res.Turns)
	}
	// The follow-up must have entered the second request.
	var seen bool
	recorded := followRecorder.all()
	if len(recorded) > 1 {
		for _, m := range recorded[1].Messages {
			if strings.Contains(m.Content, "now do the next thing") {
				seen = true
			}
		}
	}
	if !seen {
		t.Fatalf("follow-up not present in restarted turn: %+v", recorded)
	}
}

// TestFollowUpRespectsMaxTurns verifies an always-on follow-up queue cannot loop
// past the turn budget.
func TestFollowUpRespectsMaxTurns(t *testing.T) {
	faux := scriptedNativeProvider() // exhaustion returns an empty stop answer
	limits := DefaultLimits()
	limits.MaxTurns = 3
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Limits:         &limits,
		GetFollowUpMessages: func(context.Context) []Message {
			return []Message{{Role: RoleUser, Content: "again"}}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "start")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	// The run gets one tool-free wrap-up turn past the ceiling, then stops.
	if res.Turns > 4 || res.StopReason != "max_turns" {
		t.Fatalf("follow-up loop ignored budget: turns=%d stop=%q", res.Turns, res.StopReason)
	}
}

// noopTool is a permitted do-nothing tool used to keep the loop alive for a turn.
type noopTool struct{}

func (noopTool) Name() string { return "noop" }
func (noopTool) Schema() ToolSchema {
	return ToolSchema{Name: "noop", Description: "does nothing", Parameters: map[string]any{"type": "object"}}
}
func (noopTool) Run(context.Context, string) (string, error) { return "ok", nil }

// TestLifecycleEventOrder verifies a streamed run with one tool turn followed by
// a final answer emits the granular lifecycle events in the documented order,
// and that the back-compat token/tool events still appear.
func TestLifecycleEventOrder(t *testing.T) {
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "noop", `{}`),
		AssistantText("all done"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(noopTool{}),
		Policy:         NewAllowList("noop"),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var events []StreamEventType
	sink := func(ev StreamEvent) { events = append(events, ev.Type) }
	if _, err := agent.PromptStream(context.Background(), "go", sink); err != nil {
		t.Fatalf("PromptStream: %v", err)
	}

	// Reduce to the lifecycle skeleton (drop token/tool/message_update noise) and
	// assert the boundary order.
	want := []StreamEventType{
		StreamAgentStart,
		StreamTurnStart, StreamMessageStart, StreamMessageEnd,
		StreamToolExecStart, StreamToolExecEnd, StreamTurnEnd,
		StreamTurnStart, StreamMessageStart, StreamMessageEnd, StreamTurnEnd,
		StreamAgentEnd,
	}
	var skeleton []StreamEventType
	keep := map[StreamEventType]bool{
		StreamAgentStart: true, StreamTurnStart: true, StreamMessageStart: true,
		StreamMessageEnd: true, StreamToolExecStart: true, StreamToolExecEnd: true,
		StreamTurnEnd: true, StreamAgentEnd: true,
	}
	for _, e := range events {
		if keep[e] {
			skeleton = append(skeleton, e)
		}
	}
	if len(skeleton) != len(want) {
		t.Fatalf("lifecycle skeleton = %v\nwant %v", skeleton, want)
	}
	for i := range want {
		if skeleton[i] != want[i] {
			t.Fatalf("event %d = %q, want %q\nfull: %v", i, skeleton[i], want[i], skeleton)
		}
	}

	// Back-compat: the StreamTool event (completed trace) is still emitted.
	var sawTool bool
	for _, e := range events {
		if e == StreamTool {
			sawTool = true
		}
	}
	if !sawTool {
		t.Fatalf("back-compat StreamTool event missing: %v", events)
	}
}


// TestBusyGuardRejectsConcurrentRun verifies one Agent instance runs one run at
// a time: a reentrant Prompt on the *same* agent fails fast with ErrBusy instead
// of racing on the shared run state.
func TestBusyGuardRejectsConcurrentRun(t *testing.T) {
	var agent *Agent
	var reentryErr error
	reenter := func(ctx context.Context, _ Message) {
		_, reentryErr = agent.Prompt(ctx, "again")
	}
	var err error
	agent, err = New(Config{
		NativeProvider: scriptedNativeProvider(AssistantText("done")),
		Model:          "test",
		Hooks:          Hooks{MessageEnd: []MessageEndHook{reenter}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if !errors.Is(reentryErr, ErrBusy) {
		t.Fatalf("a reentrant run on the same agent must return ErrBusy, got %v", reentryErr)
	}
}

// TestFailureMessageSynthesizedOnProviderError verifies an aborting run still
// produces a clean lifecycle: a synthesized assistant failure message plus
// message_end / turn_end / agent_end events, while the error reaches the caller.
func TestFailureMessageSynthesizedOnProviderError(t *testing.T) {
	var seen []StreamEventType
	sink := func(ev StreamEvent) { seen = append(seen, ev.Type) }

	retry := RetryPolicy{MaxAttempts: 1}
	agent, err := New(Config{NativeProvider: nativeCandidate("err", failingNativeStream(errors.New("provider exploded"))), Model: "test", Retry: &retry})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.PromptStream(context.Background(), "go", sink)
	if err == nil {
		t.Fatalf("a provider error must surface to the caller")
	}
	last := res.Messages[len(res.Messages)-1]
	if last.Role != RoleAssistant || last.Content != "" {
		t.Fatalf("expected a synthesized empty assistant failure message, got %+v", last)
	}
	if res.StopReason != "error" {
		t.Fatalf("stop reason = %q, want error", res.StopReason)
	}
	has := func(want StreamEventType) bool {
		for _, e := range seen {
			if e == want {
				return true
			}
		}
		return false
	}
	for _, want := range []StreamEventType{StreamMessageEnd, StreamTurnEnd, StreamAgentEnd} {
		if !has(want) {
			t.Fatalf("failure lifecycle missing %q in %v", want, seen)
		}
	}
}

