package subagent_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/subagent"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry"
)

func TestSafeAsyncChildBeforeAndAfterParentSettlement(t *testing.T) {
	for _, late := range []bool{false, true} {
		t.Run(map[bool]string{false: "already-admitted", true: "starts-after-root"}[late], func(t *testing.T) {
			o, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{})
			if err != nil {
				t.Fatal(err)
			}
			ctx, err := telemetry.WithSafeObserver(context.Background(), o, telemetry.SafeExecution{ExecutionID: "parent-execution"})
			if err != nil {
				t.Fatal(err)
			}
			childOpened, childRelease, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
			u := func(n float64) *ai.Usage {
				return &ai.Usage{Input: n, Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}
			}
			parentScript := ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "toolUse", Usage: u(2), Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "spawn", Name: subagent.ToolSpawnSubagent, Arguments: json.RawMessage(`{"task":"async child work","async":true}`)})}, ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "parent finished"}), Usage: u(3)})
			provider := nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, opts map[string]any) (*ai.AssistantMessageEventStream, error) {
				child := false
				for _, m := range view.Messages() {
					if m.Role == "user" && m.Content.Text != nil && strings.Contains(*m.Content.Text, "async child work") {
						child = true
					}
				}
				if !child {
					return parentScript(ctx, model, view, opts)
				}
				close(childOpened)
				<-childRelease
				return ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "child finished"}), Usage: u(17)})(ctx, model, view, opts)
			})
			a, err := agentcore.New(agentcore.Config{Model: "test", NativeProvider: provider, Policy: agentcore.NewAllowList(subagent.ToolSpawnSubagent), Extensions: []agentcore.ExtensionFactory{&subagent.Plugin{AllowAsync: true}}})
			if err != nil {
				t.Fatal(err)
			}
			var pending func(context.Context) (string, error)
			launch := func(run func(context.Context) (string, error)) {
				go func() { _, err := run(context.Background()); done <- err }()
			}
			ctx = agentcore.WithBackgroundLauncher(ctx, func(_, _ string, run func(context.Context) (string, error)) (json.RawMessage, error) {
				if late {
					pending = run
				} else {
					launch(run)
					<-childOpened
				}
				return json.RawMessage(`{"job_id":"opaque-job"}`), nil
			})
			private := telemetry.NewInMemory()
			result, err := a.RunNative(ctx, agentcore.NativeRun{Telemetry: private.Context, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "parent work"}}})
			if err != nil || result.Final != "parent finished" || result.Usage.InputTokens != 5 {
				t.Fatalf("async changed parent: %+v %v", result, err)
			}
			parentRecords := o.Drain(256)
			if late {
				launch(pending)
				<-childOpened
			}
			close(childRelease)
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			childRecords := o.Drain(256)
			parentAttempts, childAttempts, tokens := 0, 0, 0.0
			for _, r := range append(parentRecords, childRecords...) {
				if r.Kind != "attempt_settled" {
					continue
				}
				tokens += r.Accounting.Tokens.Input
				switch r.Category {
				case telemetry.CategoryMain:
					parentAttempts++
				case telemetry.CategoryChild:
					childAttempts++
					if r.ExecutionID == "parent-execution" || r.OriginExecutionID == nil || *r.OriginExecutionID != "parent-execution" || r.ParentExecutionID == nil || r.ParentSpanID == nil {
						t.Fatalf("detached child correlation: %+v", r)
					}
				}
			}
			if parentAttempts != 2 || childAttempts != 1 || tokens != 22 || !o.Health().Complete {
				t.Fatalf("async usage lost/duplicated parent=%d child=%d tokens=%g", parentAttempts, childAttempts, tokens)
			}
			if late {
				for _, s := range private.GetSpans() {
					if s.Name == "agentray.agent.run" && s.ParentID != nil {
						t.Fatal("late child reopened private recorder")
					}
				}
			}
		})
	}
}

func TestSafeDurableReattachMakesNoSyntheticAttempt(t *testing.T) {
	o, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := telemetry.WithSafeObserver(context.Background(), o, telemetry.SafeExecution{ExecutionID: "durable-parent"})
	completed := false
	settings := subagent.Plugin{RunFork: func(ctx context.Context, _ *agentcore.Agent, _ subagent.ForkRequest, _ agentcore.StreamSink) (agentcore.RunResult, error) {
		if completed {
			return agentcore.RunResult{Final: "recorded child answer", StopReason: "stop"}, nil
		}
		p := nativeProvider(ai.ScriptedStream(nativeAnswer("recorded child answer")))
		_, err := p.Run(ctx, ai.NewAssistantMessageEventStream(), ai.FallbackRequest{Candidates: 1, Open: func(ctx context.Context, _, _ int) (*ai.AssistantMessageEventStream, error) {
			c := p.Candidates[0]
			return c.Stream(ctx, c.Model, ai.NormalizeContext(ai.Context{}), nil)
		}})
		completed = true
		return agentcore.RunResult{Final: "recorded child answer", StopReason: "stop"}, err
	}}
	host, err := durableParent(t, settings).OpenPiTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	for range 2 {
		if audit := executeSpawn(t, host, "same-recorded-effect", `{"task":"inspect"}`); audit.Trace.Error != "" {
			t.Fatal(audit.Trace.Error)
		}
	}
	attempts := 0
	for _, r := range o.Drain(256) {
		if r.Kind == "attempt_settled" {
			attempts++
			if r.Category != telemetry.CategoryChild {
				t.Fatal("reattach lost child category")
			}
		}
	}
	if attempts != 1 {
		t.Fatalf("completed reattach synthesized spend: %d attempts", attempts)
	}
}

func TestSafeDurableCorrectionCreatesFreshPhysicalAttempts(t *testing.T) {
	o, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := telemetry.WithSafeObserver(context.Background(), o, telemetry.SafeExecution{ExecutionID: "correction-parent"})
	if err != nil {
		t.Fatal(err)
	}
	const checkpoint = `{"messages":[{"role":"assistant","opaque":"preserve-me"}]}`
	calls := 0
	settings := subagent.Plugin{RunFork: func(ctx context.Context, _ *agentcore.Agent, req subagent.ForkRequest, _ agentcore.StreamSink) (agentcore.RunResult, error) {
		calls++
		answer := "not JSON"
		if calls == 2 {
			if req.Previous == nil || string(req.Previous.NativeState) != checkpoint {
				t.Fatal("correction lost checkpoint")
			}
			answer = `{"fruit":"banana"}`
		}
		m := nativeAnswer(answer)
		m.Usage = &ai.Usage{Input: float64(calls), Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}
		p := nativeProvider(ai.ScriptedStream(m))
		_, err := p.Run(ctx, ai.NewAssistantMessageEventStream(), ai.FallbackRequest{Candidates: 1, Open: func(ctx context.Context, _, _ int) (*ai.AssistantMessageEventStream, error) {
			candidate := p.Candidates[0]
			return candidate.Stream(ctx, candidate.Model, ai.NormalizeContext(ai.Context{}), nil)
		}})
		return agentcore.RunResult{Final: answer, NativeState: json.RawMessage(checkpoint), StopReason: "stop"}, err
	}}
	host, err := durableParent(t, settings).OpenPiTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	audit := executeSpawn(t, host, "correction-effect", `{"task":"name a fruit","output_schema":`+fruitSchema+`}`)
	if calls != 2 || audit.Trace.Error != "" {
		t.Fatalf("correction changed outcome: calls=%d error=%s", calls, audit.Trace.Error)
	}
	var attempts []telemetry.SafeRecord
	for _, r := range o.Drain(256) {
		if r.Kind == "attempt_settled" {
			attempts = append(attempts, r)
		}
	}
	if len(attempts) != 2 || *attempts[0].AttemptID == *attempts[1].AttemptID || *attempts[0].CallID == *attempts[1].CallID || attempts[0].Accounting.Tokens.Input+attempts[1].Accounting.Tokens.Input != 3 {
		t.Fatalf("correction lost/duplicated new spend: %+v", attempts)
	}
	for _, r := range attempts {
		if r.Category != telemetry.CategoryChild || r.ExecutionID == "correction-parent" || r.OriginExecutionID == nil || *r.OriginExecutionID != "correction-parent" {
			t.Fatalf("correction origin: %+v", r)
		}
	}
}
