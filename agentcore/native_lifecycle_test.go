package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

type nativeBatchTestExtension struct {
	run func(context.Context, []ToolCall) BatchDecision
}

type nativeLifecycleMemory struct{}

func (nativeLifecycleMemory) Recall(context.Context, string, string, int) ([]MemoryEntry, error) {
	return []MemoryEntry{{Content: "private memory"}}, nil
}
func (nativeLifecycleMemory) Remember(context.Context, MemoryEntry) error { return nil }

func TestStartRunLifecycleReportsMemoryWithoutContent(t *testing.T) {
	host, err := nativeHostAgent(t, Config{Memory: nativeLifecycleMemory{}}).OpenTools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	var stages []string
	if _, err := host.StartRunWithLifecycle(context.Background(), "task", func(stage string) {
		stages = append(stages, stage)
	}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(stages, []string{"Checking relevant memory", "Memory context ready (1)"}) {
		t.Fatalf("lifecycle stages = %v", stages)
	}
	if strings.Contains(strings.Join(stages, " "), "private memory") {
		t.Fatal("lifecycle leaked recalled memory content")
	}
}

func (nativeBatchTestExtension) Name() string { return "batch-test" }
func (e nativeBatchTestExtension) BeginRun(context.Context, RunInfo) (Extension, error) {
	return e, nil
}
func (e nativeBatchTestExtension) InterceptBatch(ctx context.Context, calls []ToolCall) BatchDecision {
	return e.run(ctx, calls)
}

func TestNativeDelegationBatchPolicies(t *testing.T) {
	for _, mode := range []string{"settled", "parked", "terminal", "finalizing", "cancelled", "not-started", "closed"} {
		t.Run(mode, func(t *testing.T) {
			ctx := context.Background()
			var observed []ToolCall
			var batchCalls, turnEnds int
			calls := []ToolCall{{ID: "ordinary", Name: "read", Arguments: `{}`}, {ID: "child-a", Name: "spawn", Arguments: `{"task":"a"}`}, {ID: "child-b", Name: "spawn", Arguments: `{"task":"b"}`}}
			original := slices.Clone(calls)
			ext := nativeBatchTestExtension{run: func(_ context.Context, batch []ToolCall) BatchDecision {
				batchCalls++
				observed = slices.Clone(batch)
				batch[0].ID = "extension mutation"
				return BatchDecision{AdditionalContexts: []Message{{Role: RoleUser, Content: "batch"}}}
			}}
			a := nativeHostAgent(t, Config{Extensions: []ExtensionFactory{ext}, Hooks: Hooks{TurnEnd: []TurnHook{func(context.Context, TurnInfo) { turnEnds++ }}}})
			host, err := a.OpenTools(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer host.Close()
			if mode != "not-started" {
				if _, err := host.StartRun(ctx, "task"); err != nil {
					t.Fatal(err)
				}
			}
			outcomes := []ToolOutcome{
				{AdditionalContexts: []Message{{Role: RoleUser, Content: "ordinary"}}},
				{AdditionalContexts: []Message{{Role: RoleUser, Content: "child-a"}}},
				{AdditionalContexts: []Message{{Role: RoleUser, Content: "child-b"}}},
			}
			switch mode {
			case "parked":
				outcomes[2].Parked = true
			case "terminal":
				outcomes[0].Terminate = true
			case "finalizing":
				host.finalizing.Store(true)
			case "cancelled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "closed":
				_ = host.Close()
			}
			decision, err := host.FinishDelegationBatch(ctx, calls, outcomes)
			wantError := mode == "cancelled" || mode == "not-started" || mode == "closed"
			if (err != nil) != wantError {
				t.Fatalf("decision=%+v error=%v", decision, err)
			}
			if mode == "cancelled" && !errors.Is(err, context.Canceled) {
				t.Fatalf("lost cancellation: %v", err)
			}
			if turnEnds != 0 || !reflect.DeepEqual(calls, original) {
				t.Fatalf("resumed batch reran TurnEnd or changed source calls: ends=%d calls=%+v", turnEnds, calls)
			}
			if mode == "settled" {
				if batchCalls != 1 || !reflect.DeepEqual(observed, original) || decision.End {
					t.Fatalf("missing original batch: calls=%+v decision=%+v", observed, decision)
				}
				var contexts []string
				for _, m := range decision.Inject {
					contexts = append(contexts, m.Content)
				}
				if !reflect.DeepEqual(contexts, []string{"ordinary", "child-a", "child-b", "batch"}) {
					t.Fatalf("context order: %v", contexts)
				}
			} else if batchCalls != 0 || len(decision.Inject) != 0 || decision.End != !wantError || decision.Parked != (mode == "parked") {
				t.Fatalf("unsettled/terminal batch leaked hook or context: calls=%d decision=%+v", batchCalls, decision)
			}
		})
	}
}

func TestNativeDelegationBatchCloseCancelsHook(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	started := make(chan struct{})
	ext := nativeBatchTestExtension{run: func(ctx context.Context, _ []ToolCall) BatchDecision {
		close(started)
		<-ctx.Done()
		return BatchDecision{AdditionalContexts: []Message{{Role: RoleUser, Content: "must not persist"}}}
	}}
	a := nativeHostAgent(t, Config{Extensions: []ExtensionFactory{ext}})
	host, err := a.OpenTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	if _, err := host.StartRun(ctx, "task"); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		decision, err := host.FinishDelegationBatch(ctx, []ToolCall{{ID: "a", Name: "spawn", Arguments: `{}`}}, nil)
		if len(decision.Inject) != 0 {
			err = errors.New("cancelled batch retained contexts")
		}
		done <- err
	}()
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	closed := make(chan struct{})
	go func() { _ = host.Close(); close(closed) }()
	select {
	case <-closed:
	case <-ctx.Done():
		t.Fatal("Close did not settle the batch callback")
	}
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled batch returned %v", err)
	}
}

func TestNativeLifecycleHostCloseCancelsBlockedStep(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	started := make(chan struct{})
	a := nativeHostAgent(t, Config{StepGate: func(ctx context.Context, _ int) error { close(started); <-ctx.Done(); return ctx.Err() }})
	host, err := a.OpenTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	done := make(chan error, 1)
	go func() { _, err := host.PrepareTurn(ctx, StepInfo{Turn: 1}); done <- err }()
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	closed := make(chan struct{})
	go func() { _ = host.Close(); close(closed) }()
	select {
	case <-closed:
	case <-ctx.Done():
		t.Fatal("host close did not cancel lifecycle hook")
	}
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("step returned %v", err)
	}
}

func TestNativeLifecycleMetersChildSpendAndEmitsTerminalObserverOnce(t *testing.T) {
	ctx := context.Background()
	var endCalls int
	var observed RunResult
	a := nativeHostAgent(t, Config{BudgetGate: func(_ context.Context, u Usage) bool { return u.InputTokens >= 10 }, Hooks: Hooks{AgentEnd: []AgentEndHook{func(_ context.Context, result RunResult) { endCalls++; observed = result }}}})
	host, err := a.OpenTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	a.addChildUsage(Usage{InputTokens: 10, OutputTokens: 2, CacheReadTokens: 3, CostUSD: 0.25, CostUnpriced: true})
	prepared, err := host.PrepareTurn(ctx, StepInfo{Turn: 2, Usage: Usage{InputTokens: 1}})
	if err != nil || prepared.StopReason != "budget_exhausted" {
		t.Fatalf("child spend bypassed parent budget: %+v %v", prepared, err)
	}
	result := host.CompleteRun(ctx, RunResult{Usage: Usage{InputTokens: 1, OutputTokens: 1}})
	if result.Usage.InputTokens != 11 || result.Usage.OutputTokens != 3 || result.Usage.CacheReadTokens != 3 || result.Usage.CostUSD != 0.25 || !result.Usage.CostUnpriced || observed.Usage != result.Usage {
		t.Fatalf("lost child metering: %+v observed=%+v", result, observed)
	}
	_ = host.CompleteRun(ctx, result)
	if endCalls != 1 || a.peekChildUsage() != (Usage{}) {
		t.Fatal("child usage or terminal observer was repeated")
	}
}

func TestNativeContextHooksPreserveNativeMessagesOnFailure(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []string{"panic", "error", "invalid"} {
		t.Run(kind, func(t *testing.T) {
			var reports int
			a := nativeHostAgent(t, Config{Hooks: Hooks{ErrorPolicy: HookThrow, OnError: func(string, error) { reports++ }, Context: []ContextHook{
				func(_ context.Context, messages []json.RawMessage) ([]json.RawMessage, error) {
					messages[0][0] = '!'
					switch kind {
					case "panic":
						panic("bad extension")
					case "error":
						return nil, errors.New("bad extension")
					default:
						return messages, nil
					}
				},
				func(_ context.Context, messages []json.RawMessage) ([]json.RawMessage, error) {
					return append(messages, json.RawMessage(`{"role":"system","content":"kept valid hook","timestamp":1}`)), nil
				},
			}}})
			host, err := a.OpenTools(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer host.Close()
			input := json.RawMessage(`[{"role":"assistant","content":[{"type":"thinking","thinking":"private","thinkingSignature":"native-signature"}],"opaque":{"preserve":true}}]`)
			output := host.TransformContext(ctx, input)
			if reports != 1 || !json.Valid(output) || !strings.Contains(string(output), "native-signature") || !strings.Contains(string(output), "kept valid hook") || input[1] != '{' {
				t.Fatalf("native transform lost its last valid view: reports=%d out=%s input=%s", reports, output, input)
			}
		})
	}
}

type nativeContextTestExtension struct {
	label     string
	transform ContextHook
}

func (e nativeContextTestExtension) Name() string { return e.label }
func (e nativeContextTestExtension) BeginRun(context.Context, RunInfo) (Extension, error) {
	return e, nil
}
func (e nativeContextTestExtension) TransformNativeContext(ctx context.Context, messages []json.RawMessage) ([]json.RawMessage, error) {
	return e.transform(ctx, messages)
}

func TestNativeContextExtensionsComposeAndIsolateFailedMutations(t *testing.T) {
	ctx := context.Background()
	var reports []string
	a := nativeHostAgent(t, Config{
		Hooks: Hooks{ErrorPolicy: HookThrow, OnError: func(source string, _ error) { reports = append(reports, source) }},
		Extensions: []ExtensionFactory{
			nativeContextTestExtension{label: "broken", transform: func(_ context.Context, messages []json.RawMessage) ([]json.RawMessage, error) {
				messages[0][0] = '!'
				return nil, errors.New("failed context extension")
			}},
			nativeContextTestExtension{label: "reminder", transform: func(_ context.Context, messages []json.RawMessage) ([]json.RawMessage, error) {
				return append(messages, json.RawMessage(`{"role":"system","content":"run-local reminder"}`)), nil
			}},
		},
	})
	host, err := a.OpenTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	input := json.RawMessage(`[{"role":"assistant","content":[{"type":"thinking","thinking":"private","thinkingSignature":"native-signature"}],"opaque":9007199254740993}]`)
	before := string(input)
	output := host.TransformContext(ctx, input)
	if !reflect.DeepEqual(reports, []string{"extension_context[broken]"}) || !strings.Contains(string(output), "run-local reminder") || !strings.Contains(string(output), "9007199254740993") || !strings.Contains(string(output), "native-signature") || string(input) != before {
		t.Fatalf("extension transform lost context or isolation: reports=%v output=%s", reports, output)
	}
}
