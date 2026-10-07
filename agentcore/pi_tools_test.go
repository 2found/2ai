package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type piHostTool struct {
	name string
	run  func(context.Context, string) (string, error)
}

type piRetryHostTool struct{ piHostTool }

func (piRetryHostTool) RetrySafe() bool { return true }

func (t piHostTool) Name() string { return t.name }
func (t piHostTool) Schema() ToolSchema {
	return ToolSchema{Name: t.name, Parameters: map[string]any{"type": "object"}}
}
func (t piHostTool) Run(ctx context.Context, args string) (string, error) { return t.run(ctx, args) }

func piHostAgent(t *testing.T, cfg Config) *Agent {
	t.Helper()
	cfg.NativeProvider, cfg.Model = scriptedNativeProvider(AssistantText("unused")), "test"
	a, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func piHostCall(name, args string) json.RawMessage {
	v, _ := json.Marshal(map[string]any{"toolCallId": "call", "toolName": name, "args": json.RawMessage(args)})
	return v
}

func TestPiToolsRetainsWrappedChildQuestion(t *testing.T) {
	question := &ChildQuestionError{SessionID: "parent/child", QuestionID: "child-effect", Question: json.RawMessage(`{"question":"Which scope?","options":["one","two"]}`)}
	tool := piHostTool{"delegate", func(context.Context, string) (string, error) {
		return "", fmt.Errorf("sub-agent failed: %w", question)
	}}
	a := piHostAgent(t, Config{Tools: NewToolSet(tool), Policy: NewAllowList("delegate")})
	host, err := a.OpenPiTools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	raw, audit, err := host.Execute(context.Background(), piHostCall("delegate", `{"task":"work"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Details PiToolOutcome
		IsError bool
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.IsError || !wire.Details.Parked || wire.Details.QuestionID == "" || wire.Details.QuestionID == question.QuestionID || wire.Details.Trace.Args != `{"task":"work"}` {
		t.Fatalf("delegation did not retain separate parent/child identities: %s", raw)
	}
	route := wire.Details.ChildQuestion
	if route == nil || route.SessionID != question.SessionID || route.QuestionID != question.QuestionID || string(route.Question) != string(question.Question) {
		t.Fatalf("wrapped question lost its route: %s", raw)
	}
	question.Question[0] = '!'
	if audit.ChildQuestion == nil || !json.Valid(audit.ChildQuestion.Question) {
		t.Fatal("receipt aliases the tool's question buffer")
	}
}

func TestPiToolsInvalidChildQuestionCannotCorruptReceipt(t *testing.T) {
	for _, question := range []*ChildQuestionError{
		{SessionID: "child", QuestionID: "effect", Question: json.RawMessage(`broken`)},
		{SessionID: "child", QuestionID: "effect", Question: json.RawMessage(`null`)},
		{SessionID: "child", QuestionID: "effect", Question: json.RawMessage(`[]`)},
		{SessionID: "child", Question: json.RawMessage(`{}`)},
		{QuestionID: "effect", Question: json.RawMessage(`{}`)},
	} {
		tool := piHostTool{"delegate", func(context.Context, string) (string, error) { return "", question }}
		a := piHostAgent(t, Config{Tools: NewToolSet(tool), Policy: NewAllowList("delegate")})
		host, err := a.OpenPiTools(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		raw, audit, err := host.Execute(context.Background(), piHostCall("delegate", `{}`), nil)
		host.Close()
		if err != nil || !json.Valid(raw) || audit.ChildQuestion != nil || audit.Trace.Error == "" {
			t.Fatalf("invalid child route corrupted the settled failure: %s %+v %v", raw, audit, err)
		}
	}
}

func TestPiResumeDelegationPreservesIdentityAndRechecksAuthority(t *testing.T) {
	for _, mode := range []string{"allowed", "denied", "no longer retry safe"} {
		t.Run(mode, func(t *testing.T) {
			var keys []string
			tool := piHostTool{"delegate", func(ctx context.Context, _ string) (string, error) {
				key, _ := IdempotencyKey(ctx)
				keys = append(keys, key)
				if len(keys) == 1 {
					return "", &ChildQuestionError{SessionID: "parent/child", QuestionID: "child-question", Question: json.RawMessage(`{"question":"Which?"}`)}
				}
				return "child completed", nil
			}}
			store := NewMemorySessionStore()
			first := piHostAgent(t, Config{Tools: NewToolSet(piRetryHostTool{tool}), Policy: NewAllowList("delegate"), Session: store, SessionID: "parent"})
			host, err := first.OpenPiTools(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			_, original, err := host.Execute(WithToolInvocationScope(context.Background(), "original-effect"), piHostCall("delegate", `{"task":"work"}`), nil)
			host.Close()
			if err != nil || !original.Parked {
				t.Fatalf("initial delegation failed: %+v %v", original, err)
			}
			tools := NewToolSet(piRetryHostTool{tool})
			policy := NewAllowList("delegate")
			if mode == "denied" {
				policy = NewAllowList()
			}
			if mode == "no longer retry safe" {
				tools = NewToolSet(tool)
			}
			next := piHostAgent(t, Config{Tools: tools, Policy: policy, Session: store, SessionID: "parent"})
			host, err = next.OpenPiTools(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			defer host.Close()
			_, result, err := host.ResumeDelegation(context.Background(), "original-effect", original, nil)
			switch mode {
			case "allowed":
				if err != nil || !result.Executed || result.Parked || len(keys) != 2 || keys[0] == "" || keys[0] != keys[1] {
					t.Fatalf("resume changed physical identity: %+v %v keys=%v", result, err, keys)
				}
			case "denied":
				if err != nil || result.Executed || result.Trace.Allowed || len(keys) != 1 {
					t.Fatalf("resume bypassed current policy: %+v %v", result, err)
				}
			default:
				if err == nil || len(keys) != 1 {
					t.Fatal("resume ran a tool whose retry contract changed")
				}
			}
		})
	}
}

func TestPiToolsPreservesGovernedExecutionBoundary(t *testing.T) {
	ctx := WithToolInvocationScope(context.Background(), "recorded-effect-1")
	var toolArgs, hookArgs, identity string
	var writes int
	read := piHostTool{"read", func(ctx context.Context, args string) (string, error) {
		toolArgs = args
		identity, _ = IdempotencyKey(ctx)
		return strings.Repeat("result", 100), nil
	}}
	write := piHostTool{"write", func(context.Context, string) (string, error) { writes++; return "written", nil }}
	limits := DefaultLimits()
	limits.MaxToolResultLen = 80
	a := piHostAgent(t, Config{Tools: NewToolSet(read, write), Policy: NewAllowList("read"), Limits: &limits, SessionID: "native", Session: NewMemorySessionStore(),
		Env: &Env{Credentials: stubResolver{resolve: func(args string) (string, error) {
			return strings.ReplaceAll(args, "{{cred:KEY}}", "resolved-secret"), nil
		}}},
		Hooks: Hooks{Before: []BeforeToolCall{func(_ context.Context, c ToolCall) Decision { hookArgs = c.Arguments; return Allowed() }}},
	})
	h, err := a.OpenPiTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	if _, _, err := h.Execute(context.Background(), piHostCall("read", `{}`), nil); err == nil || toolArgs != "" {
		t.Fatal("durable tool executed without a recorded per-call identity")
	}
	definitions, err := h.Definitions(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(definitions), `"name":"write"`) || !strings.Contains(string(definitions), `"executionMode":"sequential"`) {
		t.Fatalf("incorrect native declarations: %s", definitions)
	}
	_, denied, err := h.Execute(ctx, piHostCall("write", `{}`), nil)
	if err != nil || denied.Trace.Allowed || writes != 0 || hookArgs != "" {
		t.Fatalf("gate bypassed: %+v %v", denied, err)
	}
	value, audit, err := h.Execute(ctx, piHostCall("read", `{"token":"{{cred:KEY}}"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(toolArgs, "resolved-secret") || strings.Contains(hookArgs, "resolved-secret") || strings.Contains(audit.Trace.Args, "resolved-secret") || strings.Contains(string(value), "resolved-secret") {
		t.Fatal("credential boundary not preserved")
	}
	if identity == "" || identity != audit.Trace.IdempotencyKey || !audit.Executed {
		t.Fatalf("missing governed identity: %+v", audit)
	}
	firstIdentity := identity
	_, repeated, err := h.Execute(WithToolInvocationScope(ctx, "recorded-effect-2"), piHostCall("read", `{"token":"{{cred:KEY}}"}`), nil)
	if err != nil || repeated.Trace.IdempotencyKey == firstIdentity {
		t.Fatalf("reused provider call ID reused a physical effect identity: %+v %v", repeated, err)
	}
	var result struct {
		Content []struct{ Text string }
		IsError bool
	}
	if err := json.Unmarshal(value, &result); err != nil || result.IsError || len(result.Content) != 1 || len(result.Content[0].Text) > 80 {
		t.Fatalf("native result escaped output bound: %s %v", value, err)
	}
}

func TestPiToolsNestedCallsSharePolicyAndBudget(t *testing.T) {
	var reads, writes int
	read := piHostTool{"read", func(context.Context, string) (string, error) { reads++; return "read", nil }}
	write := piHostTool{"write", func(context.Context, string) (string, error) { writes++; return "written", nil }}
	outer := piHostTool{"outer", func(ctx context.Context, _ string) (string, error) {
		invoke, ok := ToolInvokerFrom(ctx)
		if !ok {
			return "", errors.New("missing nested boundary")
		}
		if _, err := invoke.InvokeTool(ctx, "write", `{}`); err == nil {
			return "", errors.New("nested write allowed")
		}
		if _, err := invoke.InvokeTool(ctx, "read", `{}`); err != nil {
			return "", err
		}
		if _, err := invoke.InvokeTool(ctx, "read", `{}`); err == nil {
			return "", errors.New("nested call exceeded budget")
		}
		return "governed", nil
	}}
	limits := DefaultLimits()
	limits.MaxToolCalls = 2
	a := piHostAgent(t, Config{Tools: NewToolSet(outer, read, write), Policy: NewAllowList("outer", "read"), Limits: &limits})
	h, err := a.OpenPiTools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	value, audit, err := h.Execute(context.Background(), piHostCall("outer", `{}`), nil)
	if err != nil || audit.Trace.Error != "" || reads != 1 || writes != 0 || !strings.Contains(string(value), "governed") {
		t.Fatalf("nested governance failed: %s %+v %v", value, audit, err)
	}
	_, audit, err = h.Execute(context.Background(), piHostCall("read", `{}`), nil)
	if err != nil || audit.Executed || audit.Trace.Reason != "tool-call budget exhausted" || reads != 1 {
		t.Fatalf("direct call got a separate budget: %+v %v", audit, err)
	}
}

type piHostExtension struct {
	closed   *atomic.Int32
	contexts *atomic.Int32
	failure  bool
}

type piHostRunContextKey struct{}

func (e piHostExtension) RunContext(ctx context.Context) context.Context {
	if e.contexts != nil {
		e.contexts.Add(1)
	}
	return context.WithValue(ctx, piHostRunContextKey{}, ctx)
}

func (e piHostExtension) Name() string {
	if e.failure {
		return "failure"
	}
	return "resource"
}
func (e piHostExtension) BeginRun(context.Context, RunInfo) (Extension, error) {
	if e.failure {
		return nil, errors.New("extension setup failed")
	}
	return e, nil
}
func (e piHostExtension) CloseRun() { e.closed.Add(1) }

func TestPiToolsCloseCancelsCallsAndReleasesAgent(t *testing.T) {
	started := make(chan struct{})
	var closed atomic.Int32
	tool := piHostTool{"wait", func(ctx context.Context, _ string) (string, error) {
		close(started)
		<-ctx.Done()
		return "", ctx.Err()
	}}
	a := piHostAgent(t, Config{Tools: NewToolSet(tool), Policy: NewAllowList("wait"), Extensions: []ExtensionFactory{piHostExtension{closed: &closed}}})
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*3)
	defer cancel()
	h, err := a.OpenPiTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	if _, err := a.Prompt(ctx, "overlap"); !errors.Is(err, ErrBusy) {
		t.Fatalf("Agent busy slot bypassed: %v", err)
	}
	done := make(chan error, 1)
	go func() { _, _, err := h.Execute(ctx, piHostCall("wait", `{}`), nil); done <- err }()
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if err := h.Close(); err != nil {
		t.Fatal(err)
	}
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("tool cancellation: %v", err)
	}
	if closed.Load() != 1 {
		t.Fatalf("extension close count: %d", closed.Load())
	}
	next, err := a.OpenPiTools(ctx)
	if err != nil {
		t.Fatalf("Agent busy slot leaked: %v", err)
	}
	_ = next.Close()
}

func TestPiToolsSetupFailureClosesResources(t *testing.T) {
	for _, preparer := range []bool{false, true} {
		var closed atomic.Int32
		cfg := Config{Extensions: []ExtensionFactory{piHostExtension{closed: &closed}}}
		if preparer {
			cfg.Tools = NewToolSet(&prepareTool{})
		} else {
			cfg.Extensions = append(cfg.Extensions, piHostExtension{failure: true})
		}
		a := piHostAgent(t, cfg)
		for attempt := 1; attempt <= 2; attempt++ {
			_, err := a.OpenPiTools(context.Background())
			if err == nil || errors.Is(err, ErrBusy) || closed.Load() != int32(attempt) {
				t.Fatalf("setup leaked resources/busy slot: preparer=%v attempt=%d closed=%d err=%v", preparer, attempt, closed.Load(), err)
			}
		}
	}
}

func TestPiToolsContextContributionsOutliveIndividualCalls(t *testing.T) {
	var closed, contexts atomic.Int32
	var runContext context.Context
	type callbackKey struct{}
	tool := piHostTool{"context", func(ctx context.Context, _ string) (string, error) {
		runContext, _ = ctx.Value(piHostRunContextKey{}).(context.Context)
		if runContext == nil || ctx.Value(callbackKey{}) != "fenced-callback" {
			return "", errors.New("lost run or callback context")
		}
		return "ok", nil
	}}
	a := piHostAgent(t, Config{Tools: NewToolSet(tool), Policy: NewAllowList("context"), Extensions: []ExtensionFactory{piHostExtension{closed: &closed, contexts: &contexts}}})
	h, err := a.OpenPiTools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	for i := 0; i < 2; i++ {
		_, audit, err := h.Execute(context.WithValue(context.Background(), callbackKey{}, "fenced-callback"), piHostCall("context", `{}`), nil)
		if err != nil || audit.Trace.Error != "" {
			t.Fatalf("context contribution: %+v %v", audit, err)
		}
		if runContext.Err() != nil {
			t.Fatal("background work inherited a completed callback's cancellation")
		}
	}
	if contexts.Load() != 1 {
		t.Fatalf("RunContext called %d times", contexts.Load())
	}
	_ = h.Close()
	if runContext.Err() == nil || closed.Load() != 1 {
		t.Fatal("run context or extension outlived Close")
	}
}
