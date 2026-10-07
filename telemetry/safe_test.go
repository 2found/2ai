package telemetry

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

func safeFixture(t *testing.T, options SafeObserverOptions) (*SafeObserver, context.Context) {
	t.Helper()
	o, err := NewSafeObserver(options)
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := WithSafeObserver(context.Background(), o, SafeExecution{ExecutionID: "execution-1"})
	if err != nil {
		t.Fatal(err)
	}
	return o, ctx
}

func TestSafeLiveSpansPreservePrivateJSONAndLateChildren(t *testing.T) {
	o, ctx := safeFixture(t, SafeObserverOptions{})
	private := NewInMemory()
	parentStarted, finished := make(chan Context, 1), make(chan error, 1)
	go func() {
		finished <- SafeContext(private.Context, ctx).StartSpan(SpanOptions{Name: "agentray.agent.run"}, func(root *Span) error {
			parentStarted <- root.Context()
			return root.StartSpan(SpanOptions{Name: "agentray.ai.request"}, func(child *Span) error {
				return child.StartSpan(SpanOptions{Name: "agentray.tool.execute"}, func(*Span) error { return nil })
			})
		})
	}()
	parent := <-parentStarted
	if err := <-finished; err != nil {
		t.Fatal(err)
	}
	records := o.Drain(256)
	if len(records) != 3 || records[0].SpanKind != "tool" || records[1].SpanKind != "model" || records[2].SpanKind != "agent" {
		t.Fatalf("settlement order: %+v", records)
	}
	if records[0].Duration > records[2].Duration || records[0].ParentSpanID == nil || *records[0].ParentSpanID != *records[1].SpanID {
		t.Fatal("span timings/parentage lost")
	}
	before, _ := json.Marshal(private.GetSpans())
	if strings.Contains(string(before), "started_at") || strings.Contains(string(before), "schema_version") || len(private.GetSpanTimings()) != 3 {
		t.Fatal("Pi JSON changed or timings missing")
	}
	// A safe observation may still settle after the private parent's callback.
	if err := parent.StartSpan(SpanOptions{Name: "agentray.ai.request"}, func(*Span) error { return nil }); err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(private.GetSpans())
	if string(before) != string(after) || len(o.Drain(10)) != 1 {
		t.Fatal("late safe child reopened private recorder")
	}
}

func TestSafeChildVisibleBeforeRootReturnAndOwnTiming(t *testing.T) {
	o, ctx := safeFixture(t, SafeObserverOptions{})
	childDone, rootRelease, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	secondStarted, secondRelease := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		_ = SafeContext(Context{}, ctx).StartSpan(SpanOptions{Name: "agentray.agent.run"}, func(root *Span) error {
			_ = root.StartSpan(SpanOptions{Name: "agentray.ai.request"}, func(span *Span) error {
				NewSafeCall(WithContext(ctx, span.Context())).StartAttempt(1, SafeLabels{}).Finish("completed", "none", SafeAccounting{})
				return nil
			})
			_ = root.StartSpan(SpanOptions{Name: "agentray.ai.request"}, func(span *Span) error {
				attempt := NewSafeCall(WithContext(ctx, span.Context())).StartAttempt(1, SafeLabels{})
				close(secondStarted)
				<-secondRelease
				attempt.Finish("completed", "none", SafeAccounting{})
				return nil
			})
			close(childDone)
			<-rootRelease
			return nil
		})
	}()
	<-secondStarted
	first := o.Drain(10)
	<-time.After(10 * time.Millisecond)
	close(secondRelease)
	<-childDone
	second := o.Drain(10)
	child := append(first, second...)
	if len(first) != 2 || len(second) != 2 || child[0].Kind != "attempt_settled" || child[1].Kind != "span_settled" || child[2].Duration < 10*time.Millisecond || child[2].SettledAt.Before(child[0].SettledAt) {
		close(rootRelease)
		<-done
		t.Fatal("child not live before root")
	}
	select {
	case <-done:
		t.Fatal("root returned before host gate")
	default:
	}
	close(rootRelease)
	<-done
	root := o.Drain(10)
	if len(root) != 1 || root[0].Duration < child[1].Duration+child[3].Duration || !root[0].SettledAt.After(child[3].SettledAt) || len(o.Drain(10)) != 0 {
		t.Fatal("root duplicates child or substitutes its latency")
	}
}

func TestSafeQueueFiniteAdmissionDrainCloseAndLoss(t *testing.T) {
	o, ctx := safeFixture(t, SafeObserverOptions{Capacity: 1})
	call := NewSafeCall(ctx)
	a := call.StartAttempt(1, SafeLabels{})
	a.Finish("completed", "none", SafeAccounting{})
	a.Finish("completed", "none", SafeAccounting{}) // at-most-once per producer
	call.StartAttempt(2, SafeLabels{}).Finish("failed", "unknown", SafeAccounting{})
	h := o.Health()
	if h.Admitted != 1 || h.Overflow != 1 || h.Queued != 1 || h.Complete {
		t.Fatalf("loss: %+v", h)
	}
	if len(o.Drain(0)) != 0 || len(o.Drain(10000)) != 1 {
		t.Fatal("drain bound/limit")
	}
	o.Close()
	call.StartAttempt(3, SafeLabels{}).Finish("completed", "none", SafeAccounting{})
	if h = o.Health(); h.ClosedAdmission != 1 || !h.Closed || h.Overflow != 1 {
		t.Fatalf("close: %+v", h)
	}
	oversize, ctx := safeFixture(t, SafeObserverOptions{RecordBytes: 1})
	NewSafeCall(ctx).StartAttempt(1, SafeLabels{}).Finish("completed", "none", SafeAccounting{})
	if h := oversize.Health(); h.Oversize != 1 || h.Admitted != 0 || h.Complete {
		t.Fatalf("oversize: %+v", h)
	}
	for _, opts := range []SafeObserverOptions{{Capacity: -1}, {Capacity: SafeQueueLimit + 1}, {RecordBytes: SafeRecordLimit + 1}} {
		if _, err := NewSafeObserver(opts); err == nil {
			t.Fatal("accepted unbounded observer")
		}
	}
}

func TestSafeConcurrentProducersDrainAndClose(t *testing.T) {
	o, ctx := safeFixture(t, SafeObserverOptions{Capacity: 16})
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			call := NewSafeCall(ctx)
			for n := 1; n <= 100; n++ {
				call.StartAttempt(n, SafeLabels{}).Finish("completed", "none", SafeAccounting{})
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range 100 {
			_ = o.Drain(3)
			_ = o.Health()
		}
		o.Close()
	}()
	wg.Wait()
	h := o.Health()
	if h.Admitted+h.Overflow+h.Oversize+h.ClosedAdmission+h.Contended != 800 || h.Queued > 16 {
		t.Fatalf("admission/loss accounting: %+v", h)
	}
}

func TestSafePrivacyAndImmutableCopies(t *testing.T) {
	o, ctx := safeFixture(t, SafeObserverOptions{})
	for _, bad := range []string{"", "secret\nheader", "https://private.example", "account@example", strings.Repeat("s", 129), "secret/response", "thinking text", "\x00"} {
		if _, err := ApproveModelLabels(bad, "model"); err == nil {
			t.Fatal("invalid label approved")
		}
		if _, err := ApproveToolLabel(bad); err == nil {
			t.Fatal("invalid tool approved")
		}
		if _, err := WithSafeObserver(ctx, o, SafeExecution{ExecutionID: bad}); err == nil {
			t.Fatal("invalid correlation accepted")
		}
	}
	// Lexically valid private names are not projected without explicit approval.
	canary := "private_canary"
	private := NewInMemory()
	cost := 0.0
	tokens := &SafeTokens{Input: 3}
	_ = SafeContext(private.Context, ctx).StartSpan(SpanOptions{Name: canary, Attributes: NewAttributes(Property{Name: "private", Value: NewObject(Property{Name: canary, Value: canary})})}, func(span *Span) error {
		span.AddEvent(canary, NewAttributes(Property{Name: "llm.trace", Value: canary}))
		NewSafeCall(WithContext(ctx, span.Context())).StartAttempt(1, SafeLabels{}).Finish(canary, canary, SafeAccounting{UsageObserved: true, PricingObserved: true, UsageSource: "explicit", PricingSource: "explicit", Tokens: tokens, KnownCostUSD: &cost, AttemptCoverage: "observed"})
		return errors.New(canary)
	})
	tokens.Input = 999
	cost = 999
	records := o.Drain(256)
	raw, err := json.Marshal(records)
	if err != nil || strings.Contains(string(raw), canary) || records[0].Accounting.Tokens.Input != 3 || *records[0].Accounting.KnownCostUSD != 0 {
		t.Fatalf("private data/reference leaked: %s %v", raw, err)
	}
	*records[0].SpanID = "mutated-consumer"
	if strings.Contains(string(mustJSON(t, private.GetSpans())), "mutated-consumer") {
		t.Fatal("safe consumer changed private recorder")
	}
}
func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestSafeMissingZeroUnpricedInvalidAndReconciliation(t *testing.T) {
	zero := 0.0
	for _, tc := range []struct {
		name         string
		a            SafeAccounting
		usage, price bool
		state        string
	}{
		{"missing", SafeAccounting{Tokens: &SafeTokens{}, KnownCostUSD: &zero}, false, false, "unknown"},
		{"observed-free", SafeAccounting{UsageObserved: true, PricingObserved: true, UsageSource: "explicit", PricingSource: "explicit", Tokens: &SafeTokens{}, KnownCostUSD: &zero}, true, true, "priced"},
		{"unpriced", SafeAccounting{UsageObserved: true, UsageSource: "wire", Tokens: &SafeTokens{Input: 2}, KnownCostUSD: &zero}, true, false, "unpriced"},
		{"negative", SafeAccounting{UsageObserved: true, UsageSource: "explicit", Tokens: &SafeTokens{Input: -1}}, false, false, "unknown"},
		{"infinite", SafeAccounting{UsageObserved: true, UsageSource: "explicit", Tokens: &SafeTokens{Output: math.Inf(1)}}, false, false, "unknown"},
		{"nan", SafeAccounting{UsageObserved: true, UsageSource: "explicit", Tokens: &SafeTokens{CacheRead: math.NaN()}}, false, false, "unknown"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := NormalizeSafeAccounting(tc.a)
			if got.UsageObserved != tc.usage || got.PricingObserved != tc.price || got.PriceState != tc.state || (!tc.usage && got.Tokens != nil) || (!tc.price && got.KnownCostUSD != nil) {
				t.Fatalf("availability: %+v", got)
			}
		})
	}
	o, ctx := safeFixture(t, SafeObserverOptions{})
	ReconcileSafe(ctx, SafeAccounting{Tokens: &SafeTokens{Input: 58}, KnownCostUSD: &zero})
	r := o.Drain(1)[0]
	if r.Kind != "reconciliation" || r.Reconciliation == nil || r.Reconciliation.Additive || r.Reconciliation.ReportedTokens.Input != 58 || r.Accounting.AttemptCoverage != "unavailable" || r.Accounting.Complete || r.AttemptID != nil {
		t.Fatalf("aggregate presented as attempt: %+v", r)
	}
}

func TestSafeFailedAdmissionCannotClaimAccounting(t *testing.T) {
	o, ctx := safeFixture(t, SafeObserverOptions{})
	cost := 1.0
	NewSafeCall(ctx).StartAttempt(1, SafeLabels{}).FinishAdmission("failed", "host_preparation", SafeAccounting{UsageObserved: true, PricingObserved: true, UsageSource: "explicit", PricingSource: "explicit", Tokens: &SafeTokens{Input: 1}, KnownCostUSD: &cost}, "", "", false)
	r := o.Drain(1)[0]
	if r.ProducerAdmitted || r.Accounting.UsageObserved || r.Accounting.PricingObserved || r.Accounting.Tokens != nil || r.Accounting.KnownCostUSD != nil {
		t.Fatal("nonexistent producer acquired usage", r)
	}
}

func TestSafeDetachedAndCoalescedOrigin(t *testing.T) {
	o, ctx := safeFixture(t, SafeObserverOptions{})
	detached := DetachSafeExecution(ctx, CategoryChild)
	NewSafeCall(detached).StartAttempt(1, SafeLabels{}).Finish("completed", "none", SafeAccounting{})
	origin := CaptureSafeOrigin(ctx)
	other, _ := WithSafeObserver(ctx, o, SafeExecution{ExecutionID: "other-execution"})
	coalesced := BindSafeOrigin(context.Background(), origin.Merge(CaptureSafeOrigin(other)), CategoryMemory)
	NewSafeCall(coalesced).StartAttempt(1, SafeLabels{}).Finish("completed", "none", SafeAccounting{})
	r := o.Drain(10)
	if len(r) != 2 || r[0].ExecutionID == "execution-1" || r[0].OriginExecutionID == nil || *r[0].OriginExecutionID != "execution-1" || r[1].OriginExecutionID != nil || r[1].ExecutionID == r[0].ExecutionID || r[1].Category != CategoryMemory {
		t.Fatalf("origins: %+v", r)
	}
	// Independent recorders using their local ID 1 cannot collide in safe scope.
	for range 2 {
		_ = SafeContext(NewInMemory().Context, ctx).StartSpan(SpanOptions{Name: "agentray.agent.run"}, func(*Span) error { return nil })
	}
	r = o.Drain(2)
	if *r[0].SpanID == *r[1].SpanID || r[0].ProducerEventID == r[1].ProducerEventID {
		t.Fatal("recorder-local IDs reused")
	}
}

func TestSafeObservationPreservesErrorAndPanicIdentity(t *testing.T) {
	_, ctx := safeFixture(t, SafeObserverOptions{})
	sentinel := errors.New("private error")
	value, err := StartSpan(SafeContext(Context{}, ctx), SpanOptions{}, func(*Span) (int, error) { return 7, sentinel })
	if value != 7 || err != sentinel {
		t.Fatal("safe observation changed result")
	}
	panicValue := &struct{ private string }{"private panic"}
	defer func() {
		if got := recover(); got != panicValue {
			t.Fatalf("panic identity changed: %v", got)
		}
	}()
	_ = SafeContext(Context{}, ctx).StartSpan(SpanOptions{}, func(*Span) error { panic(panicValue) })
}

func TestSafeRecordBoundsWithMaximumApprovedIDs(t *testing.T) {
	o, ctx := safeFixture(t, SafeObserverOptions{})
	labels, err := ApproveModelLabels(strings.Repeat("p", 128), strings.Repeat("m", 128))
	if err != nil {
		t.Fatal(err)
	}
	NewSafeCall(ctx).StartAttempt(1, labels).Finish("completed", "none", SafeAccounting{})
	records := o.Drain(1)
	if len(records) != 1 || len(mustJSON(t, records[0])) > SafeRecordLimit {
		t.Fatal("supported identifiers exceed record cap")
	}
}
