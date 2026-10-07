package ai

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/ai/protocol"
	"github.com/2found/2ai/telemetry"
)

func safeAIContext(t *testing.T) (*telemetry.SafeObserver, context.Context) {
	t.Helper()
	o, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := telemetry.WithSafeObserver(context.Background(), o, telemetry.SafeExecution{ExecutionID: "ai-execution"})
	if err != nil {
		t.Fatal(err)
	}
	return o, ctx
}
func explicitUsage(input, output, read, write, cost float64, priced bool) *Usage {
	return &Usage{Input: input, Output: output, CacheRead: read, CacheWrite: write, Cost: UsageCost{Total: cost}, Observation: UsageObservation{UsageObserved: true, PricingObserved: priced, UsageSource: "explicit", PricingSource: "explicit"}}
}
func safeAttempts(o *telemetry.SafeObserver) []telemetry.SafeRecord {
	var result []telemetry.SafeRecord
	for _, r := range o.Drain(256) {
		if r.Kind == "attempt_settled" {
			result = append(result, r)
		}
	}
	return result
}

func TestSafeRetryFallbackEightAttemptAccountingOracle(t *testing.T) {
	o, ctx := safeAIContext(t)
	labels, _ := telemetry.ApproveModelLabels("configured-provider", "configured-model")
	temporary := &protocol.ProviderError{Provider: "configured-provider", Status: 503, Message: "private raw failure"}
	mainMessages := []*Message{
		{Role: "assistant", StopReason: "error", Usage: explicitUsage(2, 1, 1, 0, .002, true)},
		{Role: "assistant", StopReason: "error", Usage: explicitUsage(3, 1, 0, 1, .003, true)},
		{Role: "assistant", StopReason: "stop", Content: TextContent("private response"), Usage: explicitUsage(5, 2, 2, 1, .005, true)},
	}
	p := FallbackProvider{Candidates: []FallbackCandidate{{SafeLabels: labels}, {SafeLabels: labels}}, Retry: protocol.RetryPolicy{MaxAttempts: 2}, Wait: func(context.Context, time.Duration) error { return nil }}
	calls := 0
	final, err := p.Run(ctx, NewAssistantMessageEventStream(), FallbackRequest{Candidates: 2, Open: func(ctx context.Context, index, number int) (*AssistantMessageEventStream, error) {
		if calls < 2 && (index != 0 || number != calls+1) || calls == 2 && (index != 1 || number != 1) {
			t.Fatal("rung ordinals changed")
		}
		m := mainMessages[calls]
		calls++
		if m.StopReason == "error" {
			recordNativeFailure(ctx, "configured-provider", temporary, false)
			return attemptFixture(AssistantMessageEvent{Type: "error", Error: m})(ctx)
		}
		return attemptFixture(AssistantMessageEvent{Type: "done", Message: m})(ctx)
	}})
	if err != nil || final.Message() != mainMessages[2] || calls != 3 {
		t.Fatal("retry changed outcome", calls, err)
	}
	for _, tc := range []struct {
		category telemetry.SafeCategory
		usage    *Usage
	}{
		{telemetry.CategoryCompaction, explicitUsage(7, 3, 1, 1, .007, true)},
		{telemetry.CategoryAdvisor, explicitUsage(11, 4, 0, 2, .011, true)},
		{telemetry.CategoryMemory, explicitUsage(13, 5, 2, 0, 0, false)},
		{telemetry.CategoryChild, explicitUsage(17, 6, 3, 1, 0, true)},
		{telemetry.CategoryMain, nil},
	} {
		message := &Message{Role: "assistant", StopReason: "stop", Usage: tc.usage}
		_, err := p.Run(telemetry.WithSafeCategory(ctx, tc.category), NewAssistantMessageEventStream(), FallbackRequest{Candidates: 1, Open: func(ctx context.Context, _, _ int) (*AssistantMessageEventStream, error) {
			return attemptFixture(AssistantMessageEvent{Type: "done", Message: message})(ctx)
		}})
		if err != nil {
			t.Fatal(err)
		}
	}
	zero := 0.0
	telemetry.ReconcileSafe(ctx, telemetry.SafeAccounting{Tokens: &telemetry.SafeTokens{Input: 58, Output: 22, CacheRead: 9, CacheWrite: 6}, KnownCostUSD: &zero})
	records := o.Drain(256)
	seenEvents, seenAttempts := map[string]bool{}, map[string]bool{}
	var totals telemetry.SafeTokens
	knownCost := 0.0
	observed, priced, unpriced, missing, attempts := 0, 0, 0, 0, 0
	for _, r := range records {
		if seenEvents[r.ProducerEventID] {
			t.Fatal("duplicate producer event")
		}
		seenEvents[r.ProducerEventID] = true
		if r.Kind != "attempt_settled" {
			if r.Accounting.AttemptCoverage != "unavailable" || r.Reconciliation == nil || r.Reconciliation.Additive {
				t.Fatal("additive root aggregate")
			}
			continue
		}
		attempts++
		if r.AttemptID == nil || seenAttempts[*r.AttemptID] || r.ProviderID == nil || *r.ProviderID != "configured-provider" || r.ModelID == nil || r.Duration < 0 || r.SettledAt.Before(r.StartedAt) {
			t.Fatalf("identity/timing: %+v", r)
		}
		seenAttempts[*r.AttemptID] = true
		if attempts <= 3 {
			if *r.CallID != *records[0].CallID {
				t.Fatal("fallback got another logical call")
			}
		}
		a := r.Accounting
		if !a.UsageObserved {
			missing++
			if a.Tokens != nil || a.KnownCostUSD != nil {
				t.Fatal("missing usage became zero")
			}
			continue
		}
		observed++
		totals.Input += a.Tokens.Input
		totals.Output += a.Tokens.Output
		totals.CacheRead += a.Tokens.CacheRead
		totals.CacheWrite += a.Tokens.CacheWrite
		if a.PriceState == "priced" {
			priced++
			knownCost += *a.KnownCostUSD
		} else {
			unpriced++
			if a.KnownCostUSD != nil {
				t.Fatal("unknown price became free")
			}
		}
	}
	if attempts != 8 || observed != 7 || priced != 6 || unpriced != 1 || missing != 1 || totals != (telemetry.SafeTokens{Input: 58, Output: 22, CacheRead: 9, CacheWrite: 6}) || math.Abs(knownCost-.028) > 1e-12 {
		t.Fatalf("oracle: attempts=%d usage=%d priced=%d unpriced=%d missing=%d totals=%+v cost=%g", attempts, observed, priced, unpriced, missing, totals, knownCost)
	}
	if len(o.Drain(256)) != 0 {
		t.Fatal("drain re-exported attempts")
	}
	raw, _ := json.Marshal(records)
	if strings.Contains(string(raw), "private") {
		t.Fatal("private content leaked")
	}
	t.Logf("AC-TELEMETRY safe eight-attempt actual JSON: %s", raw)
}

func TestSafeCancellationAndEarlyHostFailureWaitForProducer(t *testing.T) {
	for _, mode := range []string{"cancel", "partial-failure", "commit-failure", "observer-failure", "prepare-failure", "cancel-before-open", "cancel-backoff", "terminal-before-end"} {
		t.Run(mode, func(t *testing.T) {
			o, base := safeAIContext(t)
			ctx, cancel := context.WithCancel(base)
			defer cancel()
			message := &Message{Role: "assistant", StopReason: "error", Content: TextContent("private-partial-response"), Usage: explicitUsage(19, 7, 2, 1, 0, false)}
			typed := &protocol.ProviderError{Provider: "p", Status: 503, Message: "private-provider-error"}
			sentinel := errors.New("private-host-error")
			opened, visible, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan error, 1)
			calls := 0
			p := FallbackProvider{Retry: protocol.RetryPolicy{MaxAttempts: 2}, Wait: func(context.Context, time.Duration) error { cancel(); return ctx.Err() }}
			if mode == "cancel-before-open" {
				cancel()
			}
			go func() {
				_, err := p.Run(ctx, NewAssistantMessageEventStream(), FallbackRequest{Candidates: 2, Open: func(attemptCtx context.Context, index, number int) (*AssistantMessageEventStream, error) {
					calls++
					close(opened)
					if mode == "prepare-failure" {
						return nil, &PreparationError{Cause: sentinel}
					}
					if mode == "cancel-backoff" {
						return nil, typed
					}
					s := NewAssistantMessageEventStreamFor(attemptCtx)
					go func() {
						defer s.End()
						s.Push(AssistantMessageEvent{Type: "start", Partial: message})
						if mode == "cancel" || mode == "partial-failure" || mode == "commit-failure" {
							s.Push(AssistantMessageEvent{Type: "text_delta", Delta: "visible", Partial: message})
						}
						close(visible)
						if mode == "cancel" || mode == "commit-failure" {
							<-attemptCtx.Done()
						}
						if mode == "cancel" {
							message.StopReason = "aborted"
						}
						recordNativeFailure(attemptCtx, "p", typed, false)
						if mode == "terminal-before-end" {
							message.StopReason = "stop"
							s.Push(AssistantMessageEvent{Type: "done", Message: message})
						} else {
							s.Push(AssistantMessageEvent{Type: "error", Reason: message.StopReason, Error: message})
						}
						<-release
					}()
					return s, nil
				}, Commit: func(context.Context, int) error {
					if mode == "commit-failure" {
						return sentinel
					}
					return nil
				}, Observe: func(context.Context, int, FallbackAttempt) error {
					if mode == "observer-failure" {
						return sentinel
					}
					return nil
				}})
				finished <- err
			}()
			if mode == "cancel-before-open" {
				if err := <-finished; err != context.Canceled || calls != 0 || len(safeAttempts(o)) != 0 {
					t.Fatal("cancel opened fictitious producer", err, calls)
				}
				return
			}
			<-opened
			if mode == "prepare-failure" || mode == "cancel-backoff" {
				err := <-finished
				if mode == "prepare-failure" && !errors.Is(err, sentinel) || mode == "cancel-backoff" && err != context.Canceled {
					t.Fatal("original error lost", err)
				}
				r := safeAttempts(o)
				if len(r) != 1 || r[0].ProducerAdmitted || r[0].Accounting.UsageObserved || calls != 1 {
					t.Fatal("admission usage fabricated")
				}
				return
			}
			<-visible
			if mode == "cancel" {
				cancel()
			}
			// A published terminal or a host commit failure is not producer settlement.
			select {
			case err := <-finished:
				close(release)
				t.Fatal("returned before real producer settled", err)
			case <-time.After(10 * time.Millisecond):
			}
			if len(safeAttempts(o)) != 0 {
				close(release)
				<-finished
				t.Fatal("early settlement record")
			}
			close(release)
			err := <-finished
			if mode == "commit-failure" || mode == "observer-failure" {
				if err != sentinel {
					t.Fatal("host error identity lost", err)
				}
			} else if err != nil {
				t.Fatal("provider terminal changed", err)
			}
			r := safeAttempts(o)
			if len(r) != 1 || !r[0].ProducerAdmitted || calls != 1 || !r[0].Accounting.UsageObserved || *r[0].Accounting.Tokens != (telemetry.SafeTokens{Input: 19, Output: 7, CacheRead: 2, CacheWrite: 1}) {
				t.Fatalf("partial usage lost or retried: calls=%d records=%+v", calls, r)
			}
			if mode == "cancel" && r[0].Status != "cancelled" || (mode == "commit-failure" || mode == "observer-failure") && r[0].FailureKind != "host_callback" {
				t.Fatalf("structural outcome: %+v", r[0])
			}
		})
	}
}

func TestSafeNativeUsageProvenanceKeepsPiJSON(t *testing.T) {
	for _, raw := range []string{`{}`, `{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}`, `{"input":2,"output":3,"cacheRead":1,"cacheWrite":4}`} {
		var cost completionsCost
		if err := json.Unmarshal([]byte(raw), &cost); err != nil {
			t.Fatal(err)
		}
		var model completionsModel
		model.Cost = cost
		acc := newCompletionsAccumulator(model, OpenAICompletionsCompat{}, nil, NewAssistantMessageEventStream(), 1)
		usage := acc.parseUsage(json.RawMessage(`{"prompt_tokens":0,"completion_tokens":0}`))
		a := telemetry.NormalizeSafeAccounting(safeNativeAccounting(&Message{Usage: usage}))
		if !a.UsageObserved || a.PricingObserved != (raw != "{}") || (raw == "{}" && a.KnownCostUSD != nil) {
			t.Fatalf("wire zero/empty cost confused: %+v", a)
		}
		before, _ := json.Marshal(usage)
		var decoded Usage
		if err := json.Unmarshal(before, &decoded); err != nil {
			t.Fatal(err)
		}
		after, _ := json.Marshal(decoded)
		if !reflect.DeepEqual(before, after) || decoded.Observation.UsageObserved || strings.Contains(string(before), "Observation") {
			t.Fatal("observation changed transcript/checkpoint JSON")
		}
	}
	acc := newCompletionsAccumulator(completionsModel{}, OpenAICompletionsCompat{}, nil, NewAssistantMessageEventStream(), 1)
	for _, raw := range []string{`{}`, `{"prompt_tokens":-3,"completion_tokens":2}`, `{"prompt_tokens":2,"completion_tokens":0,"prompt_tokens_details":{"cached_tokens":3}}`, `{"prompt_tokens":2,"completion_tokens":null}`} {
		usage := acc.parseUsage(json.RawMessage(raw))
		if usage.Observation.UsageObserved {
			t.Fatal("missing/malformed usage observed", raw)
		}
	}
}

func TestSafeOAuthPhysicalRotationDoesNotCountWrapper(t *testing.T) {
	for _, nested := range []bool{false, true} {
		t.Run(map[bool]string{false: "direct", true: "fallback-wrapper"}[nested], func(t *testing.T) {
			o, ctx := safeAIContext(t)
			source := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "private-account-a", AccessToken: "private-token-a"}, {AccountID: "private-account-b", AccessToken: "private-token-b"}}}
			calls := 0
			open := func(ctx context.Context) (*AssistantMessageEventStream, error) {
				return nativeOAuthPoolStream(ctx, json.RawMessage(`{"id":"m","provider":"p"}`), NormalizeContext(Context{}), OpenAICompletionsStreamOptions{}, source, "fixture", func(ctx context.Context, _ json.RawMessage, _ TranscriptContext, _ OpenAICompletionsStreamOptions, _ OAuthToken) (*AssistantMessageEventStream, error) {
					calls++
					m := &Message{Role: "assistant", StopReason: "stop", Usage: explicitUsage(3, 0, 0, 0, 0, false)}
					if calls == 1 {
						m.StopReason = "error"
						recordNativeFailure(ctx, "p", &protocol.ProviderError{Provider: "p", Status: 401, Message: "private-auth-error"}, false)
						return attemptFixture(AssistantMessageEvent{Type: "error", Error: m})(ctx)
					}
					return attemptFixture(AssistantMessageEvent{Type: "done", Message: m})(ctx)
				})
			}
			if nested {
				_, err := (FallbackProvider{}).Run(ctx, NewAssistantMessageEventStream(), FallbackRequest{Candidates: 1, Open: func(ctx context.Context, _, _ int) (*AssistantMessageEventStream, error) { return open(ctx) }})
				if err != nil {
					t.Fatal(err)
				}
			} else {
				s, err := open(ctx)
				if err != nil {
					t.Fatal(err)
				}
				for {
					_, ok, err := s.Next(ctx)
					if err != nil {
						t.Fatal(err)
					}
					if !ok {
						break
					}
				}
				if err := s.WaitForEnd(ctx); err != nil {
					t.Fatal(err)
				}
			}
			r := safeAttempts(o)
			if calls != 2 || len(r) != 2 || *r[0].CallID != *r[1].CallID || *r[0].AttemptID == *r[1].AttemptID || r[0].Accounting.Tokens.Input+r[1].Accounting.Tokens.Input != 6 {
				t.Fatalf("rotation lost/duplicated physical producers: %+v", r)
			}
			raw, _ := json.Marshal(r)
			if strings.Contains(string(raw), "private-") {
				t.Fatal("account data leaked")
			}
		})
	}
}

func TestSafeProjectionOmitsEveryPrivateProducerPayload(t *testing.T) {
	o, ctx := safeAIContext(t)
	canaries := []string{"prompt-secret", "response-secret", "thinking-secret", "signature-secret", "args-secret", "result-secret", "error-secret", "credential-secret", "header-secret", "endpoint-secret", "account-secret", "provider-secret", "model-secret", "tool-secret", "attribute-secret", "response-model-secret"}
	signature, rawError, responseModel := canaries[3], canaries[6], canaries[15]
	message := &Message{Role: "assistant", StopReason: "stop", Provider: canaries[11], Model: canaries[12], ResponseModel: &responseModel, ErrorMessage: &rawError, Usage: explicitUsage(3, 0, 0, 0, 0, true), Content: BlockContent(
		ContentBlock{Type: "text", Text: canaries[1]}, ContentBlock{Type: "thinking", Thinking: canaries[2], ThinkingSignature: &signature}, ContentBlock{Type: "toolCall", Name: canaries[13], Arguments: json.RawMessage(`{"value":"args-secret"}`)},
	), Extra: map[string]json.RawMessage{"private": json.RawMessage(`{"nested":"attribute-secret"}`)}}
	labels, err := telemetry.ApproveModelLabels("public-provider", "public-model")
	if err != nil {
		t.Fatal(err)
	}
	p := FallbackProvider{Candidates: []FallbackCandidate{{Model: json.RawMessage(`{"id":"model-secret","provider":"provider-secret","baseUrl":"endpoint-secret","account":"account-secret"}`), SafeLabels: labels, Stream: func(ctx context.Context, _ json.RawMessage, _ TranscriptContext, _ map[string]any) (*AssistantMessageEventStream, error) {
		return attemptFixture(AssistantMessageEvent{Type: "done", Message: message})(ctx)
	}}}}
	s, err := p.Stream(ctx, nil, NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent(canaries[0])}, {Role: "toolResult", ToolName: canaries[13], Content: TextContent(canaries[5])}}}), map[string]any{"apiKey": canaries[7], "headers": map[string]string{"private": canaries[8]}})
	if err != nil {
		t.Fatal(err)
	}
	for {
		_, ok, err := s.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
	}
	if err := s.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
	records := safeAttempts(o)
	if len(records) != 1 || records[0].ProviderID != nil || records[0].ModelID != nil || records[0].LabelsState != "unavailable" {
		t.Fatalf("unapproved actual response model leaked: %+v", records)
	}
	raw, err := json.Marshal(struct {
		Records []telemetry.SafeRecord
		Health  telemetry.SafeHealth
	}{records, o.Health()})
	if err != nil {
		t.Fatal(err)
	}
	for _, canary := range canaries {
		if strings.Contains(string(raw), canary) {
			t.Fatal("private producer payload leaked", canary)
		}
	}
	records[0].Accounting.Tokens.Input = 999
	*records[0].Accounting.KnownCostUSD = 999
	if message.Usage.Input != 3 || message.Usage.Cost.Total != 0 || message.Content.Blocks.Get(1).ThinkingSignature != &signature {
		t.Fatal("safe consumer mutated original producer")
	}
}

type hostileSafeError struct{}

func (hostileSafeError) Error() string { panic("safe classification invoked Error") }
func (hostileSafeError) Is(error) bool { panic("safe classification invoked Is") }
func (hostileSafeError) As(any) bool   { panic("safe classification invoked As") }
func (hostileSafeError) Unwrap() error { panic("safe classification invoked Unwrap") }

func TestSafeFailureClassificationReadsNoArbitraryErrorMethods(t *testing.T) {
	if nativeFailureKind(hostileSafeError{}, false) != "unknown" {
		t.Fatal("custom error unexpectedly classified")
	}
	if nativeFailureKind(hostileSafeError{}, true) != "host_callback" {
		t.Fatal("host metadata lost")
	}
}

func TestSafeWireUsageObjectPresenceIsNotObservedZero(t *testing.T) {
	for _, tc := range []struct {
		raw      string
		observed bool
	}{
		{`{}`, false}, {`{"promptTokenCount":0}`, false},
		{`{"promptTokenCount":0,"candidatesTokenCount":0}`, true},
		{`{"promptTokenCount":3,"candidatesTokenCount":2,"thoughtsTokenCount":null}`, false},
	} {
		var usage nativeAntigravityUsage
		if err := json.Unmarshal([]byte(tc.raw), &usage); err != nil {
			t.Fatal(err)
		}
		if usage.observation.UsageObserved != tc.observed {
			t.Fatal("usage object initialized zero trusted", tc.raw, usage.observation)
		}
	}
	u := &Usage{Observation: UsageObservation{UsageObserved: true, UsageSource: "wire"}}
	observeOptionalWireUsage(u, map[string]json.RawMessage{"cache_read_input_tokens": json.RawMessage(`"malformed"`)}, "cache_read_input_tokens")
	if u.Observation.UsageObserved || u.Observation.UsageSource != "invalid" {
		t.Fatal("malformed cache observation trusted")
	}
}
