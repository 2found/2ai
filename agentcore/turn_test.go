package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/2found/2ai/ai"
)

type cancelCloseProvider struct{}

func (cancelCloseProvider) Name() string        { return "cancel-close" }
func (cancelCloseProvider) SupportsTools() bool { return true }
func (cancelCloseProvider) Chat(context.Context, ChatRequest) (ChatResponse, error) {
	return ChatResponse{}, errors.New("unexpected Chat call")
}
func (cancelCloseProvider) Stream(ctx context.Context, _ ChatRequest) (<-chan ChatDelta, error) {
	ch := make(chan ChatDelta)
	go func() {
		<-ctx.Done()
		close(ch)
	}()
	return ch, nil
}

func TestStreamTurnDoesNotTreatCancelledClosureAsSuccess(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := (&Agent{}).streamTurn(ctx, cancelCloseProvider{}, ChatRequest{}, func(StreamEvent) {}, nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("streamTurn error = %v, want context.Canceled", err)
	}
}

type terminalThenCancelProvider struct{ cancel context.CancelFunc }

func (terminalThenCancelProvider) Name() string        { return "terminal-then-cancel" }
func (terminalThenCancelProvider) SupportsTools() bool { return true }
func (terminalThenCancelProvider) Chat(context.Context, ChatRequest) (ChatResponse, error) {
	return ChatResponse{}, errors.New("unexpected Chat call")
}
func (p terminalThenCancelProvider) Stream(context.Context, ChatRequest) (<-chan ChatDelta, error) {
	ch := make(chan ChatDelta, 2)
	ch <- ChatDelta{ContentDelta: "finished"}
	ch <- ChatDelta{Done: true, StopReason: "stop"}
	close(ch)
	p.cancel()
	return ch, nil
}

func TestStreamTurnTerminalEventWinsCancellationRace(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	resp, err := (&Agent{}).streamTurn(ctx, terminalThenCancelProvider{cancel: cancel}, ChatRequest{}, func(StreamEvent) {}, nil, nil)
	if err != nil {
		t.Fatalf("terminal response was discarded after cancellation: %v", err)
	}
	if resp.Message.Content != "finished" || resp.StopReason != "stop" {
		t.Fatalf("terminal response = %+v", resp)
	}
}

func TestStreamTurnPreservesReasoningWithoutRenderingIt(t *testing.T) {
	block := ReasoningBlock{
		Type: ReasoningBlockThinking, Text: "private replay text", Signature: "sig", ReplayScope: "anthropic:scope",
	}
	p := NewFauxProvider(ChatResponse{Message: Message{
		Role: RoleAssistant, Content: "visible", ReasoningBlocks: []ReasoningBlock{block},
	}, StopReason: "stop"})
	var visible string
	resp, err := (&Agent{}).streamTurn(context.Background(), p, ChatRequest{}, func(ev StreamEvent) {
		if ev.Type == StreamToken {
			visible += ev.Token
		}
	}, nil, nil)
	if err != nil {
		t.Fatalf("streamTurn: %v", err)
	}
	if visible != "visible" {
		t.Fatalf("visible stream = %q, want only assistant content", visible)
	}
	if len(resp.Message.ReasoningBlocks) != 1 || resp.Message.ReasoningBlocks[0] != block {
		t.Fatalf("reasoning block was not preserved: %+v", resp.Message.ReasoningBlocks)
	}
}

// providerErr builds a ProviderError with a real *http.Response so the test
// exercises the same construction path a provider uses.
func providerErr(status int, message string) *ProviderError {
	rec := httptest.NewRecorder()
	rec.WriteHeader(status)
	return NewProviderError("test", rec.Result(), message)
}

// Cloudflare answers from the edge with a 52x when the *origin* is what failed,
// so the edge is healthy and the failure is usually brief. Treating those as
// permanent is not academic: the idle-game benchmark took a 521 and gave up
// without spending a single retry.
func TestIsRetryable_CloudflareOriginErrors(t *testing.T) {
	for _, status := range []int{520, 521, 522, 523, 524} {
		if !IsRetryable(providerErr(status, "error code: "+http.StatusText(status))) {
			t.Errorf("status %d classified as permanent; Cloudflare origin errors are transient", status)
		}
	}
	// 525/526 are TLS handshake failures between edge and origin — a
	// misconfiguration, not a blip, so they must stay non-retryable.
	for _, status := range []int{525, 526} {
		if IsRetryable(providerErr(status, "")) {
			t.Errorf("status %d classified as transient; a TLS misconfiguration will not fix itself", status)
		}
	}
}

// Anthropic signals overload with 529, which is outside the standard 5xx set a
// status switch usually covers.
func TestIsRetryable_AnthropicOverloaded(t *testing.T) {
	if !IsRetryable(providerErr(529, `{"type":"overloaded_error"}`)) {
		t.Fatal("529 overloaded_error classified as permanent")
	}
}

// A spent quota and an ordinary throttle both arrive as 429. Retrying the former
// burns the whole ladder on something only a human can fix, and delays the clear
// error the operator needs to see.
func TestIsRetryable_ExhaustedQuotaOutranksRetryableStatus(t *testing.T) {
	permanent := []string{
		`{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}`,
		"429 quota exceeded",
		"Monthly usage limit reached",
		"your available balance is too low",
		"billing hard limit reached",
	}
	for _, msg := range permanent {
		if IsRetryable(providerErr(http.StatusTooManyRequests, msg)) {
			t.Errorf("account limit %q classified as retryable", msg)
		}
	}
	// A plain throttle with no quota wording stays retryable — that is the case
	// the 429 rung exists for.
	if !IsRetryable(providerErr(http.StatusTooManyRequests, "rate limit exceeded, please slow down")) {
		t.Fatal("an ordinary 429 throttle is no longer retryable")
	}
}

// Some gateways wrap an upstream failure in a 200 or a 400, where the status
// describes their own handshake rather than the outcome. The message is then the
// only signal there is.
func TestIsRetryable_MessageFallbackWhenStatusIsUnhelpful(t *testing.T) {
	transient := []string{
		"Provider returned error",
		"upstream connection reset by peer",
		"stream ended before message_stop",
		"truncated SSE response: no content, tool calls, or finish_reason",
		"the model is overloaded, try again",
	}
	for _, msg := range transient {
		if !IsRetryable(providerErr(http.StatusOK, msg)) {
			t.Errorf("transient wording %q classified as permanent", msg)
		}
	}

	// A genuine client error must not be rescued by the fallback: retrying a
	// malformed request reproduces it exactly.
	permanent := []string{
		"invalid_request_error: messages[0].content is required",
		"model `does-not-exist` not found",
		"authentication_error: invalid x-api-key",
	}
	for _, msg := range permanent {
		if IsRetryable(providerErr(http.StatusBadRequest, msg)) {
			t.Errorf("client error %q classified as retryable", msg)
		}
	}
}

// The existing contract must not regress: a transport failure has no response at
// all, and that is the case the whole Status-0 convention exists to carry.
func TestIsRetryable_TransportAndStandardStatusesUnchanged(t *testing.T) {
	if !IsRetryable(NewProviderError("test", nil, "connection severed")) {
		t.Fatal("a Status 0 transport failure is no longer retryable")
	}
	for _, status := range []int{408, 429, 500, 502, 503, 504} {
		if !IsRetryable(providerErr(status, "")) {
			t.Errorf("status %d is no longer retryable", status)
		}
	}
	for _, status := range []int{400, 401, 403, 404, 422} {
		if IsRetryable(providerErr(status, "")) {
			t.Errorf("client error %d became retryable", status)
		}
	}
	if IsRetryable(nil) {
		t.Fatal("nil error is retryable")
	}
	if IsRetryable(errors.New("some unclassified failure")) {
		t.Fatal("a bare error became retryable; providers must build ProviderError")
	}
}

// The quota override has to survive wrapping, since the loop sees errors that
// have been annotated with turn context on the way up.
func TestIsRetryable_ClassifiesThroughWrappedErrors(t *testing.T) {
	wrapped := errors.Join(errors.New("provider chat (turn 4)"),
		providerErr(http.StatusTooManyRequests, "insufficient_quota"))
	if IsRetryable(wrapped) {
		t.Fatal("wrapped quota error classified as retryable")
	}
	wrapped = errors.Join(errors.New("provider chat (turn 4)"), providerErr(521, ""))
	if !IsRetryable(wrapped) {
		t.Fatal("wrapped Cloudflare 521 classified as permanent")
	}
}

// --- #1: cache-aware accounting ---
//
// This file keeps the LOOP's half: that cache token counts accumulate across
// turns and that retry/escalation layer correctly. Pricing moved to
// ai/model_pricing.go, and the per-vendor usage normalization
// that feeds both moved to ai/ with the wire code.

// TestRunSumsCacheTokens verifies the loop accumulates cache tokens across turns
// into the run total, so a consumer sees honest cache accounting end-to-end.
func TestRunSumsCacheTokens(t *testing.T) {
	r1 := AssistantText("done")
	r1.Usage = Usage{InputTokens: 5, OutputTokens: 3, CacheReadTokens: 100, CacheWriteTokens: 20}
	agent, err := New(Config{NativeProvider: scriptedNativeProvider(r1), Model: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Usage.CacheReadTokens != 100 || res.Usage.CacheWriteTokens != 20 {
		t.Fatalf("run cache usage = read %d / write %d, want 100 / 20",
			res.Usage.CacheReadTokens, res.Usage.CacheWriteTokens)
	}
}

// --- #2: same-rung retry with backoff ---

// flakyNativeStream fails with a retryable ProviderError for the first failN
// attempts, then serves the script. It counts every attempt so a test can
// assert the retry took place on the same candidate.
func flakyNativeStream(failN int, status int, calls *int32, script ...ChatResponse) ai.StreamFn {
	return func(ctx context.Context, m json.RawMessage, v ai.TranscriptContext, o map[string]any) (*ai.AssistantMessageEventStream, error) {
		n := atomic.AddInt32(calls, 1)
		if n <= int32(failN) {
			return nil, &ProviderError{Provider: "flaky", Status: status}
		}
		return scriptedNativeStream(script...)(ctx, m, v, o)
	}
}

// fastRetry is a tiny backoff so retry tests don't sleep for real.
func fastRetry() *RetryPolicy {
	return &RetryPolicy{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond}
}

// TestSameRungRetrySucceedsAfterTransientError verifies a 503 on the same model
// is retried with backoff and recovers, without escalating or aborting.
func TestSameRungRetrySucceedsAfterTransientError(t *testing.T) {
	var calls int32
	agent, err := New(Config{NativeProvider: nativeCandidate("test", flakyNativeStream(2, http.StatusServiceUnavailable, &calls, AssistantText("recovered"))), Model: "test", Retry: fastRetry()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt should recover after retries: %v", err)
	}
	if res.Final != "recovered" {
		t.Fatalf("final = %q, want recovered", res.Final)
	}
	if got := atomic.LoadInt32(&calls); got != 3 {
		t.Fatalf("provider called %d times, want 3 (2 failures + 1 success)", got)
	}
}

// TestRetryExhaustionSurfacesError verifies that once the per-rung attempt budget
// is spent on a persistently failing model (and there is no ladder to escalate
// to), the error surfaces — the loop doesn't retry forever.
func TestRetryExhaustionSurfacesError(t *testing.T) {
	var calls int32
	agent, err := New(Config{NativeProvider: nativeCandidate("test", flakyNativeStream(99, http.StatusTooManyRequests, &calls)), Model: "test", Retry: fastRetry()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err == nil {
		t.Fatalf("a persistently failing provider must surface an error")
	}
	if got := atomic.LoadInt32(&calls); got != 3 {
		t.Fatalf("provider called %d times, want 3 (MaxAttempts)", got)
	}
}

// TestNonRetryableErrorSkipsRetry verifies a client error (400) is not retried —
// it can't be fixed by trying the same model again, so it surfaces on the first
// attempt.
func TestNonRetryableErrorSkipsRetry(t *testing.T) {
	var calls int32
	agent, err := New(Config{NativeProvider: nativeCandidate("test", flakyNativeStream(99, http.StatusBadRequest, &calls)), Model: "test", Retry: fastRetry()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err == nil {
		t.Fatalf("a 400 must surface as an error")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("provider called %d times, want 1 (no retry on non-retryable)", got)
	}
}

// TestRetryThenEscalate verifies the layering: a rung's retries are spent first,
// and only then does the loop escalate to the next rung, which succeeds.
func TestRetryThenEscalate(t *testing.T) {
	var badCalls int32
	bad := flakyNativeStream(99, http.StatusServiceUnavailable, &badCalls)
	ladder := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"rung1"}`), Stream: bad},
		{Model: json.RawMessage(`{"id":"rung2"}`), Stream: scriptedNativeStream(AssistantText("from rung 2"))},
	}}
	agent, err := New(Config{
		NativeProvider: ladder,
		Model:          "rung1",
		Retry:          fastRetry(),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("should escalate to a working rung: %v", err)
	}
	if res.Final != "from rung 2" {
		t.Fatalf("final = %q, want 'from rung 2'", res.Final)
	}
	if got := atomic.LoadInt32(&badCalls); got != 3 {
		t.Fatalf("rung 1 retried %d times, want 3 before escalating", got)
	}
}

// nativeErrorMessage builds an assistant error event carrying the failure text.
func nativeErrorMessage(text string, usage ai.Usage) *ai.Message {
	return &ai.Message{Role: "assistant", StopReason: "error", Content: ai.BlockContent(), ErrorMessage: &text, Usage: &usage}
}

// Once a token reaches the sink, retrying would concatenate a second answer to
// the first. The failed attempt also remains billable, so its partial usage must
// survive on the errored RunResult.
func TestStreamFailureAfterVisibleOutputIsNotRetriedOrEscalated(t *testing.T) {
	var primaryCalls, fallbackCalls int32
	failed := &ai.Message{Role: "assistant", StopReason: "error", Usage: &ai.Usage{Input: 11, Output: 1},
		Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "partial"}),
	}
	errText := "provider overloaded"
	failed.ErrorMessage = &errText
	// The attempt commits visible output, then dies: retrying would concatenate
	// a second answer, so the run must surface the error on the spot.
	primary := nativeAttempts(&primaryCalls, nativeEmit(failed))
	fallback := nativeAttempts(&fallbackCalls, nativeEmit(&ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "fallback should not run"})}))
	ladder := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"primary-model"}`), Stream: primary},
		{Model: json.RawMessage(`{"id":"fallback-model"}`), Stream: fallback},
	}}
	agent, err := New(Config{
		NativeProvider: ladder, Model: "primary-model", Retry: fastRetry(),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var visible string
	res, err := agent.PromptStream(context.Background(), "go", func(ev StreamEvent) {
		if ev.Type == StreamToken {
			visible += ev.Token
		}
	})
	if err == nil || !strings.Contains(err.Error(), "overloaded") {
		t.Fatalf("err = %v, want the provider failure", err)
	}
	if visible != "partial" {
		t.Fatalf("visible output = %q, want one copy of the committed prefix", visible)
	}
	if got := atomic.LoadInt32(&primaryCalls); got != 1 {
		t.Fatalf("primary attempts = %d, want 1 after output committed", got)
	}
	if got := atomic.LoadInt32(&fallbackCalls); got != 0 {
		t.Fatalf("fallback attempts = %d, want 0 after output committed", got)
	}
	if res.Usage.InputTokens != 11 || res.Usage.OutputTokens != 1 {
		t.Fatalf("partial usage = %+v, want input=11 output=1", res.Usage)
	}
}

// A failure before the first token is still replay-safe and should retain the
// existing same-rung retry behavior.
func TestStreamFailureBeforeVisibleOutputStillRetries(t *testing.T) {
	var primaryCalls int32
	recovered := &ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "recovered"})}
	primary := nativeAttempts(&primaryCalls,
		failingNativeStream(&ProviderError{Provider: "primary", Status: http.StatusServiceUnavailable}),
		nativeEmit(recovered))
	agent, err := New(Config{NativeProvider: nativeCandidate("test", primary), Model: "test", Retry: fastRetry()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var visible string
	res, err := agent.PromptStream(context.Background(), "go", func(ev StreamEvent) {
		if ev.Type == StreamToken {
			visible += ev.Token
		}
	})
	if err != nil {
		t.Fatalf("PromptStream should recover before output commits: %v", err)
	}
	if res.Final != "recovered" || visible != "recovered" {
		t.Fatalf("final=%q visible=%q, want recovered", res.Final, visible)
	}
	if got := atomic.LoadInt32(&primaryCalls); got != 2 {
		t.Fatalf("primary attempts = %d, want 2", got)
	}
}

// TestRetryClassification spot-checks the retryable/non-retryable split that
// gates same-rung retry.
func TestRetryClassification(t *testing.T) {
	cases := []struct {
		err  error
		want bool
	}{
		{&ProviderError{Status: http.StatusTooManyRequests}, true},
		{&ProviderError{Status: http.StatusServiceUnavailable}, true},
		{&ProviderError{Status: http.StatusBadGateway}, true},
		{&ProviderError{Status: 0}, true}, // transport failure
		{&ProviderError{Status: http.StatusBadRequest}, false},
		{&ProviderError{Status: http.StatusUnauthorized}, false},
		{errors.New("plain"), false},
		{nil, false},
	}
	for _, c := range cases {
		if got := IsRetryable(c.err); got != c.want {
			t.Fatalf("IsRetryable(%v) = %v, want %v", c.err, got, c.want)
		}
	}
}

// TestParseRetryAfter verifies the Retry-After parser handles the delay-seconds
// form (the common case) and ignores garbage.
// Usage on a streamed turn, when the provider does not hand it over all at once.
//
// Both providers in this module happen to accumulate internally and report
// everything on the Done delta, which is why a turn that silently billed zero
// went unnoticed for so long: the only way to see it is a provider that reports
// the way the wire actually does. Anthropic states input tokens on
// message_start, before any output exists; OpenAI emits a usage-only chunk
// AFTER the chunk carrying finish_reason. A loop that reads usage off Done alone
// scores the first as zero-output and the second as zero-everything, and the
// only visible symptom is a budget gate that never trips.

// pieceProvider streams one fixed answer with usage split across deltas.
type pieceProvider struct{ deltas []ChatDelta }

func (*pieceProvider) Name() string        { return "pieces" }
func (*pieceProvider) SupportsTools() bool { return true }

func (*pieceProvider) Chat(context.Context, ChatRequest) (ChatResponse, error) {
	return ChatResponse{}, nil
}

func (p *pieceProvider) Stream(context.Context, ChatRequest) (<-chan ChatDelta, error) {
	ch := make(chan ChatDelta, len(p.deltas))
	for _, d := range p.deltas {
		ch <- d
	}
	close(ch)
	return ch, nil
}

func streamOnce(t *testing.T, deltas ...ChatDelta) ChatResponse {
	t.Helper()
	a := &Agent{}
	resp, err := a.streamTurn(context.Background(), &pieceProvider{deltas: deltas},
		ChatRequest{}, func(StreamEvent) {}, nil, nil)
	if err != nil {
		t.Fatalf("streamTurn: %v", err)
	}
	return resp
}

// The Anthropic shape: input tokens arrive first, output tokens only at the end.
func TestStreamedUsageSurvivesAnEarlyReport(t *testing.T) {
	resp := streamOnce(t,
		ChatDelta{Usage: Usage{InputTokens: 1200}},
		ChatDelta{ContentDelta: "hello"},
		ChatDelta{Done: true, StopReason: "stop", Usage: Usage{OutputTokens: 40}},
	)
	if resp.Usage.InputTokens != 1200 {
		t.Fatalf("input tokens reported before the first output token were dropped: got %d, want 1200. "+
			"The turn is then billed as if its prompt were free, and the prompt is the expensive half",
			resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 40 {
		t.Fatalf("output tokens = %d, want 40", resp.Usage.OutputTokens)
	}
	if resp.StopReason != "stop" {
		t.Fatalf("stop reason = %q, want %q", resp.StopReason, "stop")
	}
	if resp.Message.Content != "hello" {
		t.Fatalf("content = %q, want %q", resp.Message.Content, "hello")
	}
}

// The OpenAI shape: the terminal chunk carries no usage at all, and a usage-only
// chunk follows it.
func TestStreamedUsageSurvivesAReportAfterDone(t *testing.T) {
	resp := streamOnce(t,
		ChatDelta{ContentDelta: "hi"},
		ChatDelta{Done: true, StopReason: "stop"},
		ChatDelta{Usage: Usage{InputTokens: 900, OutputTokens: 12, CacheReadTokens: 800}},
	)
	want := Usage{InputTokens: 900, OutputTokens: 12, CacheReadTokens: 800}
	if resp.Usage != want {
		t.Fatalf("usage = %+v, want %+v: a usage-only chunk after the terminal one is how OpenAI "+
			"reports with stream_options.include_usage, so ignoring it bills the whole turn at zero",
			resp.Usage, want)
	}
	if resp.StopReason != "stop" {
		t.Fatalf("a later delta clobbered the terminal stop reason: got %q", resp.StopReason)
	}
}

// Reports within a turn are running totals, not increments, so restating a field
// must not double it. This is the reason the merge overwrites rather than adds.
func TestStreamedUsageIsNotDoubleCountedWhenRestated(t *testing.T) {
	resp := streamOnce(t,
		ChatDelta{Usage: Usage{InputTokens: 500}},
		ChatDelta{ContentDelta: "x"},
		ChatDelta{Usage: Usage{InputTokens: 500, OutputTokens: 7}},
		ChatDelta{Done: true, StopReason: "stop", Usage: Usage{InputTokens: 500, OutputTokens: 9}},
	)
	if resp.Usage.InputTokens != 500 {
		t.Fatalf("input tokens = %d, want 500: a restated running total was summed instead of "+
			"replaced, which over-bills every turn a provider reports more than once",
			resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 9 {
		t.Fatalf("output tokens = %d, want 9 (the newest total)", resp.Usage.OutputTokens)
	}
}

// A provider that reports once, on Done — the case both of this module's
// providers produce — must be unaffected by any of the above.
func TestStreamedUsageOnDoneAloneStillWorks(t *testing.T) {
	want := Usage{InputTokens: 10, OutputTokens: 20, CacheWriteTokens: 5, CostUSD: 0.25}
	resp := streamOnce(t,
		ChatDelta{ContentDelta: "ok"},
		ChatDelta{Done: true, StopReason: "stop", Usage: want},
	)
	if resp.Usage != want {
		t.Fatalf("usage = %+v, want %+v", resp.Usage, want)
	}
}
