package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// fakeTokenSource is a TokenSource that hands out queued tokens and records
// every Report call so tests can see which account served and how it fared.
type fakeTokenSource struct {
	tokens    []OAuthToken
	acquireN  int
	acquireEr error
	reports   []fakeReport
}

type fakeReport struct {
	tok OAuthToken
	err error
}

func (s *fakeTokenSource) Acquire(context.Context) (OAuthToken, error) {
	if s.acquireEr != nil {
		return OAuthToken{}, s.acquireEr
	}
	tok := s.tokens[s.acquireN%len(s.tokens)]
	s.acquireN++
	return tok, nil
}

func (s *fakeTokenSource) Report(_ context.Context, tok OAuthToken, err error) {
	s.reports = append(s.reports, fakeReport{tok: tok, err: err})
}

// fakeInner is a minimal wire client that records the applied token and
// returns canned results.
type fakeInner struct {
	applied  OAuthToken
	chatErr  error
	chatFn   func(OAuthToken) (protocol.ChatResponse, error)
	streamCh chan protocol.ChatDelta
	streamFn func(OAuthToken) (<-chan protocol.ChatDelta, error)
}

func (f *fakeInner) applyOAuthToken(tok OAuthToken) protocol.LLMProvider {
	f.applied = tok
	return f
}
func (f *fakeInner) Name() string        { return "fake" }
func (f *fakeInner) SupportsTools() bool { return true }
func (f *fakeInner) Chat(context.Context, protocol.ChatRequest) (protocol.ChatResponse, error) {
	if f.chatFn != nil {
		return f.chatFn(f.applied)
	}
	return protocol.ChatResponse{Message: protocol.Message{Role: protocol.RoleAssistant, Content: "ok"}}, f.chatErr
}
func (f *fakeInner) Stream(context.Context, protocol.ChatRequest) (<-chan protocol.ChatDelta, error) {
	if f.streamFn != nil {
		return f.streamFn(f.applied)
	}
	return f.streamCh, nil
}

// A pooled provider must acquire a token per request, apply it to the wire
// client, and report the outcome back to the pool — that is the whole rotation
// contract.
func TestPooledProvider_AcquireApplyReport(t *testing.T) {
	src := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a1", AccessToken: "tok-1"}}}
	inner := &fakeInner{}
	p, err := newPooledProvider(VendorClaudeCode, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Chat(context.Background(), protocol.ChatRequest{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	if inner.applied.AccessToken != "tok-1" {
		t.Fatalf("applied token = %q, want tok-1", inner.applied.AccessToken)
	}
	if len(src.reports) != 1 || src.reports[0].err != nil || src.reports[0].tok.AccountID != "a1" {
		t.Fatalf("reports = %+v, want one success report for a1", src.reports)
	}
}

// An empty pool must surface as a provider error, not a request sent with no
// credential.
func TestPooledProvider_AcquireFailureIsProviderError(t *testing.T) {
	src := &fakeTokenSource{acquireEr: errors.New("no accounts")}
	p, err := newPooledProvider(VendorOpenAICodex, &fakeInner{}, src)
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Chat(context.Background(), protocol.ChatRequest{Model: "m"})
	var pe *protocol.ProviderError
	if !errors.As(err, &pe) || pe.Provider != VendorOpenAICodex {
		t.Fatalf("err = %v, want ProviderError for %q", err, VendorOpenAICodex)
	}
}

// A failed request must report the error against the account that served it so
// the pool rotates it out; the next call draws the next account.
func TestPooledProvider_ReportsErrorAndRotates(t *testing.T) {
	src := &fakeTokenSource{tokens: []OAuthToken{
		{AccountID: "a1", AccessToken: "tok-1"},
		{AccountID: "a2", AccessToken: "tok-2"},
	}}
	inner := &fakeInner{chatErr: protocol.NewProviderError("fake", nil, "boom")}
	p, err := newPooledProvider(VendorClaudeCode, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Chat(context.Background(), protocol.ChatRequest{Model: "m"}); err == nil {
		t.Fatal("want error")
	}
	if len(src.reports) != 1 || src.reports[0].err == nil || src.reports[0].tok.AccountID != "a1" {
		t.Fatalf("reports = %+v, want one error report for a1", src.reports)
	}
	inner.chatErr = nil
	if _, err := p.Chat(context.Background(), protocol.ChatRequest{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	if inner.applied.AccessToken != "tok-2" {
		t.Fatalf("second call applied %q, want tok-2 (rotation)", inner.applied.AccessToken)
	}
}

func TestPooledProvider_AuthFailureRetriesDistinctCredential(t *testing.T) {
	src := &fakeTokenSource{tokens: []OAuthToken{
		{AccountID: "a1", AccessToken: "tok-1"},
		{AccountID: "a2", AccessToken: "tok-2"},
	}}
	var wireCalls []string
	inner := &fakeInner{chatFn: func(tok OAuthToken) (protocol.ChatResponse, error) {
		wireCalls = append(wireCalls, tok.AccountID)
		if tok.AccountID == "a1" {
			return protocol.ChatResponse{}, &protocol.ProviderError{Provider: "fake", Status: http.StatusUnauthorized, Message: "expired"}
		}
		return protocol.ChatResponse{Message: protocol.Message{Role: protocol.RoleAssistant, Content: "recovered"}}, nil
	}}
	p, err := newPooledProvider(VendorClaudeCode, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Chat(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil || resp.Message.Content != "recovered" {
		t.Fatalf("Chat = (%q, %v), want recovered", resp.Message.Content, err)
	}
	if strings.Join(wireCalls, ",") != "a1,a2" {
		t.Fatalf("wire attempts = %v, want a1 then a2", wireCalls)
	}
	if len(src.reports) != 2 || src.reports[0].err == nil || src.reports[1].err != nil {
		t.Fatalf("reports = %+v, want auth failure then success", src.reports)
	}
}

func TestPooledProvider_AuthRetryStopsOnCredentialCycle(t *testing.T) {
	src := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a1", AccessToken: "same-token"}}}
	wireCalls := 0
	inner := &fakeInner{chatFn: func(OAuthToken) (protocol.ChatResponse, error) {
		wireCalls++
		return protocol.ChatResponse{}, &protocol.ProviderError{Provider: "fake", Status: http.StatusUnauthorized, Message: "expired"}
	}}
	p, err := newPooledProvider(VendorClaudeCode, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Chat(context.Background(), protocol.ChatRequest{Model: "m"})
	var providerErr *protocol.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Status != http.StatusUnauthorized {
		t.Fatalf("error = %v, want original 401", err)
	}
	if wireCalls != 1 || src.acquireN != 2 || len(src.reports) != 1 {
		t.Fatalf("wire calls=%d acquires=%d reports=%d, want 1/2/1", wireCalls, src.acquireN, len(src.reports))
	}
}

func TestPooledProvider_StreamAuthFailureBeforeOutputRotates(t *testing.T) {
	src := &fakeTokenSource{tokens: []OAuthToken{
		{AccountID: "a1", AccessToken: "tok-1"},
		{AccountID: "a2", AccessToken: "tok-2"},
	}}
	var wireCalls []string
	inner := &fakeInner{streamFn: func(tok OAuthToken) (<-chan protocol.ChatDelta, error) {
		wireCalls = append(wireCalls, tok.AccountID)
		ch := make(chan protocol.ChatDelta, 2)
		if tok.AccountID == "a1" {
			ch <- protocol.ChatDelta{Done: true, Err: &protocol.ProviderError{Provider: "fake", Status: http.StatusUnauthorized, Message: "expired"}}
		} else {
			ch <- protocol.ChatDelta{ContentDelta: "recovered"}
			ch <- protocol.ChatDelta{Done: true, StopReason: "stop"}
		}
		close(ch)
		return ch, nil
	}}
	p, err := newPooledProvider(VendorClaudeCode, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	var content string
	for delta := range ch {
		if delta.Err != nil {
			t.Fatalf("unexpected stream error: %v", delta.Err)
		}
		content += delta.ContentDelta
	}
	if content != "recovered" || strings.Join(wireCalls, ",") != "a1,a2" {
		t.Fatalf("content=%q attempts=%v, want recovered / a1,a2", content, wireCalls)
	}
	if len(src.reports) != 2 || src.reports[0].err == nil || src.reports[1].err != nil {
		t.Fatalf("reports = %+v, want auth failure then success", src.reports)
	}
}

func TestClaudeCode_InBandAuthenticationErrorRotatesCredential(t *testing.T) {
	var auth []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = append(auth, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "text/event-stream")
		if r.Header.Get("Authorization") == "Bearer tok-1" {
			_, _ = io.WriteString(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"expired\"}}\n\n")
			return
		}
		_, _ = io.WriteString(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"recovered\"}}\n\n"+
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"+
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer srv.Close()

	src := &fakeTokenSource{tokens: []OAuthToken{
		{AccountID: "a1", AccessToken: "tok-1"},
		{AccountID: "a2", AccessToken: "tok-2"},
	}}
	inner := NewAnthropicProvider("", srv.URL)
	inner.OAuth = true
	inner.HTTP, inner.StreamHTTP = srv.Client(), srv.Client()
	p, err := newPooledProvider(VendorClaudeCode, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var content string
	for delta := range ch {
		if delta.Err != nil {
			t.Fatalf("stream error: %v", delta.Err)
		}
		content += delta.ContentDelta
	}
	if content != "recovered" {
		t.Fatalf("stream content = %q, want recovered", content)
	}
	if strings.Join(auth, ",") != "Bearer tok-1,Bearer tok-2" {
		t.Fatalf("Authorization attempts = %v, want token rotation", auth)
	}
}

func TestPooledProvider_StreamCancellationIsVisible(t *testing.T) {
	src := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a1", AccessToken: "tok-1"}}}
	inner := &fakeInner{streamFn: func(OAuthToken) (<-chan protocol.ChatDelta, error) {
		return make(chan protocol.ChatDelta), nil
	}}
	p, err := newPooledProvider(VendorClaudeCode, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	ch, err := p.Stream(ctx, protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	var got error
	for delta := range ch {
		if delta.Err != nil {
			got = delta.Err
		}
	}
	if !errors.Is(got, context.Canceled) {
		t.Fatalf("stream cancellation = %v, want context.Canceled", got)
	}
}

func TestPooledProvider_StreamAuthFailureAfterOutputDoesNotReplay(t *testing.T) {
	src := &fakeTokenSource{tokens: []OAuthToken{
		{AccountID: "a1", AccessToken: "tok-1"},
		{AccountID: "a2", AccessToken: "tok-2"},
	}}
	wireCalls := 0
	inner := &fakeInner{streamFn: func(OAuthToken) (<-chan protocol.ChatDelta, error) {
		wireCalls++
		ch := make(chan protocol.ChatDelta, 2)
		ch <- protocol.ChatDelta{ContentDelta: "partial"}
		ch <- protocol.ChatDelta{Done: true, Err: &protocol.ProviderError{Provider: "fake", Status: http.StatusUnauthorized, Message: "expired"}}
		close(ch)
		return ch, nil
	}}
	p, err := newPooledProvider(VendorClaudeCode, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	var content string
	var streamErr error
	for delta := range ch {
		content += delta.ContentDelta
		if delta.Err != nil {
			streamErr = delta.Err
		}
	}
	if content != "partial" || streamErr == nil || wireCalls != 1 || src.acquireN != 1 {
		t.Fatalf("content=%q err=%v wire=%d acquires=%d, want partial/error/1/1", content, streamErr, wireCalls, src.acquireN)
	}
}

func TestPooledProvider_Concurrency403DoesNotBurnCredential(t *testing.T) {
	src := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a1", AccessToken: "tok-1"}}}
	inner := &fakeInner{chatErr: &protocol.ProviderError{
		Provider: "fake", Status: http.StatusForbidden, Message: "concurrent request limit reached",
	}}
	p, err := newPooledProvider(VendorClaudeCode, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Chat(context.Background(), protocol.ChatRequest{Model: "m"})
	if err == nil || !protocol.IsRetryable(err) {
		t.Fatalf("concurrency cap = %v, want retryable", err)
	}
	if len(src.reports) != 0 || src.acquireN != 1 {
		t.Fatalf("reports=%d acquires=%d, concurrency cap must not rotate", len(src.reports), src.acquireN)
	}
}

// A stream that fails mid-flight reports the first error delta against the
// serving account — a 429 arriving after headers still has to rotate it.
func TestPooledProvider_StreamReportsFirstErrorDelta(t *testing.T) {
	src := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a1", AccessToken: "tok-1"}}}
	ch := make(chan protocol.ChatDelta, 2)
	ch <- protocol.ChatDelta{ContentDelta: "hi"}
	ch <- protocol.ChatDelta{Done: true, Err: errors.New("rate limited")}
	close(ch)
	p, err := newPooledProvider(VendorClaudeCode, &fakeInner{streamCh: ch}, src)
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	var sawErr error
	for d := range got {
		if d.Err != nil {
			sawErr = d.Err
		}
	}
	if sawErr == nil {
		t.Fatal("stream error delta missing")
	}
	// A successful Stream start reports nothing; the error delta is the only report.
	var errReports int
	for _, r := range src.reports {
		if r.err != nil {
			errReports++
		}
	}
	if errReports != 1 {
		t.Fatalf("error reports = %d, want 1; reports=%+v", errReports, src.reports)
	}
}

// The pooled wrapper must not implement protocol.KeyUpdater: the loop's
// per-turn key refresh would overwrite the live OAuth token with the provider
// row's sentinel key.
func TestPooledProvider_IsNotAKeyUpdater(t *testing.T) {
	p, err := newPooledProvider(VendorClaudeCode, &fakeInner{}, &fakeTokenSource{})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := any(p).(protocol.KeyUpdater); ok {
		t.Fatal("pooledProvider must not implement KeyUpdater")
	}
}

// A wire client that cannot take an OAuth token must be refused at
// construction — otherwise the request goes out unauthenticated.
func TestPooledProvider_RefusesNonOAuthInner(t *testing.T) {
	if _, err := newPooledProvider(VendorClaudeCode, NewOpenAIProvider("k", "", DefaultCompat()), &fakeTokenSource{}); err == nil {
		t.Fatal("want error for inner without applyOAuthToken")
	}
	if _, err := newPooledProvider(VendorClaudeCode, &fakeInner{}, nil); err == nil {
		t.Fatal("want error for nil TokenSource")
	}
}

// The claude-code wire is the Messages API with the CLI fingerprint: Bearer
// auth instead of x-api-key, the claude-cli User-Agent, x-app: cli, the fixed
// subscription beta set, and the Claude Code identity block leading the system
// prompt.
func TestClaudeCode_HeadersAndSystemBlock(t *testing.T) {
	var gotAuth, gotAPIKey, gotUA, gotApp, gotBeta, gotAccept string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotAPIKey = r.Header.Get("x-api-key")
		gotUA = r.Header.Get("User-Agent")
		gotApp = r.Header.Get("x-app")
		gotBeta = r.Header.Get("anthropic-beta")
		gotAccept = r.Header.Get("Accept")
		data, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(data, &body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()

	src := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a1", AccessToken: "oauth-tok"}}}
	p, err := New(Spec{Vendor: "claude-code", BaseURL: srv.URL, HTTP: srv.Client(), TokenSource: src})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Chat(context.Background(), protocol.ChatRequest{
		Model: "claude-sonnet-4-6",
		Messages: []protocol.Message{
			{Role: protocol.RoleSystem, Content: "Be terse."},
			{Role: protocol.RoleUser, Content: "hi"},
		},
	}); err != nil {
		t.Fatal(err)
	}

	if gotAuth != "Bearer oauth-tok" {
		t.Fatalf("Authorization = %q, want Bearer oauth-tok", gotAuth)
	}
	if gotAPIKey != "" {
		t.Fatalf("x-api-key must not be sent in OAuth mode, got %q", gotAPIKey)
	}
	if gotUA != claudeCodeUserAgent {
		t.Fatalf("User-Agent = %q, want %q", gotUA, claudeCodeUserAgent)
	}
	if gotApp != "cli" {
		t.Fatalf("x-app = %q, want cli", gotApp)
	}
	for _, beta := range claudeCodeBetas {
		if !strings.Contains(gotBeta, beta) {
			t.Fatalf("anthropic-beta %q missing %q", gotBeta, beta)
		}
	}
	if gotAccept != "application/json" {
		t.Fatalf("Accept = %q, want application/json", gotAccept)
	}

	system, ok := body["system"].([]any)
	if !ok || len(system) == 0 {
		t.Fatalf("system = %v, want structured blocks", body["system"])
	}
	first, _ := system[0].(map[string]any)
	if first["text"] != claudeCodeSystemInstruction {
		t.Fatalf("first system block = %v, want Claude Code identity", first["text"])
	}
	if len(system) != 2 {
		t.Fatalf("system blocks = %d, want identity + caller text", len(system))
	}
	second, _ := system[1].(map[string]any)
	if second["text"] != "Be terse." {
		t.Fatalf("second system block = %v, want caller text", second["text"])
	}
}

// The Codex wire is the ChatGPT Responses backend: stream-only, Bearer auth
// plus the chatgpt-account-id/originator/version fingerprint, system text in
// `instructions`, and SSE events under response.*.
func TestCodex_StreamDecodesTextToolCallAndUsage(t *testing.T) {
	var gotPath, gotAccount, gotOriginator, gotVersion, gotBeta, gotAccept string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAccount = r.Header.Get("chatgpt-account-id")
		gotOriginator = r.Header.Get("originator")
		gotVersion = r.Header.Get("version")
		gotBeta = r.Header.Get("OpenAI-Beta")
		gotAccept = r.Header.Get("Accept")
		data, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(data, &body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w,
			"data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n"+
				"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}}\n\n"+
				"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"input_tokens_details\":{\"cached_tokens\":4}}}}\n\n")
	}))
	defer srv.Close()

	src := &fakeTokenSource{tokens: []OAuthToken{{
		AccountID: "a1", AccessToken: "codex-tok", ProviderAccountID: "acct-9",
	}}}
	p, err := New(Spec{Vendor: "openai-codex", BaseURL: srv.URL, HTTP: srv.Client(), TokenSource: src})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{
		Model: "gpt-5.6",
		Messages: []protocol.Message{
			{Role: protocol.RoleSystem, Content: "Be terse."},
			{Role: protocol.RoleUser, Content: "hi"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	var toolCall *protocol.ToolCall
	var done protocol.ChatDelta
	for d := range ch {
		if d.Err != nil {
			t.Fatalf("stream error: %v", d.Err)
		}
		text += d.ContentDelta
		if d.ToolCall != nil {
			toolCall = d.ToolCall
		}
		if d.Done {
			done = d
		}
	}

	if gotPath != "/codex/responses" {
		t.Fatalf("path = %q, want /codex/responses", gotPath)
	}
	if gotAccount != "acct-9" {
		t.Fatalf("chatgpt-account-id = %q, want acct-9", gotAccount)
	}
	if gotOriginator != "omp" || gotVersion != codexClientVersion || gotBeta != "responses=experimental" {
		t.Fatalf("client fingerprint wrong: originator=%q version=%q beta=%q", gotOriginator, gotVersion, gotBeta)
	}
	if gotAccept != "text/event-stream" {
		t.Fatalf("Accept = %q, want text/event-stream", gotAccept)
	}
	if body["stream"] != true || body["store"] != false {
		t.Fatalf("body must pin stream:true store:false: %v", body)
	}
	if body["instructions"] != "Be terse." {
		t.Fatalf("instructions = %v, want system text", body["instructions"])
	}
	if _, banned := body["temperature"]; banned {
		t.Fatal("temperature must not be sent to the Codex backend")
	}
	input, _ := body["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input = %v, want just the user message (system is hoisted)", body["input"])
	}

	if text != "Hello" {
		t.Fatalf("text = %q, want Hello", text)
	}
	if toolCall == nil || toolCall.ID != "call_1" || toolCall.Name != "read_file" || toolCall.Arguments != `{"path":"a.txt"}` {
		t.Fatalf("tool call = %+v", toolCall)
	}
	if done.StopReason != "completed" {
		t.Fatalf("stop reason = %q, want completed", done.StopReason)
	}
	if done.Usage.InputTokens != 6 || done.Usage.OutputTokens != 5 || done.Usage.CacheReadTokens != 4 {
		t.Fatalf("usage = %+v, want in=6 out=5 cache=4", done.Usage)
	}
}

// The Antigravity wire wraps generateContent in the agent envelope: project,
// requestId, labels, and the VALIDATED tool mode; its SSE chunks carry
// candidates/usageMetadata under a "response" key.
func TestAntigravity_EnvelopeAndStreamDecode(t *testing.T) {
	var gotPath, gotAuth, gotUA string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotUA = r.Header.Get("User-Agent")
		data, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(data, &body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w,
			"data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi\"},{\"thought\":true,\"text\":\"secret\"},{\"functionCall\":{\"name\":\"read_file\",\"args\":{\"path\":\"a.txt\"}}}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":3,\"cachedContentTokenCount\":2}}}\n\n")
	}))
	defer srv.Close()

	src := &fakeTokenSource{tokens: []OAuthToken{{
		AccountID: "a1", AccessToken: "ag-tok", ProjectID: "proj-7",
	}}}
	p, err := New(Spec{Vendor: "google-antigravity", BaseURL: srv.URL, HTTP: srv.Client(), TokenSource: src})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{
		Model: "claude-sonnet-4-6",
		Messages: []protocol.Message{
			{Role: protocol.RoleSystem, Content: "Be terse."},
			{Role: protocol.RoleUser, Content: "hi"},
		},
		Tools: []protocol.ToolSchema{{Name: "read_file", Description: "read", Parameters: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	var toolCall *protocol.ToolCall
	var done protocol.ChatDelta
	for d := range ch {
		if d.Err != nil {
			t.Fatalf("stream error: %v", d.Err)
		}
		text += d.ContentDelta
		if d.ToolCall != nil {
			toolCall = d.ToolCall
		}
		if d.Done {
			done = d
		}
	}

	if gotPath != "/v1internal:streamGenerateContent" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotAuth != "Bearer ag-tok" || gotUA != AntigravityUserAgent {
		t.Fatalf("auth/UA wrong: %q / %q", gotAuth, gotUA)
	}

	if body["project"] != "proj-7" || body["model"] != "claude-sonnet-4-6" {
		t.Fatalf("envelope project/model wrong: %v / %v", body["project"], body["model"])
	}
	if body["userAgent"] != "antigravity" || body["requestType"] != "agent" {
		t.Fatalf("envelope agent fields wrong: %v", body)
	}
	if rid, _ := body["requestId"].(string); !strings.HasPrefix(rid, "agent/") {
		t.Fatalf("requestId = %v, want agent/…", body["requestId"])
	}
	inner, _ := body["request"].(map[string]any)
	if inner == nil {
		t.Fatalf("request missing: %v", body)
	}
	sys, _ := inner["systemInstruction"].(map[string]any)
	if sys["role"] != "user" {
		t.Fatalf("systemInstruction.role = %v, want user", sys["role"])
	}
	gen, _ := inner["generationConfig"].(map[string]any)
	if gen["maxOutputTokens"] != float64(64000) {
		t.Fatalf("claude maxOutputTokens = %v, want 64000", gen["maxOutputTokens"])
	}
	tc, _ := inner["toolConfig"].(map[string]any)
	fcc, _ := tc["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "VALIDATED" {
		t.Fatalf("toolConfig mode = %v, want VALIDATED", inner["toolConfig"])
	}
	labels, _ := inner["labels"].(map[string]any)
	if labels["used_claude"] != "true" || labels["last_step_index"] != "1" {
		t.Fatalf("labels = %v", labels)
	}

	if text != "Hi" {
		t.Fatalf("text = %q, want Hi (thought part must not stream)", text)
	}
	if toolCall == nil || toolCall.Name != "read_file" || toolCall.ID == "" {
		t.Fatalf("tool call = %+v", toolCall)
	}
	if done.StopReason != "stop" {
		t.Fatalf("stop reason = %q, want stop", done.StopReason)
	}
	if done.Usage.InputTokens != 8 || done.Usage.OutputTokens != 3 || done.Usage.CacheReadTokens != 2 {
		t.Fatalf("usage = %+v, want in=8 out=3 cache=2", done.Usage)
	}
}

// An in-band stream error maps onto a ProviderError with the HTTP status the
// gRPC status implies, so the retry ladder classifies it structurally.
func TestAntigravity_StreamErrorMapsStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"error\":{\"code\":429,\"message\":\"quota\",\"status\":\"RESOURCE_EXHAUSTED\"}}\n\n")
	}))
	defer srv.Close()

	src := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a1", AccessToken: "t", ProjectID: "p"}}}
	p, err := New(Spec{Vendor: "google-antigravity", BaseURL: srv.URL, HTTP: srv.Client(), TokenSource: src})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{Model: "gemini-3-pro"})
	if err != nil {
		t.Fatal(err)
	}
	var sawErr error
	for d := range ch {
		if d.Err != nil {
			sawErr = d.Err
		}
	}
	var pe *protocol.ProviderError
	if !errors.As(sawErr, &pe) || pe.Status != 429 {
		t.Fatalf("err = %v, want ProviderError status 429", sawErr)
	}
}

// A 5xx on the first endpoint fails over to the next before any event flows.
func TestAntigravity_FailsOverOn5xx(t *testing.T) {
	var hits []string
	daily := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits = append(hits, "daily")
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer daily.Close()
	sandbox := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits = append(hits, "sandbox")
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}}\n\n")
	}))
	defer sandbox.Close()

	src := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a1", AccessToken: "t", ProjectID: "p"}}}
	inner := NewAntigravityProvider()
	inner.BaseURL = "" // exercise the failover list via a custom transport below
	p, err := newPooledProvider(VendorGoogleAntigravity, inner, src)
	if err != nil {
		t.Fatal(err)
	}
	// Route the two real endpoints at the test servers via a transport rewrite.
	inner.StreamHTTP = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		switch r.URL.Host {
		case "daily-cloudcode-pa.googleapis.com":
			r.URL.Scheme, r.URL.Host = "http", strings.TrimPrefix(daily.URL, "http://")
		case "daily-cloudcode-pa.sandbox.googleapis.com":
			r.URL.Scheme, r.URL.Host = "http", strings.TrimPrefix(sandbox.URL, "http://")
		}
		return http.DefaultTransport.RoundTrip(r)
	})}
	ch, err := p.Stream(context.Background(), protocol.ChatRequest{
		Model:    "gemini-3-pro",
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for d := range ch {
		if d.Err != nil {
			t.Fatalf("stream error: %v", d.Err)
		}
		text += d.ContentDelta
	}
	if text != "ok" {
		t.Fatalf("text = %q, want ok from sandbox", text)
	}
	if len(hits) != 2 || hits[0] != "daily" || hits[1] != "sandbox" {
		t.Fatalf("endpoint hits = %v, want daily then sandbox", hits)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
