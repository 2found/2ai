package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

func writeResponsesText(w http.ResponseWriter, id, text string, input, output, cached int) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = io.WriteString(w,
		"data: {\"type\":\"response.created\",\"response\":{\"id\":\""+id+"\",\"status\":\"in_progress\"}}\n\n"+
			"data: {\"type\":\"response.output_text.delta\",\"delta\":"+mustJSON(text)+"}\n\n"+
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\""+id+"\",\"status\":\"completed\",\"usage\":{\"input_tokens\":"+itoa(input)+",\"output_tokens\":"+itoa(output)+",\"input_tokens_details\":{\"cached_tokens\":"+itoa(cached)+"}}}}\n\n")
}

func mustJSON(value string) string {
	raw, _ := json.Marshal(value)
	return string(raw)
}

func itoa(value int) string {
	raw, _ := json.Marshal(value)
	return string(raw)
}

func decodeResponsesRequest(t *testing.T, r *http.Request) responsesRequest {
	t.Helper()
	var body responsesRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	return body
}

func drainResponseStream(t *testing.T, ch <-chan protocol.ChatDelta) protocol.ChatResponse {
	t.Helper()
	var out protocol.ChatResponse
	out.Message.Role = protocol.RoleAssistant
	for delta := range ch {
		if delta.Err != nil {
			t.Fatalf("stream error: %v", delta.Err)
		}
		out.Message.Content += delta.ContentDelta
		if delta.ToolCall != nil {
			out.Message.ToolCalls = append(out.Message.ToolCalls, *delta.ToolCall)
		}
		if delta.Done {
			out.StopReason = delta.StopReason
			out.Usage = delta.Usage
		}
	}
	return out
}

func TestOpenAIResponsesChainsOnlyExactWirePrefix(t *testing.T) {
	var mu sync.Mutex
	var bodies []responsesRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeResponsesRequest(t, r)
		mu.Lock()
		bodies = append(bodies, body)
		call := len(bodies)
		mu.Unlock()
		if call == 1 {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w,
				`data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`+"\n\n"+
					`data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Hue\"}"}}`+"\n\n"+
					`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":4,"input_tokens_details":{"cached_tokens":3}}}}`+"\n\n")
			return
		}
		writeResponsesText(w, "resp_2", "sunny", 7, 2, 5)
	}))
	defer srv.Close()

	session := protocol.NewProviderSession()
	firstProvider := NewOpenAIResponsesProvider("sk-one", srv.URL)
	firstProvider.StreamHTTP = srv.Client()
	first, err := firstProvider.Chat(context.Background(), protocol.ChatRequest{
		Model: "gpt-test", SessionID: "conversation-1", ProviderSession: session,
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "weather?"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Message.ToolCalls) != 1 || first.Message.ToolCalls[0].ID != "call_1" {
		t.Fatalf("first response = %+v", first)
	}
	if first.Usage.InputTokens != 7 || first.Usage.CacheReadTokens != 3 {
		t.Fatalf("usage = %+v, want uncached=7 cached=3", first.Usage)
	}

	// Runtime agents and wire clients are rebuilt between turns. The provider
	// session, not the Go client instance, owns the chain.
	secondProvider := NewOpenAIResponsesProvider("sk-one", srv.URL)
	secondProvider.StreamHTTP = srv.Client()
	second, err := secondProvider.Chat(context.Background(), protocol.ChatRequest{
		Model: "gpt-test", SessionID: "conversation-1", ProviderSession: session,
		Messages: []protocol.Message{
			{Role: protocol.RoleUser, Content: "weather?"},
			{Role: protocol.RoleAssistant, ToolCalls: first.Message.ToolCalls},
			{Role: protocol.RoleTool, ToolCallID: "call_1", Name: "weather", Content: `{"temperature":31}`},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if second.Message.Content != "sunny" || second.StopReason != "stop" {
		t.Fatalf("second response = %+v", second)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 {
		t.Fatalf("requests = %d, want 2", len(bodies))
	}
	if !bodies[0].Store || bodies[0].PreviousResponseID != "" || len(bodies[0].Input) != 1 {
		t.Fatalf("first body = %+v, want stored full replay", bodies[0])
	}
	got := bodies[1]
	if !got.Store || got.PreviousResponseID != "resp_1" {
		t.Fatalf("second body = %+v, want chained resp_1", got)
	}
	if len(got.Input) != 1 || got.Input[0].Type != "function_call_output" || got.Input[0].CallID != "call_1" {
		t.Fatalf("second input = %+v, want only the tool-result suffix", got.Input)
	}
}

func TestOpenAIResponsesHistoryMutationBreaksChain(t *testing.T) {
	var bodies []responsesRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodies = append(bodies, decodeResponsesRequest(t, r))
		writeResponsesText(w, "resp_"+itoa(len(bodies)), "answer", 1, 1, 0)
	}))
	defer srv.Close()

	session := protocol.NewProviderSession()
	p := NewOpenAIResponsesProvider("sk", srv.URL)
	p.StreamHTTP = srv.Client()
	first, err := p.Chat(context.Background(), protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: session,
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "original"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Chat(context.Background(), protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: session,
		Messages: []protocol.Message{
			{Role: protocol.RoleUser, Content: "edited"},
			first.Message,
			{Role: protocol.RoleUser, Content: "next"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(bodies) != 2 || bodies[1].PreviousResponseID != "" || len(bodies[1].Input) != 3 {
		t.Fatalf("mutated request = %+v, want full three-item replay", bodies[1])
	}
}

func TestOpenAIResponsesStaleChainRetriesFullContext(t *testing.T) {
	var bodies []responsesRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeResponsesRequest(t, r)
		bodies = append(bodies, body)
		switch len(bodies) {
		case 1:
			writeResponsesText(w, "resp_old", "first", 1, 1, 0)
		case 2:
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"code":"previous_response_not_found","message":"Previous response not found"}}`)
		default:
			writeResponsesText(w, "resp_new", "second", 2, 1, 0)
		}
	}))
	defer srv.Close()

	session := protocol.NewProviderSession()
	p := NewOpenAIResponsesProvider("sk", srv.URL)
	p.StreamHTTP = srv.Client()
	first, err := p.Chat(context.Background(), protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: session,
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "one"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	second, err := p.Chat(context.Background(), protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: session,
		Messages: []protocol.Message{
			{Role: protocol.RoleUser, Content: "one"}, first.Message,
			{Role: protocol.RoleUser, Content: "two"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if second.Message.Content != "second" {
		t.Fatalf("response = %+v", second)
	}
	if len(bodies) != 3 {
		t.Fatalf("requests = %d, want full + stale delta + full retry", len(bodies))
	}
	if bodies[1].PreviousResponseID != "resp_old" || len(bodies[1].Input) != 1 {
		t.Fatalf("stale attempt = %+v", bodies[1])
	}
	if bodies[2].PreviousResponseID != "" || len(bodies[2].Input) != 3 || !bodies[2].Store {
		t.Fatalf("fallback = %+v, want stored full replay", bodies[2])
	}
}

func TestOpenAIResponsesStateLossAccountResetAndKeyChangeReplayFully(t *testing.T) {
	var bodies []responsesRequest
	var auth []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodies = append(bodies, decodeResponsesRequest(t, r))
		auth = append(auth, r.Header.Get("Authorization"))
		writeResponsesText(w, "resp_"+itoa(len(bodies)), "ok", 1, 1, 0)
	}))
	defer srv.Close()

	session := protocol.NewProviderSession()
	p := NewOpenAIResponsesProvider("key-a", srv.URL)
	p.StreamHTTP = srv.Client()
	first, err := p.Chat(context.Background(), protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: session,
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "one"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	history := []protocol.Message{
		{Role: protocol.RoleUser, Content: "one"}, first.Message,
		{Role: protocol.RoleUser, Content: "two"},
	}

	session.ResetAccountScoped()
	if _, err := p.Chat(context.Background(), protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: session, Messages: history,
	}); err != nil {
		t.Fatal(err)
	}
	if bodies[1].PreviousResponseID != "" || len(bodies[1].Input) != 3 {
		t.Fatalf("after account reset = %+v, want full replay", bodies[1])
	}

	p.UpdateAPIKey("key-b")
	if _, err := p.Chat(context.Background(), protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: session, Messages: history,
	}); err != nil {
		t.Fatal(err)
	}
	if bodies[2].PreviousResponseID != "" || len(bodies[2].Input) != 3 || auth[2] != "Bearer key-b" {
		t.Fatalf("after key change = %+v auth=%q, want isolated full replay", bodies[2], auth[2])
	}

	// A different process/replica has no state at all. It replays fully and asks
	// the provider not to retain a response, yet returns the same answer.
	if _, err := p.Chat(context.Background(), protocol.ChatRequest{
		Model: "m", SessionID: "s", Messages: history,
	}); err != nil {
		t.Fatal(err)
	}
	if bodies[3].PreviousResponseID != "" || bodies[3].Store || len(bodies[3].Input) != 3 {
		t.Fatalf("stateless request = %+v, want store:false full replay", bodies[3])
	}
}

func TestOpenAIResponsesSerializesOneConversationChain(t *testing.T) {
	var calls atomic.Int32
	firstRelease := make(chan struct{})
	firstSeen := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decodeResponsesRequest(t, r)
		call := calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		if call == 1 {
			close(firstSeen)
			w.(http.Flusher).Flush()
			<-firstRelease
		}
		writeResponsesText(w, "resp_"+itoa(int(call)), "ok", 1, 1, 0)
	}))
	defer srv.Close()

	p := NewOpenAIResponsesProvider("sk", srv.URL)
	p.StreamHTTP = srv.Client()
	session := protocol.NewProviderSession()
	request := protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: session,
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "one"}},
	}
	first, err := p.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	<-firstSeen
	firstDone := make(chan struct{})
	go func() {
		for range first {
		}
		close(firstDone)
	}()

	secondResult := make(chan error, 1)
	go func() {
		second, err := p.Stream(context.Background(), request)
		if err == nil {
			for range second {
			}
		}
		secondResult <- err
	}()
	select {
	case err := <-secondResult:
		t.Fatalf("second request escaped chain serialization early: %v", err)
	case <-time.After(75 * time.Millisecond):
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("server calls while first chain active = %d, want 1", got)
	}
	close(firstRelease)
	select {
	case <-firstDone:
	case <-time.After(time.Second):
		t.Fatal("first stream did not finish")
	}
	select {
	case err := <-secondResult:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("second request did not start after first completed")
	}
}

func TestOpenAIResponsesEncodesReasoningSchemaAndTools(t *testing.T) {
	p := NewOpenAIResponsesProvider("sk", "")
	body := p.encode(protocol.ChatRequest{
		Model: "gpt-test", MaxTokens: 123, ReasoningEffort: "high", CacheKey: "conversation-1",
		Messages:     []protocol.Message{{Role: protocol.RoleSystem, Content: "system"}, {Role: protocol.RoleUser, Content: "hi"}},
		Tools:        []protocol.ToolSchema{{Name: "read", Description: "read a file"}},
		OutputSchema: &protocol.OutputSchema{Name: "verdict", Strict: true, Schema: map[string]any{"type": "object"}},
	})
	if body.Instructions != "system" || body.MaxOutputTokens != 123 || body.PromptCacheKey != "conversation-1" {
		t.Fatalf("body = %+v", body)
	}
	if body.Reasoning == nil || body.Reasoning.Effort != "high" {
		t.Fatalf("reasoning = %+v", body.Reasoning)
	}
	if body.Text == nil || body.Text.Format.Type != "json_schema" || !body.Text.Format.Strict || body.Text.Format.Name != "verdict" {
		t.Fatalf("text format = %+v", body.Text)
	}
	if len(body.Tools) != 1 || body.Tools[0].Parameters["type"] != "object" {
		t.Fatalf("tools = %+v", body.Tools)
	}
}

func TestOpenAIResponsesStaleCircuitBreakerSurvivesAccountReset(t *testing.T) {
	state := newOpenAIResponsesState().(*openAIResponsesState)
	chain := state.chain("model\x00session")
	chain.mu.Lock()
	plan := responsesPlan{
		canonical: responsesRequest{Model: "m", Stream: true, Store: true},
		wire:      responsesRequest{Model: "m", Stream: true, Store: true},
		chain:     chain,
		chained:   true,
		release:   chain.mu.Unlock,
	}
	for range responsesStaleLimit {
		plan.stale(false)
	}
	if !chain.disabled || chain.stale != responsesStaleLimit || plan.wire.Store {
		t.Fatalf("chain after stale failures = %+v wire.store=%v", chain, plan.wire.Store)
	}
	plan.release()

	state.ResetAccountScoped()
	chain.mu.Lock()
	defer chain.mu.Unlock()
	if !chain.disabled {
		t.Fatal("account reset erased the endpoint circuit breaker")
	}
	if chain.stale != 0 || chain.canAppend || chain.lastID != "" {
		t.Fatalf("account reset left account-bound state: %+v", chain)
	}
}

func TestOpenAIResponsesDoesNotWeakenUnrelatedBadRequest(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decodeResponsesRequest(t, r)
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"code":"invalid_json_schema","message":"Invalid response_format schema"}}`)
	}))
	defer srv.Close()

	p := NewOpenAIResponsesProvider("sk", srv.URL)
	p.StreamHTTP = srv.Client()
	_, err := p.Stream(context.Background(), protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: protocol.NewProviderSession(),
		Messages:     []protocol.Message{{Role: protocol.RoleUser, Content: "one"}},
		OutputSchema: &protocol.OutputSchema{Name: "bad", Schema: map[string]any{"type": "broken"}},
	})
	if err == nil {
		t.Fatal("invalid schema unexpectedly succeeded")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("requests = %d, unrelated 400 must not trigger a weakened retry", got)
	}
}
