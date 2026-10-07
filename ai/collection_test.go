package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// Two registered providers each list exactly the IDs their stub /models
// (or vendor equivalent) returned — no hardcoded catalog.
func TestCollectionListsModelsFromVendorAPIs(t *testing.T) {
	openaiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/models") {
			if got := r.Header.Get("Authorization"); got != "Bearer sk-oai" {
				t.Errorf("openai list auth = %q", got)
			}
			_, _ = w.Write([]byte(`{"data":[{"id":"oai-alpha"},{"id":"oai-beta"}]}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer openaiSrv.Close()

	anthSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			if got := r.Header.Get("x-api-key"); got != "sk-ant" {
				t.Errorf("anthropic list auth = %q", got)
			}
			if got := r.Header.Get("anthropic-version"); got == "" {
				t.Error("missing anthropic-version")
			}
			_, _ = w.Write([]byte(`{"data":[{"id":"ant-one"},{"id":"ant-two"}]}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer anthSrv.Close()

	col := NewCollection()
	oai, err := New(Spec{ID: "prov-oai", Vendor: "openai", Name: "OpenAI", APIKey: "sk-oai", BaseURL: openaiSrv.URL, HTTP: openaiSrv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	ant, err := New(Spec{ID: "prov-ant", Vendor: "anthropic", Name: "Anthropic", APIKey: "sk-ant", BaseURL: anthSrv.URL, HTTP: anthSrv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	col.Register(oai)
	col.Register(ant)

	got, err := col.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Errors) != 0 {
		t.Fatalf("list errors: %+v", got.Errors)
	}
	want := map[string]string{
		"oai-alpha": "prov-oai",
		"oai-beta":  "prov-oai",
		"ant-one":   "prov-ant",
		"ant-two":   "prov-ant",
	}
	if len(got.Models) != len(want) {
		t.Fatalf("listed %d models, want %d: %+v", len(got.Models), len(want), got.Models)
	}
	for _, m := range got.Models {
		owner, ok := want[m.ID]
		if !ok {
			t.Errorf("invented model id %q (not returned by either stub)", m.ID)
			continue
		}
		if m.ProviderID != owner {
			t.Errorf("model %q owner = %q, want %q", m.ID, m.ProviderID, owner)
		}
	}
}

// Chat/Stream for a model lands on the owning provider's stub host.
func TestCollectionDispatchesChatAndStreamToOwner(t *testing.T) {
	var openaiHits, anthHits int
	openaiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/models"):
			_, _ = w.Write([]byte(`{"data":[{"id":"oai-only"}]}`))
		case r.URL.Path == "/chat/completions":
			openaiHits++
			if r.Header.Get("Authorization") != "Bearer sk-oai" {
				t.Errorf("openai chat auth = %q", r.Header.Get("Authorization"))
			}
			if strings.Contains(r.Header.Get("Accept"), "text/event-stream") {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi-oai\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\ndata: [DONE]\n"))
				return
			}
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"from-openai"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer openaiSrv.Close()

	anthSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/models":
			_, _ = w.Write([]byte(`{"data":[{"id":"ant-only"}]}`))
		case r.URL.Path == "/v1/messages":
			anthHits++
			if r.Header.Get("x-api-key") != "sk-ant" {
				t.Errorf("anthropic chat auth = %q", r.Header.Get("x-api-key"))
			}
			if strings.Contains(r.Header.Get("Accept"), "text/event-stream") {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n")
				_, _ = io.WriteString(w, "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi-ant\"}}\n\n")
				_, _ = io.WriteString(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n")
				_, _ = io.WriteString(w, "data: {\"type\":\"message_stop\"}\n\n")
				return
			}
			_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"from-anthropic"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":2}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer anthSrv.Close()

	col := NewCollection()
	oai, err := New(Spec{ID: "a", Vendor: "openai", APIKey: "sk-oai", BaseURL: openaiSrv.URL, HTTP: openaiSrv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	ant, err := New(Spec{ID: "b", Vendor: "anthropic", APIKey: "sk-ant", BaseURL: anthSrv.URL, HTTP: anthSrv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	col.Register(oai)
	col.Register(ant)
	if _, err := col.ListModels(context.Background()); err != nil {
		t.Fatal(err)
	}

	chatOAI, err := col.Chat(context.Background(), protocol.ChatRequest{
		Model: "oai-only", Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("chat openai: %v", err)
	}
	if chatOAI.Message.Content != "from-openai" {
		t.Errorf("openai chat content = %q", chatOAI.Message.Content)
	}
	if openaiHits != 1 || anthHits != 0 {
		t.Fatalf("after openai chat: openaiHits=%d anthHits=%d", openaiHits, anthHits)
	}

	chatAnt, err := col.Chat(context.Background(), protocol.ChatRequest{
		Model: "ant-only", Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("chat anthropic: %v", err)
	}
	if chatAnt.Message.Content != "from-anthropic" {
		t.Errorf("anthropic chat content = %q", chatAnt.Message.Content)
	}
	if openaiHits != 1 || anthHits != 1 {
		t.Fatalf("after anthropic chat: openaiHits=%d anthHits=%d", openaiHits, anthHits)
	}

	stream, err := col.Stream(context.Background(), protocol.ChatRequest{
		Model: "oai-only", Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("stream openai: %v", err)
	}
	var got string
	for d := range stream {
		if d.Err != nil {
			t.Fatalf("stream err: %v", d.Err)
		}
		got += d.ContentDelta
	}
	if got != "hi-oai" {
		t.Errorf("openai stream = %q", got)
	}
	if openaiHits != 2 {
		t.Fatalf("openai stream did not hit openai stub (hits=%d)", openaiHits)
	}

	stream, err = col.Stream(context.Background(), protocol.ChatRequest{
		Model: "ant-only", Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("stream anthropic: %v", err)
	}
	got = ""
	for d := range stream {
		if d.Err != nil {
			t.Fatalf("stream err: %v", d.Err)
		}
		got += d.ContentDelta
	}
	if got != "hi-ant" {
		t.Errorf("anthropic stream = %q", got)
	}
	if anthHits != 2 {
		t.Fatalf("anthropic stream did not hit anthropic stub (hits=%d)", anthHits)
	}
}

// OpenAI Chat/Stream still encode tools on the wire and report usage.
func TestOpenAIChatAndStreamToolsAndUsage(t *testing.T) {
	var sawTools bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			_, _ = w.Write([]byte(`{"data":[{"id":"m1"}]}`))
			return
		}
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"tools"`) && strings.Contains(string(body), `"lookup"`) {
			sawTools = true
		}
		if strings.Contains(r.Header.Get("Accept"), "text/event-stream") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\ndata: [DONE]\n"))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"done","tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":11,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":4}}}`))
	}))
	defer srv.Close()

	p, err := New(Spec{ID: "o", Vendor: "openai", APIKey: "k", BaseURL: srv.URL, HTTP: srv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	req := protocol.ChatRequest{
		Model:    "m1",
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "go"}},
		Tools:    []protocol.ToolSchema{{Name: "lookup", Description: "d", Parameters: map[string]any{"type": "object"}}},
	}
	resp, err := p.Chat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if !sawTools {
		t.Fatal("openai Chat request did not encode tools")
	}
	if resp.Usage.InputTokens != 7 || resp.Usage.OutputTokens != 2 || resp.Usage.CacheReadTokens != 4 {
		t.Fatalf("openai chat usage = %+v, want in=7 (11-4 cached) out=2 cache=4", resp.Usage)
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "lookup" {
		t.Fatalf("openai chat tools = %+v", resp.Message.ToolCalls)
	}

	sawTools = false
	ch, err := p.Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	var usage protocol.Usage
	for d := range ch {
		if d.Err != nil {
			t.Fatal(d.Err)
		}
		if d.Done {
			usage = d.Usage
		}
	}
	if !sawTools {
		t.Fatal("openai Stream request did not encode tools")
	}
	if usage.InputTokens != 7 || usage.OutputTokens != 2 || usage.CacheReadTokens != 4 {
		t.Fatalf("openai stream usage = %+v", usage)
	}
}

// Anthropic Chat/Stream still encode tools on the wire and report usage.
func TestAnthropicChatAndStreamToolsAndUsage(t *testing.T) {
	var sawTools bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			_, _ = w.Write([]byte(`{"data":[{"id":"claude-x"}]}`))
			return
		}
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"tools"`) && strings.Contains(string(body), `"lookup"`) {
			sawTools = true
		}
		if strings.Contains(r.Header.Get("Accept"), "text/event-stream") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":9,\"output_tokens\":0,\"cache_read_input_tokens\":3,\"cache_creation_input_tokens\":1}}}\n\n")
			_, _ = io.WriteString(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu1\",\"name\":\"lookup\"}}\n\n")
			_, _ = io.WriteString(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"x\\\"}\"}}\n\n")
			_, _ = io.WriteString(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n")
			_, _ = io.WriteString(w, "data: {\"type\":\"message_stop\"}\n\n")
			return
		}
		_, _ = w.Write([]byte(`{"content":[{"type":"tool_use","id":"tu1","name":"lookup","input":{"q":"x"}}],"stop_reason":"tool_use","usage":{"input_tokens":9,"output_tokens":5,"cache_read_input_tokens":3,"cache_creation_input_tokens":1}}`))
	}))
	defer srv.Close()

	p, err := New(Spec{ID: "a", Vendor: "anthropic", APIKey: "k", BaseURL: srv.URL, HTTP: srv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	req := protocol.ChatRequest{
		Model:    "claude-x",
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "go"}},
		Tools:    []protocol.ToolSchema{{Name: "lookup", Description: "d", Parameters: map[string]any{"type": "object"}}},
	}
	resp, err := p.Chat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if !sawTools {
		t.Fatal("anthropic Chat request did not encode tools")
	}
	if resp.Usage.InputTokens != 9 || resp.Usage.OutputTokens != 5 || resp.Usage.CacheReadTokens != 3 || resp.Usage.CacheWriteTokens != 1 {
		t.Fatalf("anthropic chat usage = %+v", resp.Usage)
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "lookup" {
		t.Fatalf("anthropic chat tools = %+v", resp.Message.ToolCalls)
	}

	sawTools = false
	ch, err := p.Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	var usage protocol.Usage
	var calls []protocol.ToolCall
	for d := range ch {
		if d.Err != nil {
			t.Fatal(d.Err)
		}
		if d.ToolCall != nil {
			calls = append(calls, *d.ToolCall)
		}
		if d.Done {
			usage = d.Usage
		}
	}
	if !sawTools {
		t.Fatal("anthropic Stream request did not encode tools")
	}
	if usage.InputTokens != 9 || usage.OutputTokens != 5 || usage.CacheReadTokens != 3 || usage.CacheWriteTokens != 1 {
		t.Fatalf("anthropic stream usage = %+v", usage)
	}
	if len(calls) != 1 || calls[0].Name != "lookup" || calls[0].Arguments != `{"q":"x"}` {
		t.Fatalf("anthropic stream tools = %+v", calls)
	}
}

func TestGoogleListModelsUsesVendorAPI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1beta/models") {
			http.NotFound(w, r)
			return
		}
		if r.URL.Query().Get("key") != "gk" {
			t.Errorf("missing/wrong key query: %s", r.URL.RawQuery)
		}
		_, _ = w.Write([]byte(`{"models":[{"name":"models/gemini-flash"},{"name":"models/gemini-pro"}]}`))
	}))
	defer srv.Close()

	p, err := New(Spec{ID: "g", Vendor: "google", Name: "Gemini", APIKey: "gk", BaseURL: srv.URL, HTTP: srv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "gemini-flash" || models[1].ID != "gemini-pro" {
		t.Fatalf("google models = %+v", models)
	}
}

func TestOpenAICompatVendorRequiresBaseURL(t *testing.T) {
	if _, err := New(Spec{Vendor: "groq", APIKey: "k"}); err == nil {
		t.Fatal("expected error for compat vendor without base URL")
	}
}

func TestNewResolvesBuiltInVendors(t *testing.T) {
	cases := []struct {
		vendor string
		want   string
	}{
		{"openai", "openai"},
		{"anthropic", "anthropic"},
		{"google", "google"},
		{"gemini", "google"},
	}
	for _, tc := range cases {
		p, err := New(Spec{Vendor: tc.vendor, APIKey: "k"})
		if err != nil {
			t.Fatalf("%s: %v", tc.vendor, err)
		}
		if p.Name() != tc.want {
			t.Errorf("%s Name() = %q, want %q", tc.vendor, p.Name(), tc.want)
		}
	}
}

func TestListModelsSurfacesVendorError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusUnauthorized)
	}))
	defer srv.Close()
	p, err := New(Spec{ID: "x", Vendor: "openai", APIKey: "k", BaseURL: srv.URL, HTTP: srv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	col := NewCollection()
	col.Register(p)
	got, err := col.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Errors) != 1 || got.Errors[0].ProviderID != "x" {
		t.Fatalf("expected per-provider error, got %+v", got)
	}
	if len(got.Models) != 0 {
		t.Fatalf("failed list must not invent models: %+v", got.Models)
	}
}

func TestChatOnUsesNamedProvider(t *testing.T) {
	var hits []string
	srvA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits = append(hits, "A"+r.URL.Path)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"A"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srvA.Close()
	srvB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits = append(hits, "B"+r.URL.Path)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"B"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srvB.Close()

	a, err := New(Spec{ID: "pa", Vendor: "openai-compat", APIKey: "ka", BaseURL: srvA.URL, HTTP: srvA.Client()})
	if err != nil {
		t.Fatal(err)
	}
	b, err := New(Spec{ID: "pb", Vendor: "openai-compat", APIKey: "kb", BaseURL: srvB.URL, HTTP: srvB.Client()})
	if err != nil {
		t.Fatal(err)
	}
	col := NewCollection()
	col.Register(a)
	col.Register(b)
	resp, err := col.ChatOn(context.Background(), "pb", protocol.ChatRequest{
		Model: "same-id", Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "x"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "B" {
		t.Fatalf("content = %q, want B", resp.Message.Content)
	}
	for _, h := range hits {
		if strings.HasPrefix(h, "A") {
			t.Fatalf("request leaked to provider A: %v", hits)
		}
	}
}

// sanity: a listed-model JSON shape used by the workspace API is what Collection returns
func TestListedModelJSONRoundTrip(t *testing.T) {
	m := Model{ProviderID: "p1", ProviderVendor: "openai", ProviderName: "Work", ID: "stub-1"}
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"id":"stub-1"`) || !strings.Contains(string(raw), `"provider_id":"p1"`) {
		t.Fatalf("json = %s", raw)
	}
}
