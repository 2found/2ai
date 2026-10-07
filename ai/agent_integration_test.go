package ai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

func verdictSchema() *agentcore.OutputSchema {
	return &agentcore.OutputSchema{
		Name: "verdict",
		Schema: map[string]any{
			"type":                 "object",
			"properties":           map[string]any{"verdict": map[string]any{"type": "string"}},
			"required":             []any{"verdict"},
			"additionalProperties": false,
		},
	}
}

func TestAgentRetriesPreOutputInBandThrottle(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if calls.Add(1) == 1 {
			_, _ = w.Write([]byte("data: {\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow down\"}}\n\n"))
			return
		}
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"recovered\"},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n"))
	}))
	defer srv.Close()

	p := nativeIntegrationProvider(t, srv.URL)
	retry := agentcore.RetryPolicy{MaxAttempts: 2, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond}
	agent, err := agentcore.New(agentcore.Config{NativeProvider: p, Model: "m", Retry: &retry})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var visible strings.Builder
	result, err := agent.PromptStream(context.Background(), "go", func(event agentcore.StreamEvent) {
		visible.WriteString(event.Token)
	})
	if err != nil {
		t.Fatalf("PromptStream: %v", err)
	}
	if result.Final != "recovered" || visible.String() != "recovered" || calls.Load() != 2 {
		t.Fatalf("final=%q visible=%q calls=%d, want recovered/recovered/2", result.Final, visible.String(), calls.Load())
	}
}

func TestAgentDoesNotReplayInBandErrorAfterVisibleOutput(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n" +
			"data: {\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow down\"}}\n\n"))
	}))
	defer srv.Close()

	p := nativeIntegrationProvider(t, srv.URL)
	retry := agentcore.RetryPolicy{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond}
	agent, err := agentcore.New(agentcore.Config{NativeProvider: p, Model: "m", Retry: &retry})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var visible strings.Builder
	_, err = agent.PromptStream(context.Background(), "go", func(event agentcore.StreamEvent) {
		visible.WriteString(event.Token)
	})
	if err == nil {
		t.Fatal("PromptStream succeeded after an in-band failure")
	}
	if visible.String() != "partial" || calls.Load() != 1 {
		t.Fatalf("visible=%q calls=%d, want partial/1", visible.String(), calls.Load())
	}
}

// TestNativeAgentThreadsOutputSchema checks the actual native HTTP request.
func TestNativeAgentThreadsOutputSchema(t *testing.T) {
	var calls atomic.Int32
	var received struct {
		ResponseFormat struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Name   string          `json:"name"`
				Schema json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"choices":[{"index":0,"delta":{"content":"{\"verdict\":\"ok\"}"},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: nativeIntegrationProvider(t, srv.URL),
		Model:          "m",
		OutputSchema:   verdictSchema(),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	result, err := agent.Prompt(context.Background(), "judge this")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if result.Final != `{"verdict":"ok"}` || calls.Load() != 1 {
		t.Fatalf("result=%q calls=%d", result.Final, calls.Load())
	}
	if received.ResponseFormat.Type != "json_schema" || received.ResponseFormat.JSONSchema.Name != "verdict" || len(received.ResponseFormat.JSONSchema.Schema) == 0 {
		t.Fatalf("missing output schema: %+v", received.ResponseFormat)
	}
}

func nativeIntegrationProvider(t *testing.T, endpoint string) *ai.FallbackProvider {
	t.Helper()
	client, err := ai.NewNativeClient(ai.ClientSpec{Name: "openai", APIKey: "key", BaseURL: endpoint})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := client.Candidate("m", 100000)
	if err != nil {
		t.Fatal(err)
	}
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{candidate}}
}
