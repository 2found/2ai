package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func strictTestTool(mode protocol.ToolStrictness) protocol.ToolSchema {
	return protocol.ToolSchema{
		Name: "read", Strict: mode,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string", "pattern": "^[a-z]+$"},
			},
			"required":             []string{"path"},
			"additionalProperties": false,
		},
	}
}

func TestProjectedToolParametersStrictIsSemanticAndCopyOnWrite(t *testing.T) {
	tool := strictTestTool(protocol.ToolStrictEnabled)
	original := tool.Parameters["properties"].(map[string]any)["path"].(map[string]any)
	parameters, strict := projectedToolParameters(tool, toolSchemaOpenAIResponses)
	if strict == nil || !*strict {
		t.Fatalf("strict = %v, want true", strict)
	}
	if parameters["additionalProperties"] != false {
		t.Fatalf("additionalProperties = %#v", parameters["additionalProperties"])
	}
	path := parameters["properties"].(map[string]any)["path"].(map[string]any)
	if _, exists := path["pattern"]; exists {
		t.Fatalf("unsupported strict constraint leaked: %#v", path)
	}
	if original["pattern"] == nil || tool.Parameters["additionalProperties"] != false {
		t.Fatalf("canonical schema mutated: %#v", tool.Parameters)
	}

	optional := strictTestTool(protocol.ToolStrictEnabled)
	delete(optional.Parameters, "required")
	_, strict = projectedToolParameters(optional, toolSchemaOpenAIResponses)
	if strict != nil {
		t.Fatalf("optional schema changed semantics under strict: %v", *strict)
	}
	openMap := strictTestTool(protocol.ToolStrictEnabled)
	openMap.Parameters["additionalProperties"] = true
	_, strict = projectedToolParameters(openMap, toolSchemaOpenAIResponses)
	if strict != nil {
		t.Fatalf("open-map schema was narrowed under strict: %v", *strict)
	}
	implicitOpen := strictTestTool(protocol.ToolStrictEnabled)
	delete(implicitOpen.Parameters, "additionalProperties")
	_, strict = projectedToolParameters(implicitOpen, toolSchemaOpenAIResponses)
	if strict != nil {
		t.Fatalf("implicit-open schema was narrowed under strict: %v", *strict)
	}

	disabled := strictTestTool(protocol.ToolStrictDisabled)
	parameters, strict = projectedToolParameters(disabled, toolSchemaGeneric)
	if strict == nil || *strict || !reflect.DeepEqual(parameters, toolParameters(disabled.Parameters, toolSchemaGeneric)) {
		t.Fatalf("explicit loose projection = strict %v parameters %#v", strict, parameters)
	}
}

func TestOpenAIFamilyEncodersCarryPerToolStrictness(t *testing.T) {
	req := protocol.ChatRequest{Model: "m", Tools: []protocol.ToolSchema{strictTestTool(protocol.ToolStrictEnabled)}}
	chat := NewOpenAIProvider("k", "", DefaultCompat()).encode(req)
	responses := NewOpenAIResponsesProvider("k", "").encode(req)
	codex := NewCodexProvider().encode(req)
	if len(chat.Tools) != 1 || chat.Tools[0].Function.Strict == nil || !*chat.Tools[0].Function.Strict {
		t.Fatalf("chat strict tool = %#v", chat.Tools)
	}
	if len(responses.Tools) != 1 || responses.Tools[0].Strict == nil || !*responses.Tools[0].Strict {
		t.Fatalf("responses strict tool = %#v", responses.Tools)
	}
	if len(codex.Tools) != 1 || codex.Tools[0].Strict == nil || !*codex.Tools[0].Strict {
		t.Fatalf("codex strict tool = %#v", codex.Tools)
	}
}

func TestOpenAIChatRetriesStrictRejectionOnceAndRemembersSession(t *testing.T) {
	var mu sync.Mutex
	var strictValues []*bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body oaiRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var strict *bool
		if len(body.Tools) > 0 {
			strict = body.Tools[0].Function.Strict
		}
		mu.Lock()
		strictValues = append(strictValues, strict)
		mu.Unlock()
		if strict != nil {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"invalid schema for function: strict tools unsupported"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	provider := NewOpenAIProvider("k", srv.URL, DefaultCompat())
	provider.HTTP = srv.Client()
	session := protocol.NewProviderSession()
	req := protocol.ChatRequest{Model: "m", ProviderSession: session, Tools: []protocol.ToolSchema{strictTestTool(protocol.ToolStrictEnabled)}}
	for i := 0; i < 2; i++ {
		response, err := provider.Chat(context.Background(), req)
		if err != nil || response.Message.Content != "ok" {
			t.Fatalf("chat %d = %+v err=%v", i, response, err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(strictValues) != 3 || strictValues[0] == nil || !*strictValues[0] || strictValues[1] != nil || strictValues[2] != nil {
		t.Fatalf("strict attempts = %#v, want [true nil nil]", strictValues)
	}
}

func TestOpenAIStreamRetriesInBandStrictRejectionOnlyBeforeOutput(t *testing.T) {
	for _, tc := range []struct {
		name         string
		visibleFirst bool
		wantCalls    int32
	}{
		{name: "pre-output", wantCalls: 2},
		{name: "post-output", visibleFirst: true, wantCalls: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				call := calls.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				if call == 1 {
					if tc.visibleFirst {
						_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"content":"partial"}}]}`+"\n\n")
					}
					_, _ = io.WriteString(w, `data: {"error":{"message":"invalid schema for function: strict tools unsupported"}}`+"\n\n")
					return
				}
				_, _ = io.WriteString(w,
					`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n"+
						"data: [DONE]\n\n")
			}))
			defer srv.Close()

			provider := NewOpenAIProvider("k", srv.URL, DefaultCompat())
			provider.StreamHTTP = srv.Client()
			stream, err := provider.Stream(context.Background(), protocol.ChatRequest{
				Model: "m", ProviderSession: protocol.NewProviderSession(),
				Tools: []protocol.ToolSchema{strictTestTool(protocol.ToolStrictEnabled)},
			})
			if err != nil {
				t.Fatal(err)
			}
			var content string
			var streamErr error
			for delta := range stream {
				content += delta.ContentDelta
				if delta.Err != nil {
					streamErr = delta.Err
				}
			}
			if calls.Load() != tc.wantCalls {
				t.Fatalf("calls = %d, want %d", calls.Load(), tc.wantCalls)
			}
			if tc.visibleFirst {
				if content != "partial" || streamErr == nil {
					t.Fatalf("committed stream = content %q err %v", content, streamErr)
				}
			} else if content != "ok" || streamErr != nil {
				t.Fatalf("retried stream = content %q err %v", content, streamErr)
			}
		})
	}
}

func TestOpenAIResponsesRetriesStrictRejectionBeforeOutput(t *testing.T) {
	var mu sync.Mutex
	var strictValues []*bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeResponsesRequest(t, r)
		var strict *bool
		if len(body.Tools) > 0 {
			strict = body.Tools[0].Strict
		}
		mu.Lock()
		strictValues = append(strictValues, strict)
		mu.Unlock()
		if strict != nil {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"type":"error","error":{"message":"strict tool schema is not supported"}}`+"\n\n")
			return
		}
		writeResponsesText(w, "resp", "ok", 1, 1, 0)
	}))
	defer srv.Close()

	provider := NewOpenAIResponsesProvider("k", srv.URL)
	provider.StreamHTTP = srv.Client()
	req := protocol.ChatRequest{
		Model: "m", SessionID: "s", ProviderSession: protocol.NewProviderSession(),
		Tools: []protocol.ToolSchema{strictTestTool(protocol.ToolStrictEnabled)},
	}
	response, err := provider.Chat(context.Background(), req)
	if err != nil || response.Message.Content != "ok" {
		t.Fatalf("response = %+v err=%v", response, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(strictValues) != 2 || strictValues[0] == nil || !*strictValues[0] || strictValues[1] != nil {
		t.Fatalf("strict attempts = %#v, want [true nil]", strictValues)
	}
}

func TestCodexRetriesStrictRejectionOnceAndRemembersSession(t *testing.T) {
	var mu sync.Mutex
	var strictValues []*bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body codexRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var strict *bool
		if len(body.Tools) > 0 {
			strict = body.Tools[0].Strict
		}
		mu.Lock()
		strictValues = append(strictValues, strict)
		mu.Unlock()
		if strict != nil {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"strict tool schema is unsupported"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w,
			`data: {"type":"response.output_text.delta","delta":"ok"}`+"\n\n"+
				`data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}`+"\n\n")
	}))
	defer srv.Close()

	provider := NewCodexProvider()
	provider.BaseURL = srv.URL
	provider.StreamHTTP = srv.Client()
	session := protocol.NewProviderSession()
	req := protocol.ChatRequest{Model: "m", ProviderSession: session, Tools: []protocol.ToolSchema{strictTestTool(protocol.ToolStrictEnabled)}}
	for i := 0; i < 2; i++ {
		stream, err := provider.Stream(context.Background(), req)
		if err != nil {
			t.Fatalf("stream %d: %v", i, err)
		}
		response := drainResponseStream(t, stream)
		if response.Message.Content != "ok" {
			t.Fatalf("response %d = %+v", i, response)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(strictValues) != 3 || strictValues[0] == nil || !*strictValues[0] || strictValues[1] != nil || strictValues[2] != nil {
		t.Fatalf("strict attempts = %#v, want [true nil nil]", strictValues)
	}
}

func TestStrictFallbackDetectorIsNarrow(t *testing.T) {
	req := protocol.ChatRequest{Tools: []protocol.ToolSchema{strictTestTool(protocol.ToolStrictEnabled)}}
	if shouldRetryWithoutStrictTools(req, toolSchemaGeneric, protocol.NewProviderError("openai", &http.Response{StatusCode: http.StatusUnauthorized}, "strict tool unsupported")) {
		t.Fatal("authentication error must not retry")
	}
	if shouldRetryWithoutStrictTools(req, toolSchemaGeneric, protocol.NewProviderError("openai", &http.Response{StatusCode: http.StatusBadRequest}, "invalid request")) {
		t.Fatal("unrelated bad request must not retry")
	}
	if !shouldRetryWithoutStrictTools(req, toolSchemaGeneric, protocol.NewProviderError("openai", &http.Response{StatusCode: http.StatusBadRequest}, "invalid schema for function")) {
		t.Fatal("strict function schema rejection should retry")
	}
	optional := req
	optional.Tools = []protocol.ToolSchema{strictTestTool(protocol.ToolStrictEnabled)}
	delete(optional.Tools[0].Parameters, "required")
	if shouldRetryWithoutStrictTools(optional, toolSchemaGeneric, protocol.NewProviderError("openai", &http.Response{StatusCode: http.StatusBadRequest}, "invalid schema for function")) {
		t.Fatal("projection omitted strict, so the detector must not retry")
	}
}
