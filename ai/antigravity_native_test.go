package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestAntigravityNativePool(t *testing.T) {
	for _, mode := range []string{"success", "rotate", "stream-auth", "truncated", "quota", "concurrency", "callback", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			source := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a", AccessToken: "first-fixture", ProjectID: "first-project"}, {AccountID: "b", AccessToken: "second-fixture", ProjectID: "second-project"}}}
			count := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				count++
				var payload struct{ Project string }
				_ = json.NewDecoder(r.Body).Decode(&payload)
				expected := "first"
				if count == 2 {
					expected = "second"
				}
				if payload.Project != expected+"-project" || r.Header.Get("Authorization") != "Bearer "+expected+"-fixture" || r.Header.Get("User-Agent") != AntigravityUserAgent {
					t.Error("account/project mismatch")
				}
				if r.URL.Path != "/v1internal:streamGenerateContent" || r.URL.Query().Get("alt") != "sse" {
					t.Error("wrong endpoint")
				}
				if mode == "rotate" && count == 1 {
					http.Error(w, "expired", 401)
					return
				}
				if mode == "concurrency" {
					http.Error(w, "concurrency cap", 403)
					return
				}
				if mode == "quota" {
					w.Header().Set("Retry-After", "2")
					http.Error(w, "quota", 429)
					return
				}
				if mode == "cancel" {
					w.Header().Set("Content-Type", "text/event-stream")
					w.WriteHeader(200)
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				}
				if mode == "stream-auth" && count == 1 {
					fmt.Fprint(w, "data: {\"error\":{\"code\":401,\"status\":\"UNAUTHENTICATED\",\"message\":\"expired\"}}\n\n")
					return
				}
				fmt.Fprint(w, "data: "+`{"response":{"candidates":[{"content":{"parts":[{"text":"thought","thought":true,"thoughtSignature":"c2ln"},{"text":"answer","thoughtSignature":"dGV4dA=="},{"functionCall":{"id":"call","name":"write","args":{}},"thoughtSignature":"dG9vbA=="}]}}]}}`+"\r\n\r\n")
				if mode != "truncated" {
					fmt.Fprint(w, "data: "+`{"response":{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":8,"cachedContentTokenCount":3,"candidatesTokenCount":2,"thoughtsTokenCount":1}}}`+"\n\n")
				}
			}))
			defer server.Close()
			model, _ := json.Marshal(map[string]any{"id": "gemini-3-pro", "api": VendorGoogleAntigravity, "provider": VendorGoogleAntigravity, "baseUrl": server.URL, "input": []string{"text"}})
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			ctx, capture := WithNativeProviderFailure(ctx)
			opts := OpenAICompletionsStreamOptions{Client: server.Client()}
			if mode == "callback" {
				opts.OnResponse = func(context.Context, CompletionsResponse, json.RawMessage) error {
					return &completionsRequestError{status: 401, message: "callback"}
				}
			}
			if mode == "cancel" {
				opts.OnResponse = func(context.Context, CompletionsResponse, json.RawMessage) error { cancel(); return nil }
			}
			stream, err := StreamAntigravityPooled(ctx, model, NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent("hello")}}}), opts, source)
			if err != nil {
				t.Fatal(err)
			}
			wait, done := context.WithTimeout(context.Background(), 4*time.Second)
			defer done()
			result, err := stream.SnapshotResult(wait)
			if err != nil {
				t.Fatal(err)
			}
			if err = stream.WaitForEnd(wait); err != nil {
				t.Fatal(err)
			}
			if mode == "success" || mode == "rotate" || mode == "stream-auth" {
				if result.StopReason != "toolUse" || result.Content.Blocks.Len() != 3 || *result.Content.Blocks.Get(0).ThinkingSignature != "c2ln" || *result.Content.Blocks.Get(1).TextSignature != "dGV4dA==" || *result.Content.Blocks.Get(2).ThoughtSignature != "dG9vbA==" {
					t.Fatalf("lost native result: %+v", result)
				}
				if result.Usage.Input != 5 || result.Usage.Output != 3 || result.Usage.CacheRead != 3 || result.Usage.TotalTokens != 11 {
					t.Fatalf("usage: %+v", result.Usage)
				}
				if !result.Usage.Observation.UsageObserved || result.Usage.Observation.UsageSource != "wire" {
					t.Fatalf("Antigravity usage provenance missing: %+v", result.Usage)
				}
				if capture.Failure() != nil {
					t.Fatal("stale failed attempt")
				}
				var parsed completionsModel
				_ = json.Unmarshal(model, &parsed)
				replay, err := antigravityNativePayload(parsed, NormalizeContext(Context{Messages: []Message{*result, {Role: "toolResult", ToolCallID: "call", ToolName: "write", Content: BlockContent(ContentBlock{Type: "text", Text: "done"})}}}), nil, source.tokens[0], 1)
				if err != nil || !strings.Contains(string(replay), `"thoughtSignature":"dG9vbA=="`) || !strings.Contains(string(replay), `"output":"done"`) {
					t.Fatalf("replay: %s %v", replay, err)
				}
			} else if mode == "cancel" {
				if result.StopReason != "aborted" {
					t.Fatal(result.StopReason)
				}
			} else if result.StopReason != "error" {
				t.Fatal(result.StopReason)
			}
			want := 1
			if mode == "rotate" || mode == "stream-auth" {
				want = 2
			}
			if count != want {
				t.Fatalf("requests %d want %d", count, want)
			}
			if mode == "concurrency" && len(source.reports) != 0 {
				t.Fatal("concurrency cap penalized account")
			}
			if mode == "callback" && !capture.HostFailure() {
				t.Fatal("callback origin lost")
			}
			encoded, _ := json.Marshal(result)
			if strings.Contains(string(encoded), "fixture") || strings.Contains(string(encoded), "project") {
				t.Fatal("credential leaked")
			}
		})
	}
}

func TestAntigravityCrossProviderReplay(t *testing.T) {
	signature := "c2ln"
	source := Message{Role: "assistant", API: "openai-codex-responses", Provider: VendorOpenAICodex, Model: "codex", StopReason: "toolUse", Content: BlockContent(ContentBlock{Type: "toolCall", ID: "call|item", Name: "write", Arguments: json.RawMessage(`{}`), ThoughtSignature: &signature})}
	result := Message{Role: "toolResult", ToolCallID: "call|item", ToolName: "write", Content: BlockContent(ContentBlock{Type: "text", Text: "done"})}
	payload, err := antigravityNativePayload(completionsModel{ID: "gemini-3-pro", API: VendorGoogleAntigravity, Provider: VendorGoogleAntigravity, Input: []string{"text"}}, NormalizeContext(Context{Messages: []Message{source, result}}), nil, OAuthToken{ProjectID: "project"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(payload), `"id":"call_item"`) != 2 || strings.Contains(string(payload), "thoughtSignature") {
		t.Fatalf("cross-provider ID/signature mismatch: %s", payload)
	}
	if source.Content.Blocks.Get(0).ID != "call|item" || source.Content.Blocks.Get(0).ThoughtSignature == nil {
		t.Fatal("caller history was mutated")
	}
}
