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
	"time"

	"github.com/2found/2ai/ai/protocol"
)

func TestClaudeCodeNativePoolHTTP(t *testing.T) {
	for _, mode := range []string{"success", "rotate", "repeat", "quota", "concurrency", "callback"} {
		t.Run(mode, func(t *testing.T) {
			source := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a", AccessToken: "first-fixture"}, {AccountID: "b", AccessToken: "second-fixture"}}}
			if mode == "repeat" {
				source.tokens = source.tokens[:1]
			}
			requests := 0
			keys := []string{}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests++
				keys = append(keys, r.Header.Get("Authorization"))
				if r.Header.Get("X-Api-Key") != "" || !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer ") {
					t.Error("Claude Code was not Bearer authenticated")
				}
				if !strings.Contains(r.Header.Get("Anthropic-Beta"), "oauth-2025-04-20") || !strings.HasPrefix(r.Header.Get("User-Agent"), "claude-cli/") {
					t.Error("Claude Code fingerprint missing")
				}
				raw, _ := io.ReadAll(r.Body)
				if !strings.Contains(string(raw), "You are Claude Code, Anthropic's official CLI for Claude.") {
					t.Error("OAuth system prompt missing")
				}
				if requests == 1 && mode != "success" && mode != "callback" {
					status := 401
					message := "expired"
					if mode == "quota" {
						status = 429
					}
					if mode == "concurrency" {
						status = 403
						message = "concurrency cap"
					}
					w.Header().Set("Retry-After", "2")
					w.WriteHeader(status)
					_, _ = w.Write([]byte(`{"error":{"message":"` + message + `"}}`))
					return
				}
				nativeFactoryServeAnswer(w, "anthropic-messages")
			}))
			defer server.Close()
			rawModel, _ := json.Marshal(nativeFactoryModel("anthropic-messages", VendorClaudeCode, server.URL))
			opts := AnthropicStreamOptions{Client: server.Client(), Options: json.RawMessage(`{"apiKey":"stale-key","maxRetries":0}`)}
			if mode == "callback" {
				opts.OnResponse = func(context.Context, CompletionsResponse, json.RawMessage) error {
					return &completionsRequestError{status: 401, message: "callback unauthorized"}
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			ctx, capture := WithNativeProviderFailure(ctx)
			stream, err := StreamClaudeCodePooled(ctx, rawModel, NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent("hello")}}}), opts, source)
			if err != nil {
				t.Fatal(err)
			}
			starts, terminals := 0, 0
			for {
				event, ok, err := stream.Next(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !ok {
					break
				}
				if event.Type == "start" {
					starts++
				}
				if event.Type == "done" || event.Type == "error" {
					terminals++
				}
				if _, err = stream.SnapshotEvent(event); err != nil {
					t.Fatal(err)
				}
			}
			result, err := stream.SnapshotResult(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if err = stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			wantRequests := 1
			if mode == "rotate" {
				wantRequests = 2
			}
			if requests != wantRequests || terminals != 1 {
				t.Fatalf("requests=%d terminals=%d", requests, terminals)
			}
			if mode == "success" || mode == "rotate" {
				if result.StopReason != "stop" || starts != 1 || capture.Failure() != nil {
					t.Fatalf("unsuccessful pool result: %s", mustFactoryJSON(t, result))
				}
				if source.reports[len(source.reports)-1].err != nil {
					t.Fatal("successful account reported failure")
				}
			} else if result.StopReason != "error" {
				t.Fatal("failed account returned success")
			}
			if mode == "rotate" {
				var reported *protocol.ProviderError
				if source.acquireN != 2 || len(source.reports) != 2 || !errors.As(source.reports[0].err, &reported) || reported.Status != 401 || reported.Provider != VendorClaudeCode || strings.Join(keys, ",") != "Bearer first-fixture,Bearer second-fixture" {
					t.Fatal("incorrect rotation/report lifecycle")
				}
			}
			if mode == "callback" && (isOAuthAuthFailure(source.reports[0].err) || !capture.HostFailure()) {
				t.Fatal("callback failure treated as account rejection")
			}
			if mode == "quota" {
				var failure *protocol.ProviderError
				if !errors.As(capture.Failure(), &failure) || failure.Status != 429 || failure.RetryAfter != 2*time.Second {
					t.Fatal("host lost retry metadata")
				}
			}
			if mode == "concurrency" && len(source.reports) != 0 {
				t.Fatal("concurrency cap blamed on account")
			}
			for _, token := range source.tokens {
				if strings.Contains(mustFactoryJSON(t, result), token.AccessToken) {
					t.Fatal("credential leaked into native result")
				}
			}
		})
	}
}

func TestNativeOAuthPoolDoesNotRotateAfterContent(t *testing.T) {
	for _, kind := range []string{"text_delta", "thinking_delta", "toolcall_start"} {
		t.Run(kind, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			source := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a", AccessToken: "a"}, {AccountID: "b", AccessToken: "b"}}}
			var partial *Message
			out, err := nativeOAuthPoolStream(ctx, json.RawMessage(`{"id":"m","api":"anthropic-messages","provider":"claude-code"}`), NormalizeContext(Context{}), AnthropicStreamOptions{}, source, "Claude Code", func(ctx context.Context, _ json.RawMessage, _ TranscriptContext, _ OpenAICompletionsStreamOptions, _ OAuthToken) (*AssistantMessageEventStream, error) {
				stream := NewAssistantMessageEventStreamFor(ctx)
				partial = &Message{Role: "assistant", Content: TextContent("visible"), Usage: &Usage{}, StopReason: "error"}
				stream.Push(AssistantMessageEvent{Type: "start", Partial: partial})
				stream.Push(AssistantMessageEvent{Type: kind, Delta: "visible", Partial: partial})
				recordNativeFailure(ctx, VendorClaudeCode, &completionsRequestError{status: 401, message: "unauthorized"}, false)
				stream.Push(AssistantMessageEvent{Type: "error", Reason: "error", Error: partial})
				stream.End()
				return stream, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			result, err := out.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if err = out.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			if result != partial || source.acquireN != 1 || len(source.reports) != 1 || !isOAuthAuthFailure(source.reports[0].err) {
				t.Fatal("visible content was replayed or its identity replaced")
			}
		})
	}
}

func TestClaudeCodeNativePoolCancellation(t *testing.T) {
	for _, before := range []bool{true, false} {
		t.Run(map[bool]string{true: "before acquire", false: "during request"}[before], func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			source := &fakeTokenSource{tokens: []OAuthToken{{AccountID: "a", AccessToken: "fixture"}}}
			options := AnthropicStreamOptions{Options: json.RawMessage(`{"maxRetries":0}`), Client: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) { <-r.Context().Done(); return nil, r.Context().Err() })}}
			if before {
				cancel()
			} else {
				options.OnPayload = func(context.Context, json.RawMessage, json.RawMessage) (json.RawMessage, error) {
					cancel()
					return nil, nil
				}
			}
			stream, err := StreamClaudeCodePooled(ctx, json.RawMessage(`{"id":"m","api":"anthropic-messages","provider":"claude-code","baseUrl":"https://unused.test","maxTokens":4096,"contextWindow":10000,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}}`), NormalizeContext(Context{}), options, source)
			if err != nil {
				t.Fatal(err)
			}
			reader, stop := context.WithTimeout(context.Background(), time.Second)
			defer stop()
			result, err := stream.SnapshotResult(reader)
			if err != nil {
				t.Fatal(err)
			}
			if err = stream.WaitForEnd(reader); err != nil {
				t.Fatal(err)
			}
			want := 1
			if before {
				want = 0
			}
			if result.StopReason != "aborted" || source.acquireN != want {
				t.Fatalf("reason=%s acquisitions=%d", result.StopReason, source.acquireN)
			}
		})
	}
}
