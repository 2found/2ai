package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

func TestNativeFailureHTTPMetadataKeepsWireUnchanged(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(503)
		_, _ = w.Write([]byte(`{"error":{"message":"overloaded"}}`))
	}))
	defer server.Close()
	for _, tc := range []struct {
		name, api string
		start     func(context.Context, json.RawMessage, TranscriptContext, OpenAICompletionsStreamOptions) *AssistantMessageEventStream
	}{
		{"openai", "openai-completions", StreamOpenAICompletions},
		{"openai", "openai-responses", StreamOpenAIResponses},
		{"anthropic", "anthropic-messages", StreamAnthropic},
		{"azure-openai-responses", "azure-openai-responses", StreamAzureResponses},
		{"radius", "pi-messages", func(ctx context.Context, raw json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions) *AssistantMessageEventStream {
			settings := PiMessagesStreamOptions{Values: catalogDecode(t, options.Options).(*Object), Now: func() float64 { return float64(options.Now()) }}
			if options.OnPayload != nil {
				settings.OnPayload = func(ctx context.Context, payload any, model *Object) (any, error) {
					body, err := json.Marshal(payload)
					if err != nil {
						return nil, err
					}
					next, err := options.OnPayload(ctx, body, raw)
					if err != nil || next == nil {
						return Undefined, err
					}
					return catalogDecode(t, next), nil
				}
			}
			return StreamPiMessages(ctx, catalogDecode(t, raw).(*Object), transcript, settings)
		}},
	} {
		t.Run(tc.api, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			raw, _ := json.Marshal(map[string]any{"id": "fixture", "api": tc.api, "provider": tc.name, "baseUrl": server.URL, "maxTokens": 100, "contextWindow": 1000, "input": []string{"text"}})
			options := OpenAICompletionsStreamOptions{Options: json.RawMessage(`{"apiKey":"fixture","maxRetries":0}`), Now: func() int64 { return 1 }}
			baseline := tc.start(ctx, raw, TranscriptContext{}, options)
			before, err := baseline.SnapshotResult(ctx)
			if err != nil {
				t.Fatal(err)
			}
			observed, capture := WithNativeProviderFailure(ctx)
			stream := tc.start(observed, raw, TranscriptContext{}, options)
			if err = stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			after, err := stream.SnapshotResult(ctx)
			if err != nil {
				t.Fatal(err)
			}
			a, _ := json.Marshal(before)
			b, _ := json.Marshal(after)
			if string(a) != string(b) {
				t.Fatalf("failure observation changed wire: %s / %s", a, b)
			}
			var failure *protocol.ProviderError
			if !errors.As(capture.Failure(), &failure) || failure.Status != 503 || failure.RetryAfter != 7*time.Second || !protocol.IsRetryable(failure) || capture.HostFailure() {
				t.Fatalf("metadata missing: %v", capture.Failure())
			}
			failure.Status = 401
			if capture.Failure().(*protocol.ProviderError).Status != 503 {
				t.Fatal("returned metadata aliases capture")
			}
			options.OnPayload = func(context.Context, json.RawMessage, json.RawMessage) (json.RawMessage, error) {
				return nil, &protocol.ProviderError{Status: 503, Message: "callback failed"}
			}
			observed, capture = WithNativeProviderFailure(ctx)
			stream = tc.start(observed, raw, TranscriptContext{}, options)
			if err = stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			if !capture.HostFailure() || protocol.IsRetryable(capture.Failure()) {
				t.Fatal("callback error reclassified as provider failure")
			}
		})
	}
}

func TestNativeFailureTypedProviderBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name     string
		cause    error
		callback bool
		status   int
		typed    bool
	}{
		{"codex", &codexHTTPError{Status: 429, Headers: http.Header{"Retry-After": []string{"7"}}, Message: "quota"}, false, 429, true},
		{"anthropic-client", &AnthropicClientError{Status: 503, Headers: http.Header{"Retry-After": []string{"7"}}, Message: "busy"}, false, 503, true},
		{"callback-codex", &codexHTTPError{Status: 503, Message: "callback"}, true, 0, false},
		{"callback-anthropic", &AnthropicClientError{Status: 503, Message: "callback"}, true, 0, false},
		{"foreign-provider-error", &protocol.ProviderError{Status: 503, Message: "injected"}, false, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, capture := WithNativeProviderFailure(context.Background())
			recordNativeFailure(ctx, tc.name, tc.cause, tc.callback)
			var typed *protocol.ProviderError
			if errors.As(capture.Failure(), &typed) != tc.typed {
				t.Fatal("wrong failure provenance", capture.Failure())
			}
			if tc.typed && (typed.Status != tc.status || typed.RetryAfter != 7*time.Second) {
				t.Fatal("typed metadata lost", typed)
			}
			if capture.HostFailure() != tc.callback {
				t.Fatal("callback provenance lost")
			}
			recordNativeFailure(ctx, tc.name, nil, false)
			if capture.Failure() != nil || capture.HostFailure() {
				t.Fatal("next provider attempt retained stale failure")
			}
		})
	}
}
