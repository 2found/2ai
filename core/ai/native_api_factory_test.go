package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
)

var nativeAPIFactories = map[string]func(OpenAICompletionsStreamOptions) *ProviderStreams{
	"openai-completions": OpenAICompletionsAPI, "openai-responses": OpenAIResponsesAPI, "anthropic-messages": AnthropicMessagesAPI, "azure-openai-responses": AzureOpenAIResponsesAPI, "openai-codex-responses": OpenAICodexResponsesAPI,
}

func nativeFactoryServeAnswer(w http.ResponseWriter, api string) {
	w.Header().Set("Content-Type", "text/event-stream")
	if api == "anthropic-messages" {
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"answer\",\"model\":\"fixture\",\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	} else if api == "openai-completions" {
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n")
	} else {
		_, _ = io.WriteString(w, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"answer\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":3}}}\n\n")
	}
}
func nativeFactoryReadBody(t *testing.T, r *http.Request) []byte {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Error(err)
		return nil
	}
	if r.Header.Get("Content-Encoding") == "zstd" {
		decoder, err := zstd.NewReader(nil)
		if err != nil {
			t.Error(err)
			return nil
		}
		defer decoder.Close()
		body, err = decoder.DecodeAll(body, nil)
		if err != nil {
			t.Error(err)
		}
	}
	return body
}
func nativeFactoryToken(api string) string {
	if api == "openai-codex-responses" {
		return "h." + base64.StdEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_account_id":"account"}}`)) + ".s"
	}
	return "fixture-key"
}
func nativeFactoryModel(api, provider, endpoint string) *Object {
	return NewObject(Property{Name: "id", Value: "fixture"}, Property{Name: "name", Value: "Fixture"}, Property{Name: "api", Value: api}, Property{Name: "provider", Value: provider}, Property{Name: "baseUrl", Value: endpoint}, Property{Name: "reasoning", Value: false}, Property{Name: "input", Value: NewArray("text")}, Property{Name: "maxTokens", Value: 100}, Property{Name: "contextWindow", Value: 1000}, Property{Name: "cost", Value: NewObject(Property{Name: "input", Value: 1}, Property{Name: "output", Value: 2}, Property{Name: "cacheRead", Value: 0}, Property{Name: "cacheWrite", Value: 0})})
}

func TestNativeAPIFactoriesHTTP(t *testing.T) {
	for api, create := range nativeAPIFactories {
		t.Run(api, func(t *testing.T) {
			for _, simple := range []bool{false, true} {
				t.Run(map[bool]string{false: "stream", true: "simple"}[simple], func(t *testing.T) {
					var requests atomic.Int32
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						requests.Add(1)
						body := nativeFactoryReadBody(t, r)
						if !strings.Contains(string(body), `"factoryMarker":true`) {
							t.Errorf("payload callback lost: %s", body)
						}
						if r.Header.Get("X-Factory") != "request" {
							t.Error("request headers lost")
						}
						nativeFactoryServeAnswer(w, api)
					}))
					defer server.Close()
					// The request-specific client must override the default client.
					defaults := OpenAICompletionsStreamOptions{Client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
						t.Error("default client used despite fetch override")
						return nil, context.Canceled
					})}, Now: func() int64 { return 1000000 }}
					factory := create(defaults)
					model := nativeFactoryModel(api, "fixture", server.URL+"/v1")
					values := NewObject(Property{Name: "apiKey", Value: nativeFactoryToken(api)}, Property{Name: "transport", Value: "sse"}, Property{Name: "maxRetries", Value: 0}, Property{Name: "fetch", Value: server.Client()}, Property{Name: "headers", Value: NewObject(Property{Name: "X-Factory", Value: "request"})})
					callbacks := []string{}
					values.Set("onPayload", func(_ context.Context, raw, model json.RawMessage) (json.RawMessage, error) {
						callbacks = append(callbacks, "payload")
						var payload map[string]json.RawMessage
						if err := json.Unmarshal(raw, &payload); err != nil {
							return nil, err
						}
						payload["factoryMarker"] = json.RawMessage(`true`)
						return json.Marshal(payload)
					})
					values.Set("onResponse", func(response CompletionsResponse, _ json.RawMessage) error {
						callbacks = append(callbacks, "response")
						if response.Status != 200 {
							t.Error("wrong response")
						}
						return nil
					})
					values.Set("onProviderStreamEvent", func(*json.RawMessage, json.RawMessage) error { callbacks = append(callbacks, "event"); return nil })
					call := factory.Stream
					if simple {
						call = factory.StreamSimple
					}
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
					defer cancel()
					source, err := call(ctx, model, NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent("hello"), Timestamp: 1}}}), values)
					if err != nil {
						t.Fatal(err)
					}
					for {
						_, ok, err := source.Next(ctx)
						if err != nil {
							t.Fatal(err)
						}
						if !ok {
							break
						}
					}
					result, ok, err := source.Result(ctx)
					if err != nil || !ok {
						t.Fatalf("result: %v", err)
					}
					if result.StopReason != "stop" || result.Content.Blocks.Len() != 1 || result.Content.Blocks.Get(0).Text != "answer" || requests.Load() != 1 {
						t.Fatalf("wrong native result: %s", mustFactoryJSON(t, result))
					}
					if result.Usage == nil || !result.Usage.Observation.UsageObserved || !result.Usage.Observation.PricingObserved || result.Usage.Observation.UsageSource != "wire" || result.Usage.Observation.PricingSource != "model" {
						t.Fatalf("native %s provenance missing: %+v", api, result.Usage)
					}
					if len(callbacks) < 3 || callbacks[0] != "payload" || callbacks[1] != "response" {
						t.Fatalf("callback order: %v", callbacks)
					}
				})
			}
		})
	}
}
func mustFactoryJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestNativeAPIFactoriesAdmissionFailures(t *testing.T) {
	for api, create := range nativeAPIFactories {
		t.Run(api, func(t *testing.T) {
			factory := create(OpenAICompletionsStreamOptions{Now: func() int64 { return 1000000 }})
			if factory.FetchDeferred != nil || factory.CancelDeferred != nil {
				t.Fatal("lazy factory invented deferred capability")
			}
			for _, call := range []ModelStreamFunc{factory.Stream, factory.StreamSimple} {
				source, err := call(context.Background(), nativeFactoryModel(api, "fixture", "https://unused.test"), NormalizeContext(Context{}), NewObject(Property{Name: "fetch", Value: "invalid"}))
				if err != nil {
					t.Fatal("lazy API rejected synchronously", err)
				}
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				result, ok, err := source.Result(ctx)
				if err != nil || !ok || result.StopReason != "error" || result.ErrorMessage == nil || *result.ErrorMessage != "native fetch must be an *http.Client" {
					t.Fatalf("wrong lazy failure %v %v", result, err)
				}
			}
		})
	}
}
