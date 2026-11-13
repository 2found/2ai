package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestPiOpenAICompletionsHTTPOracle(t *testing.T) {
	testOpenAIHTTPOracle(t, "testdata/pi-completions-http.json", 47, StreamOpenAICompletions)
}
func TestPiOpenAIResponsesHTTPOracle(t *testing.T) {
	testOpenAIHTTPOracle(t, "testdata/pi-responses-http.json", 55, StreamOpenAIResponses)
}
func TestPiAnthropicHTTPOracle(t *testing.T) {
	testOpenAIHTTPOracle(t, "testdata/pi-anthropic-http.json", 62, StreamAnthropic)
}
func testOpenAIHTTPOracle(t *testing.T, fixturePath string, count int, streamFn func(context.Context, json.RawMessage, TranscriptContext, OpenAICompletionsStreamOptions) *AssistantMessageEventStream) {
	clearAnthropicFederationEnv(t)
	for _, name := range []string{"OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "PI_CACHE_RETENTION", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_RESOURCE_NAME", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME_MAP"} {
		t.Setenv(name, "")
	}
	t.Setenv("ANTHROPIC_CUSTOM_HEADERS", "")
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, SDKVersion, Success string
		Model                               map[string]json.RawMessage
		IgnoredHeaders                      []string
		Cases                               []struct {
			Input struct {
				Name, Fail     string
				Model, Options map[string]json.RawMessage
				Context        Context
				Replacement    json.RawMessage
				Mutate         bool
				Responses      []struct {
					Status  int
					Body    string
					Headers map[string]string
				}
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	sdkVersion := "7.19.0"
	if samplingString(fixture.Model["api"]) == "anthropic-messages" {
		sdkVersion = "0.129.0"
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || fixture.SDKVersion != sdkVersion || len(fixture.Cases) != count {
		t.Fatal("unexpected HTTP oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			var mu sync.Mutex
			requests := []any{}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
					return
				}
				headers := map[string]string{}
				for name, values := range r.Header {
					headers[strings.ToLower(name)] = strings.Join(values, ", ")
				}
				for _, key := range append(fixture.IgnoredHeaders, "accept-encoding", "content-length") {
					delete(headers, key)
				}
				var payload json.RawMessage
				if len(body) > 0 {
					payload = body
				}
				mu.Lock()
				request := map[string]any{"path": r.URL.Path, "headers": headers, "body": payload}
				if samplingString(fixture.Model["api"]) == "anthropic-messages" || samplingString(fixture.Model["api"]) == "azure-openai-responses" {
					request["query"] = r.URL.RawQuery
				}
				requests = append(requests, request)
				index := len(requests) - 1
				mu.Unlock()
				status, text := 200, fixture.Success
				w.Header()["Date"] = nil
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("X-Test", "response")
				if len(tc.Input.Responses) > 0 {
					response := tc.Input.Responses[min(index, len(tc.Input.Responses)-1)]
					if response.Status != 0 {
						status = response.Status
					}
					text = response.Body
					for name, value := range response.Headers {
						w.Header().Set(name, value)
					}
				}
				w.WriteHeader(status)
				w.(http.Flusher).Flush()
				// Small writes allow the transport to split UTF-8 and CRLF boundaries.
				for i := 0; i < len(text); i++ {
					_, _ = w.Write([]byte{text[i]})
				}
			}))
			defer server.Close()
			model := map[string]json.RawMessage{}
			for name, value := range fixture.Model {
				model[name] = value
			}
			for name, value := range tc.Input.Model {
				model[name] = value
			}
			modelHeaders, _ := samplingObject(fixture.Model["headers"])
			extraHeaders, _ := samplingObject(tc.Input.Model["headers"])
			for name, value := range extraHeaders {
				modelHeaders[name] = value
			}
			// Preserve source key insertion order for the case-collision fixture.
			keys := samplingObjectKeys(fixture.Model["headers"])
			for _, name := range samplingObjectKeys(tc.Input.Model["headers"]) {
				found := false
				for _, key := range keys {
					if key == name {
						found = true
					}
				}
				if !found {
					keys = append(keys, name)
				}
			}
			model["headers"] = marshalSamplingObject(modelHeaders, keys)
			model["baseUrl"], _ = json.Marshal(server.URL + "/v1")
			rawModel, _ := json.Marshal(model)
			controls := map[string]json.RawMessage{"apiKey": json.RawMessage(`"fixture-key"`)}
			for name, value := range tc.Input.Options {
				controls[name] = value
			}
			rawControls, _ := json.Marshal(controls)
			callbacks := []any{}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			stream := streamFn(ctx, rawModel, NormalizeContext(tc.Input.Context), OpenAICompletionsStreamOptions{
				Options: rawControls, Client: server.Client(), Now: func() int64 { return 100 },
				OnPayload: func(_ context.Context, payload, model json.RawMessage) (json.RawMessage, error) {
					callbacks = append(callbacks, map[string]any{"type": "payload", "payload": append(json.RawMessage(nil), payload...)})
					if tc.Input.Fail == "payload" {
						return nil, errors.New("payload failed")
					}
					return tc.Input.Replacement, nil
				},
				OnResponse: func(_ context.Context, response CompletionsResponse, _ json.RawMessage) error {
					callbacks = append(callbacks, map[string]any{"type": "response", "status": response.Status, "headers": response.Headers})
					if tc.Input.Fail == "response" {
						return errors.New("response failed")
					}
					return nil
				},
				OnProviderStreamEvent: func(_ context.Context, chunk *json.RawMessage, _ json.RawMessage) error {
					callbacks = append(callbacks, map[string]any{"type": "chunk", "chunk": append(json.RawMessage(nil), (*chunk)...)})
					if tc.Input.Fail == "chunk" {
						return errors.New("chunk failed")
					}
					if tc.Input.Mutate {
						var value map[string]any
						if json.Unmarshal(*chunk, &value) == nil {
							if value["type"] == "content_block_delta" {
								if delta, ok := value["delta"].(map[string]any); ok && delta["type"] == "text_delta" {
									delta["text"] = "changed"
									*chunk, _ = json.Marshal(value)
								}
							}
							if item, ok := value["item"].(map[string]any); ok {
								if content, ok := item["content"].([]any); ok && len(content) > 0 {
									content[0].(map[string]any)["text"] = "changed"
									*chunk, _ = json.Marshal(value)
								}
							}

							if choices, ok := value["choices"].([]any); ok && len(choices) > 0 {
								choice := choices[0].(map[string]any)
								if delta, ok := choice["delta"].(map[string]any); ok {
									delta["content"] = "changed"
									*chunk, _ = json.Marshal(value)
								}
							}
						}
					}
					return nil
				},
			})
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			result, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			events := []AssistantMessageEvent{}
			for {
				event, ok, err := stream.Next(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !ok {
					break
				}
				events = append(events, event)
			}
			mu.Lock()
			actual, err := json.Marshal(map[string]any{"requests": requests, "callbacks": callbacks, "events": events, "result": result})
			mu.Unlock()
			if err != nil {
				t.Fatal(err)
			}
			decode := func(raw []byte) any {
				var value any
				d := json.NewDecoder(bytes.NewReader(raw))
				d.UseNumber()
				if err := d.Decode(&value); err != nil {
					t.Fatal(err)
				}
				return value
			}
			if !reflect.DeepEqual(decode(actual), decode(tc.Expected)) {
				t.Fatalf("HTTP mismatch\nGo: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}

func TestCompletionsHTTPTimeoutAndAbort(t *testing.T) {
	testOpenAIHTTPTimeoutAndAbort(t, "openai-completions", StreamOpenAICompletions)
}
func TestResponsesHTTPTimeoutAndAbort(t *testing.T) {
	testOpenAIHTTPTimeoutAndAbort(t, "openai-responses", StreamOpenAIResponses)
}
func TestAnthropicHTTPTimeoutAndAbort(t *testing.T) {
	testOpenAIHTTPTimeoutAndAbort(t, "anthropic-messages", StreamAnthropic)
}
func testOpenAIHTTPTimeoutAndAbort(t *testing.T, api string, streamFn func(context.Context, json.RawMessage, TranscriptContext, OpenAICompletionsStreamOptions) *AssistantMessageEventStream) {
	for _, scenario := range []string{"header timeout", "body outlives header timeout", "abort body"} {
		t.Run(scenario, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			bodyStarted := make(chan struct{})
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				if scenario == "header timeout" {
					select {
					case <-r.Context().Done():
					case <-release:
					}
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(200)
				w.(http.Flusher).Flush()
				close(bodyStarted)
				if scenario == "abort body" {
					select {
					case <-r.Context().Done():
					case <-release:
					}
					return
				}
				timer := time.NewTimer(100 * time.Millisecond)
				defer timer.Stop()
				select {
				case <-timer.C:
				case <-r.Context().Done():
					return
				}

				if api == "anthropic-messages" {
					_, _ = io.WriteString(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"finished\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n")
					return
				}
				if api == "openai-responses" || api == "azure-openai-responses" {
					_, _ = io.WriteString(w, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"content\":[{\"type\":\"output_text\",\"text\":\"finished\"}]}}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
					return
				}
				_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"finished\"},\"finish_reason\":\"stop\"}]}\n\n")
			}))
			defer func() { close(release); server.Close() }()
			model, _ := json.Marshal(map[string]any{"id": "test", "api": api, "provider": "openai", "baseUrl": server.URL})
			stream := streamFn(ctx, model, NormalizeContext(Context{}), OpenAICompletionsStreamOptions{Client: server.Client(), Options: json.RawMessage(`{"apiKey":"key","timeoutMs":30}`)})
			if scenario == "abort body" {
				<-bodyStarted
				cancel()
			}
			waitCtx, stop := context.WithTimeout(context.Background(), time.Second)
			defer stop()
			if err := stream.WaitForEnd(waitCtx); err != nil {
				t.Fatal(err)
			}
			result, err := stream.SnapshotResult(waitCtx)
			if err != nil {
				t.Fatal(err)
			}
			switch scenario {
			case "header timeout":
				if result.StopReason != "error" || result.ErrorMessage == nil || *result.ErrorMessage != "Request timed out." {
					t.Fatalf("%+v", result)
				}
			case "body outlives header timeout":
				if result.StopReason != "stop" || result.Content.Blocks.Get(0).Text != "finished" {
					t.Fatalf("%+v", result)
				}
			case "abort body":
				if result.StopReason != "aborted" {
					t.Fatalf("%+v", result)
				}
			}
		})
	}
}
