package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
)

func TestPiCodexHTTPOracle(t *testing.T)   { checkCodexHTTPOracle(t, false) }
func TestPiCodexSimpleOracle(t *testing.T) { checkCodexHTTPOracle(t, true) }
func checkCodexHTTPOracle(t *testing.T, simple bool) {
	file, count := "testdata/pi-codex-http.json", 24
	if simple {
		file, count = "testdata/pi-codex-simple.json", 44
	}
	raw, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          map[string]json.RawMessage
		Context        Context
		Token          string
		Cases          []struct {
			Input struct {
				Model                                                        map[string]json.RawMessage
				Name                                                         string
				Headers, Payload                                             json.RawMessage
				Options                                                      map[string]json.RawMessage
				PayloadFailure, ResponseFailure, EventFailure, EventMutation bool
				Responses                                                    []struct {
					Status  int
					Body    *string
					Headers map[string]string
				}
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != count {
		t.Fatal("unexpected HTTP oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			chosen := map[string]json.RawMessage{}
			for key, value := range fixture.Model {
				chosen[key] = value
			}
			if len(tc.Input.Headers) > 0 {
				chosen["headers"] = tc.Input.Headers
			}
			for key, value := range tc.Input.Model {
				chosen[key] = value
			}
			model, _ := json.Marshal(chosen)
			controls := map[string]json.RawMessage{"apiKey": json.RawMessage(marshalSamplingString(fixture.Token)), "transport": json.RawMessage(`"sse"`)}
			for key, value := range tc.Input.Options {
				controls[key] = value
			}
			options, _ := json.Marshal(controls)
			requests := []any{}
			callbacks := []string{}
			attempt := 0
			decoder, err := zstd.NewReader(nil)
			if err != nil {
				t.Fatal(err)
			}
			defer decoder.Close()
			client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				headers := map[string]string{}
				for key, values := range req.Header {
					headers[strings.ToLower(key)] = strings.Join(values, ", ")
				}
				headers["user-agent"] = "<platform>"
				compressed, err := io.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				_ = req.Body.Close()
				body, err := decoder.DecodeAll(compressed, nil)
				if err != nil {
					return nil, err
				}
				requests = append(requests, map[string]any{"url": req.URL.String(), "headers": headers, "body": json.RawMessage(body)})
				status := 200
				data := `data: {"type":"response.done","response":{"id":"response","status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}` + "\n\n"
				responseHeaders := http.Header{"Content-Type": []string{"text/event-stream"}, "X-Fixture": []string{"yes"}}
				if attempt < len(tc.Input.Responses) {
					spec := tc.Input.Responses[attempt]
					if spec.Status != 0 {
						status = spec.Status
					}
					if spec.Body != nil {
						data = *spec.Body
					}
					for key, value := range spec.Headers {
						responseHeaders.Set(key, value)
					}
				}
				attempt++
				return &http.Response{StatusCode: status, Header: responseHeaders, Body: io.NopCloser(strings.NewReader(data)), Request: req}, nil
			})}
			streamOptions := CodexResponsesStreamOptions{Options: options, Client: client, Now: func() int64 { return 100 },
				OnPayload: func(context.Context, json.RawMessage, json.RawMessage) (json.RawMessage, error) {
					callbacks = append(callbacks, "payload")
					if tc.Input.PayloadFailure {
						return nil, errors.New("payload failure")
					}
					return tc.Input.Payload, nil
				},
				OnResponse: func(context.Context, CompletionsResponse, json.RawMessage) error {
					callbacks = append(callbacks, "response")
					if tc.Input.ResponseFailure {
						return errors.New("response failure")
					}
					return nil
				},
				OnProviderStreamEvent: func(_ context.Context, raw *json.RawMessage, _ json.RawMessage) error {
					callbacks = append(callbacks, "event")
					if tc.Input.EventFailure {
						return errors.New("event failure")
					}
					if tc.Input.EventMutation {
						*raw = json.RawMessage(strings.Replace(string(*raw), `"id":"response"`, `"id":"mutated"`, 1))
					}
					return nil
				},
			}
			var stream *AssistantMessageEventStream
			var admission error
			if simple {
				stream, admission = StreamCodexResponsesSimple(ctx, model, NormalizeContext(fixture.Context), streamOptions)
			} else {
				stream = StreamCodexResponses(ctx, model, NormalizeContext(fixture.Context), streamOptions)
			}
			var result *Message
			if admission == nil {
				result, err = stream.Result(ctx)
				if err != nil {
					t.Fatal(err)
				}
			}
			actual := map[string]any{"requests": requests, "callbacks": callbacks, "result": result}
			if simple {
				var failure *string
				if admission != nil {
					value := admission.Error()
					failure = &value
				}
				actual["error"] = failure
			}
			assertPiJSON(t, tc.Expected, actual)
		})
	}
}

func TestCodexSSEHTTPTimeoutAndCancellation(t *testing.T) {
	for _, mode := range []string{"headers", "body", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			observed := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				if mode == "headers" {
					<-r.Context().Done()
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(200)
				w.(http.Flusher).Flush()
				if mode == "cancel" {
					<-r.Context().Done()
					return
				}
				// Hold the body beyond the configured header timeout. Receiving headers
				// must remove that deadline without severing the caller's cancellation.
				select {
				case <-r.Context().Done():
					return
				case <-time.After(75 * time.Millisecond):
				}
				_, _ = io.WriteString(w, "data: {\"type\":\"response.done\",\"response\":{\"status\":\"completed\",\"output\":[]}}\n\n")
			}))
			defer server.Close()
			token := "header." + base64.StdEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_account_id":"account"}}`)) + ".signature"
			model, _ := json.Marshal(map[string]any{"id": "test", "api": "openai-codex-responses", "provider": "openai-codex", "baseUrl": server.URL})
			options, _ := json.Marshal(map[string]any{"apiKey": token, "timeoutMs": 20})
			stream := StreamCodexResponsesSSE(ctx, model, NormalizeContext(Context{}), CodexResponsesStreamOptions{Options: options, OnResponse: func(context.Context, CompletionsResponse, json.RawMessage) error { close(observed); return nil }})
			if mode == "cancel" {
				select {
				case <-observed:
					cancel()
				case <-ctx.Done():
					t.Fatal("headers never arrived")
				}
			}
			wait, stop := context.WithTimeout(context.Background(), time.Second)
			defer stop()
			result, err := stream.Result(wait)
			if err != nil {
				t.Fatal(err)
			}
			switch mode {
			case "headers":
				if result.StopReason != "error" || result.ErrorMessage == nil || *result.ErrorMessage != "Codex SSE response headers timed out after 20ms" {
					t.Fatalf("header timeout: %+v", result)
				}
			case "body":
				if result.StopReason != "stop" {
					t.Fatalf("header deadline leaked into body: %+v", result)
				}
			case "cancel":
				if result.StopReason != "aborted" || result.ErrorMessage == nil || *result.ErrorMessage != "Request was aborted" {
					t.Fatalf("cancel: %+v", result)
				}
			}
		})
	}
}
