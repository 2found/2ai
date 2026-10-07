package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/2found/2ai/jsonjs"
)

func TestPiProviderText(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-provider-text.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Name, Provider string
			Model, Options json.RawMessage
			Messages       []Message
			Expected       json.RawMessage
			HTTPExpected   json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 252 {
		t.Fatal("unexpected provider text oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			transcript := NormalizeContext(Context{Messages: tc.Messages})
			var actual json.RawMessage
			var err error
			switch tc.Provider {
			case "gemini":
				actual, err = BuildOpenAICompletionsParams(tc.Model, transcript, tc.Options)
			case "codex":
				actual, err = BuildCodexResponsesParams(tc.Model, transcript, tc.Options, nil)
			case "claude":
				actual, err = BuildAnthropicParams(tc.Model, transcript, true, tc.Options)
			default:
				t.Fatal("unknown provider", tc.Provider)
			}
			if err != nil {
				t.Fatal(err)
			}
			got, err := jsonjs.DecodeJSON(actual)
			if err != nil {
				t.Fatal(err)
			}
			want, err := jsonjs.DecodeJSON(tc.Expected)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
			if strings.Contains(tc.Name, "/mixed/") {
				t.Run("http", func(t *testing.T) {
					expected := tc.Expected
					if len(tc.HTTPExpected) > 0 {
						expected = tc.HTTPExpected
					}
					checkProviderTextHTTP(t, tc.Provider, tc.Model, transcript, tc.Options, expected)
				})
			}
		})
	}
}

func checkProviderTextHTTP(t *testing.T, provider string, model json.RawMessage, transcript TranscriptContext, rawOptions, expected json.RawMessage) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var controls map[string]json.RawMessage
	if err := json.Unmarshal(rawOptions, &controls); err != nil {
		t.Fatal(err)
	}
	key := "test"
	if provider == "codex" {
		key = "header." + base64.RawURLEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_account_id":"test"}}`)) + ".signature"
		controls["transport"] = json.RawMessage(`"sse"`)
	} else if provider == "claude" {
		key = "sk-ant-oat01-test"
	}
	controls["apiKey"] = jsonjs.QuoteString(key)
	options, err := json.Marshal(controls)
	if err != nil {
		t.Fatal(err)
	}
	captured := make(chan []byte, 1)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		if request.Header.Get("Content-Encoding") == "zstd" {
			decoder, err := zstd.NewReader(nil)
			if err != nil {
				return nil, err
			}
			body, err = decoder.DecodeAll(body, nil)
			decoder.Close()
			if err != nil {
				return nil, err
			}
		}
		select {
		case captured <- body:
		default:
			return nil, fmt.Errorf("unexpected retry")
		}
		response := `data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n"
		if provider == "codex" {
			response = `data: {"type":"response.done","response":{"id":"r","status":"completed","output":[],"usage":{"input_tokens":0,"output_tokens":0}}}` + "\n\n"
		}
		if provider == "claude" {
			response = "event: message_delta\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}` + "\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
		}
		return &http.Response{StatusCode: 200, Status: "200 OK", Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(response)), Request: request}, nil
	})}
	streamOptions := OpenAICompletionsStreamOptions{Options: options, Client: client}
	var stream *AssistantMessageEventStream
	switch provider {
	case "gemini":
		stream = StreamOpenAICompletions(ctx, model, transcript, streamOptions)
	case "codex":
		stream = StreamCodexResponses(ctx, model, transcript, streamOptions)
	case "claude":
		stream = StreamAnthropic(ctx, model, transcript, streamOptions)
	}
	result, err := stream.Result(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "stop" {
		t.Fatalf("transport: %+v", result)
	}
	var body []byte
	select {
	case body = <-captured:
	default:
		t.Fatal("request was not sent")
	}
	got, err := jsonjs.DecodeJSON(body)
	if err != nil {
		t.Fatal(err)
	}
	want, err := jsonjs.DecodeJSON(expected)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HTTP: %s\nPi: %s", body, expected)
	}
}

func TestPiSanitizeSurrogates(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-provider-text.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Sanitizer []struct {
			Name           string
			Text, Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Sanitizer) != 12 {
		t.Fatal("unexpected sanitizer coverage")
	}
	for _, tc := range fixture.Sanitizer {
		t.Run(tc.Name, func(t *testing.T) {
			input, err := jsonjs.DecodeJSON(tc.Text)
			if err != nil {
				t.Fatal(err)
			}
			want, err := jsonjs.DecodeJSON(tc.Expected)
			if err != nil {
				t.Fatal(err)
			}
			if got := SanitizeSurrogates(input.(string)); got != want {
				t.Fatalf("got %s, want %s", jsonjs.QuoteString(got), tc.Expected)
			}
		})
	}
	// Go can assemble the two preserved units without normalizing them to UTF-8.
	high, _ := jsonjs.DecodeJSON([]byte(`"\ud83d"`))
	low, _ := jsonjs.DecodeJSON([]byte(`"\ude80"`))
	if got := SanitizeSurrogates(high.(string) + low.(string)); got != "🚀" {
		t.Fatalf("split pair: %s", jsonjs.QuoteString(got))
	}
}
