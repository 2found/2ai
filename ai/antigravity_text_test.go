package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

func TestAntigravityProviderText(t *testing.T) {
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
	for _, tc := range fixture.Sanitizer {
		t.Run(tc.Name, func(t *testing.T) {
			value, err := jsonjs.DecodeJSON(tc.Text)
			if err != nil {
				t.Fatal(err)
			}
			expected, err := jsonjs.DecodeJSON(tc.Expected)
			if err != nil {
				t.Fatal(err)
			}
			text := value.(string)
			model := json.RawMessage(`{"id":"gemini-3-pro","api":"google-antigravity","provider":"google-antigravity","baseUrl":"https://fixture.example","input":["text"]}`)
			var parsed completionsModel
			if err := json.Unmarshal(model, &parsed); err != nil {
				t.Fatal(err)
			}
			signature := "c2ln"
			transcript := NormalizeContext(Context{Messages: []Message{
				{Role: "system", Content: TextContent(text)},
				{Role: "user", Content: TextContent(text)},
				{Role: "assistant", API: parsed.API, Provider: parsed.Provider, Model: parsed.ID, StopReason: "toolUse", Content: BlockContent(ContentBlock{Type: "text", Text: text}, ContentBlock{Type: "thinking", Thinking: text, ThinkingSignature: &signature}, ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)})},
				{Role: "toolResult", ToolCallID: "call", ToolName: "echo", Content: BlockContent(ContentBlock{Type: "text", Text: text})},
			}})
			token := OAuthToken{AccountID: "fixture", AccessToken: "fixture", ProjectID: "project"}
			payload, err := antigravityNativePayload(parsed, transcript, nil, token, 1)
			if err != nil {
				t.Fatal(err)
			}
			check := func(raw []byte) {
				t.Helper()
				decoded, err := jsonjs.DecodeJSON(raw)
				if err != nil {
					t.Fatal(err)
				}
				request := decoded.(map[string]any)["request"].(map[string]any)
				parts := func(value any) []any { return value.(map[string]any)["parts"].([]any) }
				texts := []any{}
				if text != "" {
					texts = append(texts, parts(request["systemInstruction"])[0].(map[string]any)["text"])
				}
				contents := request["contents"].([]any)
				texts = append(texts, parts(contents[0])[0].(map[string]any)["text"], parts(contents[1])[0].(map[string]any)["text"], parts(contents[1])[1].(map[string]any)["text"])
				response := parts(contents[2])[0].(map[string]any)["functionResponse"].(map[string]any)["response"].(map[string]any)
				texts = append(texts, response["output"])
				for _, got := range texts {
					if got != expected {
						t.Fatalf("provider text=%v, want %s; body=%s", got, tc.Expected, raw)
					}
				}
			}
			check(payload)
			if tc.Name != "mixed" {
				return
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			body := make(chan []byte, 1)
			client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				raw, err := io.ReadAll(request.Body)
				if err != nil {
					return nil, err
				}
				select {
				case body <- raw:
				default:
					return nil, fmt.Errorf("unexpected retry")
				}
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("data: " + `{"response":{"candidates":[{"finishReason":"STOP"}]}}` + "\n\n")), Request: request}, nil
			})}
			stream, err := StreamAntigravityPooled(ctx, model, transcript, OpenAICompletionsStreamOptions{Client: client}, &fakeTokenSource{tokens: []OAuthToken{token}})
			if err != nil {
				t.Fatal(err)
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
			select {
			case raw := <-body:
				check(raw)
			default:
				t.Fatal("no request")
			}
		})
	}
}
