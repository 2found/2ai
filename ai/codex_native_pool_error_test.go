package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

func TestCodexPoolFailurePreservesHTTPMetadataAndWireMessage(t *testing.T) {
	for _, status := range []int{401, 403, 429, 500} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			key := "h." + base64.StdEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_account_id":"a"}}`)) + ".s"
			controls, _ := json.Marshal(map[string]any{"apiKey": key, "transport": "sse"})
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: status, Status: http.StatusText(status), Header: http.Header{"Retry-After": []string{"7"}}, Body: io.NopCloser(strings.NewReader(`{"error":{"message":"rejected"}}`))}, nil
			})}
			var cause error
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			stream := streamCodexResponsesObserved(ctx, json.RawMessage(`{"id":"test","api":"openai-codex-responses","provider":"openai-codex"}`), NormalizeContext(Context{}), CodexResponsesStreamOptions{Options: controls, Client: client}, true, func(err error) { cause = err })
			result, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			expected := "rejected"
			if status == 429 {
				expected = "You have hit your ChatGPT usage limit."
			}
			if result.ErrorMessage == nil || *result.ErrorMessage != expected {
				t.Fatalf("Pi message changed: %+v", result)
			}
			reported := codexPoolFailure(cause)
			var typed *protocol.ProviderError
			if !errors.As(reported, &typed) || typed.Status != status || typed.RetryAfter != 7*time.Second || typed.Provider != VendorOpenAICodex || typed.Message != expected {
				t.Fatalf("pool metadata lost: %#v", reported)
			}
			if isOAuthAuthFailure(reported) != (status == 401 || status == 403) {
				t.Fatalf("wrong auth rotation classification: %v", reported)
			}
		})
	}
}
func TestCodexPoolDoesNotGuessAuthenticationFromText(t *testing.T) {
	for _, cause := range []error{errors.New("401 unauthorized"), &codexProviderCallbackError{errors.New("unauthorized")}, &codexAPIError{Code: "invalid_request_error", Message: "401 unauthorized"}} {
		if got := codexPoolFailure(cause); got.Error() != cause.Error() || isOAuthAuthFailure(got) {
			t.Fatalf("untyped failure became auth rejection: %v", got)
		}
	}
	concurrency := codexPoolFailure(&codexHTTPError{Status: 403, Message: "concurrency cap"})
	if isOAuthAuthFailure(concurrency) || !isOAuthConcurrencyCap(concurrency) {
		t.Fatal("transient concurrency cap would rotate account")
	}
}
