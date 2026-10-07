package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
)

// AnthropicMessageClient is the Go equivalent of Pi's injected messaging client.
// It owns authentication, request encoding and endpoint adaptation. CreateResponse
// must honor ctx and return an unread response body, or an error. The provider
// closes returned bodies and owns callbacks, SSE decoding and accumulation.
type AnthropicMessageClient struct {
	CreateResponse func(context.Context, json.RawMessage, AnthropicRequestOptions) (*http.Response, error)
}

type AnthropicRequestOptions struct {
	// Raw JSON preserves absent/null/fractional timeout values. Validation belongs
	// to the injected client, just as it does for a prebuilt SDK client in Pi.
	TimeoutMS  json.RawMessage `json:"timeout,omitempty"`
	MaxRetries int             `json:"maxRetries"`
}

// AnthropicClientError retains the status/headers used by Pi's retry policy.
// Status zero denotes an unspecified status (for example a connection failure).
type AnthropicClientError struct {
	Status  int
	Headers http.Header
	Message string
}

func (e *AnthropicClientError) Error() string { return e.Message }

// StreamAnthropicWithClient skips built-in auth/client construction and OAuth
// request identity. A nil client selects the ordinary native HTTP provider.
// Simple mode intentionally does not accept this override, matching Pi.
func StreamAnthropicWithClient(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options AnthropicStreamOptions, client *AnthropicMessageClient) *AssistantMessageEventStream {
	if client != nil {
		copied := *client
		client = &copied
	}
	return streamAnthropic(ctx, rawModel, transcript, options, nil, client)
}

func runAnthropicClient(ctx context.Context, acc *anthropicAccumulator, rawModel json.RawMessage, transcript TranscriptContext, controls map[string]json.RawMessage, options AnthropicStreamOptions, client *AnthropicMessageClient) error {
	params, err := prepareAnthropicPayload(ctx, acc, rawModel, transcript, options, nil)
	if err != nil {
		return err
	}
	if client.CreateResponse == nil {
		return errors.New("Anthropic client has no CreateResponse callback")
	}
	requestOptions := AnthropicRequestOptions{TimeoutMS: controls["timeoutMs"], MaxRetries: 0}
	retries := 0.0
	_ = json.Unmarshal(controls["maxRetries"], &retries)
	var maxDelay *float64
	_ = json.Unmarshal(controls["maxRetryDelayMs"], &maxDelay)
	response, err := retryCompletionsRequest(ctx, func() (*http.Response, error) {
		response, err := client.CreateResponse(ctx, params, requestOptions)
		var provider *AnthropicClientError
		if errors.As(err, &provider) {
			err = &completionsRequestError{status: provider.Status, headers: provider.Headers, message: provider.Message}
		}
		return response, err
	}, retries, maxDelay, defaultCompletionsRetryTiming())
	if err != nil {
		return err
	}
	return consumeAnthropicResponse(ctx, acc, rawModel, options, response)
}
