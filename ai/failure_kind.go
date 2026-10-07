package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net"

	"github.com/2found/2ai/ai/protocol"
)

// nativeFailureKind projects structural error metadata, never provider text.
// It does not decide replay safety or change retry policy.
func nativeFailureKind(cause error, callback bool) string {
	if cause == nil {
		return "none"
	}
	if callback {
		return "host_callback"
	}
	var callbackError *codexProviderCallbackError
	var preparation *PreparationError
	if errors.As(cause, &preparation) {
		return "host_preparation"
	}
	if errors.As(cause, &callbackError) {
		return "host_callback"
	}
	if errors.Is(cause, context.Canceled) {
		return "request_cancelled"
	}
	if errors.Is(cause, context.DeadlineExceeded) {
		return "request_timeout"
	}
	var provider *protocol.ProviderError
	var openAI *completionsRequestError
	var codex *codexHTTPError
	var anthropic *AnthropicClientError
	var messages *PiMessagesResponseError
	status := -1
	switch {
	case errors.As(cause, &provider):
		status = provider.Status
	case errors.As(cause, &openAI):
		status = openAI.status
	case errors.As(cause, &codex):
		status = codex.Status
	case errors.As(cause, &anthropic):
		status = anthropic.Status
	case errors.As(cause, &messages):
		status = messages.Status
	}
	switch {
	case status == 0:
		return "provider_transport"
	case status == 401 || status == 403:
		return "provider_auth"
	case status == 429:
		return "provider_rate_limit"
	case status >= 500 && status <= 599:
		return "provider_unavailable"
	case status >= 400 && status <= 499:
		return "provider_rejected"
	}
	var wire *codexProtocolError
	var api *codexAPIError
	var syntax *json.SyntaxError
	var jsonType *json.UnmarshalTypeError
	var transport net.Error
	switch {
	case errors.As(cause, &wire):
		return "provider_protocol"
	case errors.As(cause, &api):
		return "provider_rejected"
	case errors.As(cause, &syntax), errors.As(cause, &jsonType):
		return "provider_decode"
	case errors.As(cause, &transport):
		return "provider_transport"
	default:
		return "unknown"
	}
}

func boundedFailureKind(kind string) string {
	switch kind {
	case "none", "host_callback", "host_preparation", "request_cancelled", "request_timeout", "provider_transport", "provider_auth", "provider_rate_limit", "provider_unavailable", "provider_rejected", "provider_protocol", "provider_decode", "unknown":
		return kind
	}
	return "unknown"
}
