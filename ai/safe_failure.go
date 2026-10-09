package ai

import (
	"context"
	"encoding/json"
	"net"
	"net/url"

	"github.com/2found/2ai/ai/protocol"
)

// Classify only structural errors. Provider text is never a classification key.
func nativeFailureKind(cause error, host bool) string {
	if host {
		return "host_callback"
	}
	if cause == nil {
		return "none"
	}
	if cause == context.Canceled {
		return "request_cancelled"
	}
	if cause == context.DeadlineExceeded {
		return "request_timeout"
	}
	status := -1
	// Never invoke arbitrary Error/Is/As/Unwrap methods to observe a failure.
	// Unknown custom wrappers remain unknown; admission/retry retains its own
	// existing authoritative error behavior.
	switch e := cause.(type) {
	case *PreparationError:
		return "host_preparation"
	case *codexProviderCallbackError:
		return "host_callback"
	case *protocol.ProviderError:
		if e != nil {
			status = e.Status
		}
	case *completionsRequestError:
		if e != nil {
			status = e.status
		}
	case *codexHTTPError:
		if e != nil {
			status = e.Status
		}
	case *AnthropicClientError:
		if e != nil {
			status = e.Status
		}
	case *PiMessagesResponseError:
		if e != nil {
			status = e.Status
		}
	case *url.Error:
		return "provider_transport"
	case *codexProtocolError:
		return "provider_protocol"
	case *codexAPIError:
		return "provider_rejected"
	case *json.SyntaxError, *json.UnmarshalTypeError:
		return "provider_decode"
	case net.Error:
		return "provider_transport"
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
	return "unknown"
}
