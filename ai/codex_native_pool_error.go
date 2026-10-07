package ai

import (
	"errors"
	"net/http"

	"github.com/2found/2ai/ai/protocol"
)

// Native stream messages retain Pi's wire format. Pool reporting separately
// needs a typed status and retry hint; never infer an auth failure from text.
func codexPoolFailure(cause error) error {
	if cause == nil {
		return nil
	}
	if codexNonTransportError(cause) {
		return errors.New(cause.Error())
	}
	var httpError *codexHTTPError
	if errors.As(cause, &httpError) {
		return protocol.NewProviderError(VendorOpenAICodex, &http.Response{StatusCode: httpError.Status, Header: httpError.Headers}, httpError.Message)
	}
	return cause
}
